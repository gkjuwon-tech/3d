#!/usr/bin/env python3
"""Render -> redraw -> integrate, until the generated views agree.

Round r: render mesh M_r's normals from the generated cameras; show each
two-panel sheet (opposite views) to the image model next to the generated
appearance sheet and ask for the sculpture's normals drawn ON the current model
(it copies what it is given very faithfully: E7, 4.5 deg); integrate those
normals into M_r (mesh_from_normals.py) to get M_{r+1}. Every view now starts
from the same mesh, so the views stop being eight different cats.

    python3 tools/genlab/refine_loop.py --mesh data/cat/mfn/mesh.ply \
        --app data/catgen/views --out data/cat/loop --rounds 3 \
        --style data/cat/gen/sheet1_a.png data/cat/gen/sheet2_d.png
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from gen_image import generate  # noqa: E402
from paint import two_panel, panels, enc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER = os.path.join(ROOT, "assets", "blender", "blender")
PAIRS = [("01_front", "03_back"), ("02_right", "04_left"), ("d315", "d135"), ("d045", "d225")]

PROMPT = """Make the surface-normal sheet of the sculpture, drawn on its current 3D model.

First attached image: a sheet of two panels, the target sculpture seen by two
orthographic cameras. It defines what the sculpture looks like.
Second attached image: the matching sheet of normal maps rendered from our current 3D
model of it, panel for panel, same cameras, same framing. Its placement and its 3D
structure are real and shared by every camera, but its forms are lumpy and unfinished.

Draw, in each panel, the normal map of the target sculpture built on the current model:
keep the model's outline, placement and large 3D structure wherever they agree with
the first sheet; correct the forms that are wrong or lumpy so they match the first
sheet (the head, face, ears, arms, hands, bodice, skirt folds, lace, tail, flames);
add all the fine detail.

Encoding: each panel in its own camera frame, R = (x+1)/2, G = (y+1)/2, B = (z+1)/2 with
x right, y up, z toward that panel's viewer; surfaces facing the camera are light
purple-blue (128,128,255). Black background, white gutter with black registration
squares. It must be a normal map, not a picture: no lighting, no shadows. Keep the
sheet's wide 2:1 proportions and the exact framing of the second sheet."""

STYLE = """

Any further attached images show the target sculpture from other cameras (its
turnaround): follow their details and style."""


def render_normals(mesh, out, margin):
    """camera-space normal maps of mesh in the ring views (Blender geometry pass)"""
    shutil.rmtree(out, ignore_errors=True)
    log = open(out + ".log", "w")
    subprocess.run([BLENDER, "-b", "-P", os.path.join(ROOT, "tools", "render_orthoviews.py"), "--",
                    "--mesh", mesh, "--out", out, "--res", "1024", "--samples", "8",
                    "--no-normalize", "--margin", str(margin), "--aux", "ring4"],
                   stdout=log, stderr=log, check=True)
    subprocess.run([BLENDER, "-b", "-P", os.path.join(ROOT, "tools", "render_orthoviews.py"), "--",
                    "--mesh", mesh, "--out", out, "--res", "1024", "--samples", "8",
                    "--no-normalize", "--margin", str(margin), "--aux", "diagonal8", "--aux-only"],
                   stdout=log, stderr=log, check=True)
    subprocess.run([BLENDER, "-b", "-P", os.path.join(ROOT, "tools", "exr_to_npy.py"), "--",
                    "--dir", out], stdout=log, stderr=log, check=True)
    meta = json.load(open(f"{out}/cameras.json"))
    res = {}
    for v in meta["views"]:
        n = np.load(f"{out}/normal_npy/{v}.npy").astype(np.float64)
        ok = np.linalg.norm(n, axis=-1) > 0.5
        R = np.array(meta["views"][v]["matrix_world"])[:3, :3]
        nc = n @ R; nc[~ok] = 0
        res[v] = (nc, ok)
    for sub in ("depth", "normal", ".unused", "depth_npy"):
        shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
    return res


def taubin(v, f, iters, lam=0.5, mu=-0.53):
    """volume-preserving smoothing: keeps the large forms, drops the blotches"""
    from scipy import sparse
    n = len(v)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    for _ in range(iters):
        for k in (lam, mu):
            v = v + k * (A @ v / deg[:, None] - v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--app", required=True, help="generated view set: rgb/, mask/, cameras.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--style", nargs="*", default=[])
    ap.add_argument("--iters", type=int, default=6, help="mesh_from_normals iterations per round")
    ap.add_argument("--pairs", default=None, help="a:b,c:d,... opposite-view sheets")
    ap.add_argument("--presmooth", type=int, default=0, help="Taubin iterations on the input mesh")
    a = ap.parse_args()
    meta = json.load(open(f"{a.app}/cameras.json"))
    os.makedirs(a.out, exist_ok=True)
    global PAIRS
    if a.pairs:
        PAIRS = [tuple(p.split(":")) for p in a.pairs.split(",")]
    mesh = a.mesh
    if a.presmooth:
        from eval_hull import read_ply
        from displace import write_ply
        v, f = read_ply(mesh)
        v = taubin(v.astype(np.float64), f.astype(np.int64), a.presmooth)
        mesh = f"{a.out}/start.ply"
        write_ply(mesh, v.astype(np.float32), f.astype(np.int32))
    for r in range(a.rounds):
        rd = f"{a.out}/r{r}"
        os.makedirs(f"{rd}/normals", exist_ok=True)
        if os.path.exists(f"{rd}/mesh.ply"):
            mesh = f"{rd}/mesh.ply"
            continue
        cur = render_normals(os.path.abspath(mesh), os.path.abspath(f"{rd}/render"), meta["ortho_scale"])
        jobs = []
        for k, pair in enumerate(PAIRS):
            ref = [f"{rd}/s{k}_app.png", f"{rd}/s{k}_cur.png"]
            two_panel([Image.open(f"{a.app}/rgb/{v}.png") for v in pair]).save(ref[0])
            two_panel([enc(cur[v][0], cur[v][1]) for v in pair]).save(ref[1])
            out = f"{rd}/s{k}_gen.png"
            if not os.path.exists(out):
                jobs.append((PROMPT + (STYLE if a.style else ""), out, ref + list(a.style)))
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda j: print(f"  r{r} wrote {os.path.basename(j[1])} ({generate(*j):.0f}s)",
                                        flush=True), jobs))
        for k, pair in enumerate(PAIRS):
            for v, pnl in zip(pair, panels(Image.open(f"{rd}/s{k}_gen.png"))):
                g = np.asarray(pnl.resize((1024, 1024), Image.LANCZOS), np.float64)
                n = g / 255 * 2 - 1
                n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-9)
                m = np.asarray(Image.open(f"{a.app}/mask/{v}.png").convert("L")) > 127
                n[~m] = 0
                np.save(f"{rd}/normals/{v}.npy", n.astype(np.float32))
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "mesh_from_normals.py"),
                        "--mesh", mesh, "--views", a.app, "--normals", f"{rd}/normals",
                        "--out", f"{rd}/mesh.ply", "--iters", str(a.iters)], check=True)
        mesh = f"{rd}/mesh.ply"
        print(f"round {r} done -> {mesh}", flush=True)


if __name__ == "__main__":
    main()
