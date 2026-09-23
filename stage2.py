#!/usr/bin/env python3
"""Stage 2 -- images in, a 3D mesh out. No mesh, no learned model, no GPU.

Reads only data/<name>/views/{mask,rgb,cameras.json} and data/<name>/normals.

  1. hull     visual hull as an exact distance field from the silhouettes,
              and its depth per view by sphere tracing (tools/hull_field.py)
  2. depth    per view: normals integrated robustly, placed in depth by
              matching normals across the other 13 views, re-solved against
              those anchors (tools/depth_mv.py), several views at a time
  3. fuse     hull field and depth maps fused into one continuous signed
              distance and meshed at its zero level (tools/fuse_field.py)
  4. retopo   QuadriFlow quad base plus a subdivided, shrink-wrapped quad
              mesh carrying the detail (tools/retopo.py)
  5. score    only if --gt-mesh is given: Chamfer, containment, volume,
              F-score and normal error against the mesh stage 1 rendered,
              and showcase renders of both

Every step is skipped when its output already exists, so an interrupted run
resumes where it stopped; --force redoes everything.

Run:
  python3 stage2.py --name bunny --gt-mesh assets/bunny.ply
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
BLENDER = os.path.join(ROOT, "assets", "blender", "blender")
PY = sys.executable


def run(cmd, log, parallel=False):
    print(f"$ {' '.join(cmd)}", flush=True)
    f = open(log, "a")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if parallel:
        return p
    if p.wait():
        sys.exit(f"failed ({p.returncode}); see {log}")


def tool(name):
    return os.path.join(ROOT, "tools", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--views", default=None, help="override data/<name>/views")
    ap.add_argument("--normals", default=None, help="override data/<name>/normals")
    ap.add_argument("--gt-mesh", default=None, help="score against this mesh")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--quads", type=int, default=40000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    d = os.path.join(a.data, a.name)
    views = a.views or os.path.join(d, "views")
    normals = a.normals or os.path.join(d, "normals")
    rec = os.path.join(d, "recon")
    hull, depth = os.path.join(rec, "hull"), os.path.join(rec, "depth")
    os.makedirs(depth, exist_ok=True)
    log = os.path.join(rec, "stage2.log")
    meta = json.load(open(os.path.join(views, "cameras.json")))
    names = list(meta["views"])
    T = {}
    t_all = time.time()

    # 1. hull -----------------------------------------------------------------
    t0 = time.time()
    if a.force or not os.path.exists(os.path.join(hull, "views", "depth_npy",
                                                   f"{names[-1]}.npy")):
        run([PY, tool("hull_field.py"), "--views", views, "--out", hull], log)
    T["hull"] = time.time() - t0

    # 2. depth, several views at once ------------------------------------------
    t0 = time.time()
    todo = [v for v in names if a.force or
            not os.path.exists(os.path.join(depth, f"{v}_cost.npy"))]
    chunks = [todo[i::a.workers] for i in range(a.workers)]
    procs = [run([PY, tool("depth_mv.py"), "--views", views,
                  "--hull-views", os.path.join(hull, "views"),
                  "--normals-dir", normals, "--out", depth,
                  "--only-views", ",".join(c)], log, parallel=True)
             for c in chunks if c]
    for p in procs:
        if p.wait():
            sys.exit(f"depth worker failed; see {log}")
    T["depth"] = time.time() - t0

    # 3. fuse -----------------------------------------------------------------
    t0 = time.time()
    mesh = os.path.join(rec, "mesh")
    if a.force or not os.path.exists(mesh + ".ply"):
        run([PY, tool("fuse_field.py"), "--occ", os.path.join(hull, "grid.npz"),
             "--views", views, "--depth-dir", depth, "--normals-dir", normals,
             "--hull-cache", os.path.join(hull, "H.npy"), "--out", mesh], log)
    T["fuse"] = time.time() - t0

    # 4. retopo ---------------------------------------------------------------
    t0 = time.time()
    rdir = os.path.join(rec, "retopo")
    if a.force or not os.path.exists(os.path.join(rdir, f"{a.name}_quads_detail.obj")):
        run([BLENDER, "-b", "-P", tool("retopo.py"), "--", "--mesh", mesh + ".ply",
             "--out-dir", rdir, "--name", a.name, "--faces", str(a.quads)], log)
    T["retopo"] = time.time() - t0

    # 5. score ----------------------------------------------------------------
    if a.gt_mesh:
        t0 = time.time()
        rep = os.path.join(rec, "report.txt")
        with open(rep, "w") as f:
            for target in (os.path.join(hull, "hull.ply"), mesh + ".ply"):
                f.write(f"== {os.path.basename(target)}\n")
                f.flush()
                subprocess.run([PY, tool("eval_hull.py"), "--gt-mesh", a.gt_mesh,
                                "--views", views, "--recon", target],
                               cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            f.flush()
            subprocess.run([PY, tool("eval_surface.py"), "--gt-mesh", a.gt_mesh,
                            "--views", views, "--cache",
                            os.path.join(d, "gt_samples.npz"),
                            "--regions", "lucy" if a.name == "lucy" else "none",
                            "--recon", os.path.join(hull, "hull.ply"), mesh + ".ply"],
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            f.flush()
            subprocess.run([PY, tool("containment.py"), "--field",
                            mesh + "_field.npy", "--occ", os.path.join(hull, "grid.npz"),
                            "--gt-mesh", a.gt_mesh, "--views", views, "--cache",
                            os.path.join(d, "gt_samples.npz")],
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        for tag, target in (("gt", a.gt_mesh), ("hull", os.path.join(hull, "hull.ply")),
                            ("recon", mesh + ".ply")):
            out = os.path.join(rec, "show", tag)
            if not os.path.exists(os.path.join(out, "d_low_hero.png")):
                run([BLENDER, "-b", "-P", tool("showcase.py"), "--", "--mesh",
                     os.path.abspath(target), "--out", out, "--res", "900",
                     "--samples", "64"], log)
        T["score"] = time.time() - t0
        print(open(rep).read())
    print("timings: " + "  ".join(f"{k} {v/60:.1f}min" for k, v in T.items())
          + f"  total {(time.time()-t_all)/60:.1f}min")


if __name__ == "__main__":
    main()
