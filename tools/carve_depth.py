#!/usr/bin/env python3
"""Carve the hull with depth recovered from normals, at image resolution.

The earlier attempt integrated at the voxel grid's resolution and carved in the
grid's own frame. Both were wrong. Block-averaging the normals to 512 destroys
their agreement with the surface -- the same integration that lands 30% below
the hull at 2048 lands above it at 512 -- and depth recovered per view belongs
in the frame it was rendered in.

So: integrate at the rendered resolution, then project voxels into that frame
to carve. A voxel goes only if it sits in front of the recovered surface by
more than the margin, and nothing is ever added, so the failure mode is a model
too thin rather than a model that explodes.

Run:
  python3 tools/carve_depth.py --occ out/final_mesh_occ.npz \
      --views refs/lucy_gt --hull-views out/final_views --out out/carved
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
from PIL import Image

BG = 1e9
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)


def _load(name):
    spec = importlib.util.spec_from_file_location(name,
                                                  os.path.join(_here, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


iz = _load("integrate_normals")
dc = _load("discontinuity")
vh = _load("visual_hull")


def recover_depth(views_dir, hull_dir, view, meta, budget, w_inner,
                  anchor_blur, nz_floor, maxiter, normals_dir=None):
    gt_path = os.path.join(views_dir, "depth_npy", f"{view}.npy")
    hull = np.load(os.path.join(hull_dir, "depth_npy", f"{view}.npy"))
    if normals_dir:
        # already camera-space, e.g. solved by photometric stereo
        n = np.load(os.path.join(normals_dir, f"{view}.npy")).astype(np.float64)
    else:
        n = np.load(os.path.join(views_dir, "normal_npy", f"{view}.npy"))
        R = np.array(meta["views"][view]["matrix_world"])[:3, :3]
        n = (n.reshape(-1, 3) @ R).reshape(n.shape)
    n /= np.linalg.norm(n, axis=2, keepdims=True).clip(1e-9)
    rgb = np.asarray(Image.open(os.path.join(views_dir, "rgb", f"{view}.png")
                                ).convert("L"), dtype=np.float64) / 255.0
    hull = np.where(hull < BG, hull, np.nan)
    hit = np.isfinite(hull)
    px = meta["ortho_scale"] / hull.shape[0]

    nz = n[..., 2]
    ok = (np.abs(nz) > nz_floor) & hit
    gx = np.where(ok, (n[..., 0] / np.where(ok, nz, 1)) * px, 0.0)
    gy = np.where(ok, -(n[..., 1] / np.where(ok, nz, 1)) * px, 0.0)

    ew = None
    if budget > 0:
        w = {}
        for axis, name in ((0, "y"), (1, "x")):
            det, valid = dc.detectors(n, rgb, hull, hit, axis)
            s = np.zeros_like(det["graze"])
            for key in ("graze", "turn"):
                s = np.maximum(s, dc.rank_normalise(det[key], valid))
            thr = np.percentile(s[valid], 100 - budget)
            w[name] = dc.pad_to(np.where(valid & (s >= thr), 0.0, 1.0),
                                hit.shape, axis)
        ew = (w["y"], w["x"])

    z, info, _ = iz.solve_view(hull, hit, gx, gy, ok, w_rim=1e5,
                               w_inner=w_inner, w_grad=1.0, w_smooth=0.0,
                               anchor_blur=anchor_blur, tol=1e-8,
                               maxiter=maxiter, pixel_size=px, edge_w=ew)
    floor = np.where(np.isfinite(hull), hull, -np.inf)
    z = np.maximum(z, floor)
    gt = np.load(gt_path) if os.path.exists(gt_path) else None
    return z, hit, hull, gt


def carve_view(occ, centers, meta, view, depth, margin, slab=24, votes=None):
    """Mark voxels in front of the recovered surface, in the view's frame.

    With `votes`, nothing is removed here: each view adds one vote per voxel it
    considers empty, and removal waits for a quorum. A carve is an intersection,
    so a single view's worst few percent would otherwise decide those voxels for
    everyone -- which is what shredded one side of the first result while the
    other side came out clean. Per-view errors land in different places, which
    is exactly what a quorum exploits.
    """
    info = meta["views"][view]
    m = np.array(info["matrix_world"], dtype=np.float64)
    right, up, back, loc = m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3]
    ortho = meta["ortho_scale"]
    res = depth.shape[0]
    d = np.where(np.isfinite(depth), depth, -np.inf)   # unknown carves nothing

    nx, ny, nz_ = occ.shape
    ux = right[0] * centers[0] - float(right @ loc)
    uy, uz = right[1] * centers[1], right[2] * centers[2]
    vx = up[0] * centers[0] - float(up @ loc)
    vy, vz = up[1] * centers[1], up[2] * centers[2]
    fx = -back[0] * centers[0] + float(back @ loc)
    fy, fz = -back[1] * centers[1], -back[2] * centers[2]

    removed = 0
    for i0 in range(0, nx, slab):
        i1 = min(i0 + slab, nx)
        u = ux[i0:i1, None, None] + uy[None, :, None] + uz[None, None, :]
        v = vx[i0:i1, None, None] + vy[None, :, None] + vz[None, None, :]
        w = fx[i0:i1, None, None] + fy[None, :, None] + fz[None, None, :]
        col = np.clip(((u / ortho + 0.5) * res).astype(np.int32), 0, res - 1)
        row = np.clip(((0.5 - v / ortho) * res).astype(np.int32), 0, res - 1)
        front = w < (d[row, col] - margin)
        if votes is None:
            removed += int((occ[i0:i1] & front).sum())
            occ[i0:i1] &= ~front
        else:
            votes[i0:i1] += front
            removed += int((occ[i0:i1] & front).sum())
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--hull-views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin-voxels", type=float, default=2.0)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--w-inner", type=float, default=0.2)
    ap.add_argument("--anchor-blur", type=float, default=8.0)
    ap.add_argument("--nz-floor", type=float, default=0.15)
    ap.add_argument("--maxiter", type=int, default=4000)
    ap.add_argument("--normals-dir", default=None,
                   help="camera-space normals per view; default is the view "
                        "set's own ground-truth normal pass")
    ap.add_argument("--only-views", default=None)
    ap.add_argument("--quorum", type=int, default=0,
                   help="views that must independently agree a voxel is empty "
                        "before it is removed; 0 carves on any single view")
    ap.add_argument("--min-touch", type=float, default=50.0,
                   help="a view carves only if its solution rests on the hull "
                        "floor over at least this percent of its pixels. "
                        "Measured without ground truth, this ranks the six "
                        "canonical views in exactly the order their true "
                        "accuracy does: a solve consistent with a hull that is "
                        "tight in places sits on it, and a solve that has "
                        "wandered floats above it everywhere.")
    args = ap.parse_args()

    z = np.load(args.occ)
    dims = z["dims"]
    occ = np.unpackbits(z["occ"])[:int(np.prod(dims))].reshape(dims).astype(bool)
    lo, h = z["lo"], float(z["h"])
    centers = [lo[a] + (np.arange(dims[a]) + 0.5) * h for a in range(3)]
    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    margin = args.margin_voxels * h

    views = list(meta["views"])
    if args.only_views:
        views = args.only_views.split(",")
    votes = np.zeros(occ.shape, dtype=np.uint8) if args.quorum > 0 else None
    start = int(occ.sum())
    print(f"grid {tuple(dims)}  {start:,} voxels  margin {margin:.5f} "
          f"({args.margin_voxels} voxels)")

    for v in views:
        t0 = time.time()
        d, hit, hull, gt = recover_depth(args.views, args.hull_views, v, meta,
                                         args.budget, args.w_inner,
                                         args.anchor_blur, args.nz_floor,
                                         args.maxiter, args.normals_dir)
        note = ""
        if gt is not None:
            m = hit & np.isfinite(d) & (gt < BG)
            base = np.abs(hull[m] - gt[m]).mean()
            mae = np.abs(d[m] - gt[m]).mean()
            over = 100 * ((d[m] - margin) > gt[m]).mean()
            note = (f"  MAE {mae:.5f} vs hull {base:.5f} "
                    f"({100*(1-mae/base):+.1f}%)  over {over:.2f}%")
        m_all = hit & np.isfinite(d) & np.isfinite(hull)
        touch = 100.0 * np.mean(np.abs(d[m_all] - hull[m_all]) < 1e-9)
        if touch < args.min_touch:
            print(f"  {v:<13} SKIPPED   floor contact {touch:.1f}% < "
                  f"{args.min_touch}%{note}")
            continue
        marked = carve_view(occ, centers, meta, v, d, margin, votes=votes)
        verb = "voted" if votes is not None else "removed"
        print(f"  {v:<13} {verb} {marked:>9,}  contact {touch:.1f}%{note}"
              f"  {time.time()-t0:.0f}s")

    if votes is not None:
        hist = np.bincount(votes[occ].ravel(), minlength=16)[:8]
        print("  votes per occupied voxel: "
              + "  ".join(f"{i}:{c:,}" for i, c in enumerate(hist) if c))
        occ &= votes < args.quorum
    print(f"\nremoved {start - occ.sum():,} of {start:,} "
          f"({100*(start-occ.sum())/start:.2f}%)  remaining {occ.sum():,}")
    np.savez_compressed(args.out + "_occ.npz", occ=np.packbits(occ),
                        dims=dims, lo=lo, h=h)
    from scipy import ndimage
    from skimage import measure
    padded = ndimage.gaussian_filter(np.pad(occ.astype(np.float32), 1), 1.0,
                                     truncate=3.0)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5,
                                                spacing=(h, h, h))
    verts += lo - h
    vh.write_ply(args.out + ".ply", verts.astype(np.float32),
                 faces.astype(np.int32))
    print(f"wrote {args.out}.ply  {len(faces):,} faces")


if __name__ == "__main__":
    main()
