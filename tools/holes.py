#!/usr/bin/env python3
"""Measure holes: ground-truth surface the reconstruction left outside itself.

A hole is material that exists and was carved away. Every ground-truth
surface sample is looked up in the fused field (trilinear, voxel units, >0
outside); one that reads more than `tol` voxels outside is a hole sample.
Reported whole-model and per named region (eval_surface.REGIONS), and saved
for tools/blame.py to trace back to the views that emptied it.

Run:
  python3 tools/holes.py --field out/fused_field.npy --grid out/hull/grid.npz \
      --save out/holes.npz
"""
import argparse
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_surface import REGIONS, gt_samples  # noqa: E402


def hole_report(F, lo, h, gp, tol, regions=True):
    idx = ((gp - lo) / h - 0.5).T
    f = ndimage.map_coordinates(F, idx, order=1, mode="nearest")
    hole = f > tol
    out = {"all": (100 * hole.mean(), int(hole.sum()))}
    if regions:
        for name, (c, r) in REGIONS.items():
            m = np.linalg.norm(gp - np.array(c), axis=1) < r
            if m.sum():
                out[name] = (100 * hole[m].mean(), int(hole[m].sum()))
    return out, hole, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True, nargs="+")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--gt-mesh", default="assets/lucy_le.ply")
    ap.add_argument("--views", default="refs/lucy_gt")
    ap.add_argument("--cache", default="data/lucy_v2/gt_samples.npz")
    ap.add_argument("--samples", type=int, default=2_000_000)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    z = np.load(args.grid)
    lo, h = z["lo"], float(z["h"])
    gp, gn = gt_samples(args.gt_mesh, args.views, args.samples, args.cache)
    names = ["all"] + list(REGIONS)
    print(f"{'field':<34}" + "".join(f"{n:>12}" for n in names))
    for path in args.field:
        F = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        rep, hole, f = hole_report(F, lo, h, gp, args.tol)
        print(f"{os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path):<34}"
              + "".join(f"{rep[n][0]:>11.2f}%" if n in rep else f"{'-':>12}" for n in names))
        if args.save:
            np.savez(args.save, p=gp[hole], n=gn[hole], f=f[hole])


if __name__ == "__main__":
    main()
