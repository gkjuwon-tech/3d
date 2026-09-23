#!/usr/bin/env python3
"""Containment measured on the continuous field, with how far outside.

eval_hull tests containment on the occupancy grid: a ground-truth surface
point counts as contained if the voxel it falls in is occupied. That was
right for a hull dilated by a voxel footprint. For a surface that is tight
against the truth it is not: a point exactly on the surface lands in a voxel
whose centre is outside half the time, and the grid calls it a violation.
Here the fused field is sampled trilinearly at each ground-truth point, and
the report says how far outside the violations are, in voxels.

Run:
  python3 tools/containment.py --field out/fused_field.npy --occ out/fused_occ.npz
"""
import argparse
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_surface import gt_samples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True, nargs="+")
    ap.add_argument("--occ", required=True, help="grid definition")
    ap.add_argument("--gt-mesh", default="assets/lucy_le.ply")
    ap.add_argument("--views", default="refs/lucy_gt")
    ap.add_argument("--cache", default="out/gt_samples.npz")
    ap.add_argument("--samples", type=int, default=2_000_000)
    args = ap.parse_args()
    z = np.load(args.occ)
    lo, h = z["lo"], float(z["h"])
    gp, _ = gt_samples(args.gt_mesh, args.views, args.samples, args.cache)
    idx = ((gp - lo) / h - 0.5).T               # continuous voxel index
    for path in args.field:
        F = np.load(path, mmap_mode="r")
        F = np.asarray(F, dtype=np.float32)
        f = ndimage.map_coordinates(F, idx, order=1, mode="nearest")
        print(f"{os.path.basename(path)}")
        for t in (0.0, 0.5, 1.0, 2.0, 3.0):
            print(f"  within {t:3.1f} vox of inside: {100*(f <= t).mean():8.4f}%")
        out = f[f > 0]
        if len(out):
            print(f"  outside: {100*len(out)/len(f):.3f}%  median {np.median(out):.2f} "
                  f"p95 {np.percentile(out,95):.2f} max {out.max():.2f} vox")


if __name__ == "__main__":
    main()
