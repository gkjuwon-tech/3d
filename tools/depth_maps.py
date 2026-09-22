#!/usr/bin/env python3
"""Recover a depth map per view from its normals and save it.

carve_depth integrates and carves in one pass and throws the depth away, so
every experiment on how depth is fused paid ten minutes of integration first.
This keeps the integration and writes one float32 map per view, NaN off the
silhouette, so fusion can be iterated on its own.

Run:
  python3 tools/depth_maps.py --views refs/lucy_gt --hull-views out/final_views \
      --normals-dir out/ps_normals --out out/depth_ps
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carve_depth import recover_depth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--hull-views", required=True)
    ap.add_argument("--normals-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--w-inner", type=float, default=0.2)
    ap.add_argument("--anchor-blur", type=float, default=8.0)
    ap.add_argument("--nz-floor", type=float, default=0.35)
    ap.add_argument("--maxiter", type=int, default=4000)
    ap.add_argument("--only-views", default=None)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    views = args.only_views.split(",") if args.only_views else list(meta["views"])
    os.makedirs(args.out, exist_ok=True)
    for v in views:
        t0 = time.time()
        d, hit, hull, gt = recover_depth(args.views, args.hull_views, v, meta,
                                         args.budget, args.w_inner,
                                         args.anchor_blur, args.nz_floor,
                                         args.maxiter, args.normals_dir)
        d = np.where(hit, d, np.nan).astype(np.float32)
        np.save(os.path.join(args.out, f"{v}.npy"), d)
        note = ""
        if gt is not None:
            m = hit & np.isfinite(d) & (gt < 1e9)
            note = (f"  MAE {np.abs(d[m]-gt[m]).mean():.5f}"
                    f"  hull {np.abs(hull[m]-gt[m]).mean():.5f}")
        print(f"  {v:<13}{note}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
