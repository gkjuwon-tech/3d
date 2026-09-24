#!/usr/bin/env python3
"""Per-view depth from one view's normals alone (no cross-view matching).

Generated views disagree in their detail, so matching normals across views
finds almost nothing (the cat: 110-3,800 anchors per view, Lucy: ~150k). Each
view's normal map on its own, though, integrates into a convincing relief. The
relief is placed against the silhouette hull: a screened integration toward the
hull's depth at scales above --scale pixels, then fused like any depth map
(tools/fuse_field.py).

    python3 tools/genlab/relief_depth.py --views data/catgen/views \
        --normals data/catgen/normals --hull data/catfuse/hull --out data/catfuse/depth
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from detail_depth import solve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--normals", required=True)
    ap.add_argument("--hull", required=True, help="hull_field.py output (views/depth_npy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=200.0)
    ap.add_argument("--cut-px", type=float, default=6.0)
    ap.add_argument("--behind", type=float, default=0.25,
                    help="quantile of (relief - hull) set to zero: the hull contains the "
                         "object, so most of the relief should lie behind its front")
    a = ap.parse_args()
    meta = json.load(open(f"{a.views}/cameras.json"))
    os.makedirs(a.out, exist_ok=True)
    for v in meta["views"]:
        m = np.asarray(Image.open(f"{a.views}/mask/{v}.png").convert("L")) > 127
        res = m.shape[0]
        px = meta["ortho_scale"] / res
        hd = np.load(f"{a.hull}/views/depth_npy/{v}.npy").astype(np.float64)
        if hd.shape[0] != res:
            hd = np.asarray(Image.fromarray(hd.astype(np.float32)).resize((res, res), Image.NEAREST))
        ok = m & np.isfinite(hd) & (hd < 1e3)
        mm = ndimage.binary_erosion(ok, iterations=1)
        n = np.load(f"{a.normals}/{v}.npy").astype(np.float64)
        z0 = np.where(mm, hd, 0)
        z = solve(z0, mm, n, px, a.scale, a.cut_px)
        off = np.quantile((z - hd)[mm], a.behind)
        z = z - off
        np.save(f"{a.out}/{v}.npy", z.astype(np.float32))
        print(f"{v:10} {mm.sum():,} px; relief vs hull front: median {np.median((z-hd)[mm])/px:+.1f} px,"
              f" {100*((z-hd)[mm] < 0).mean():.0f}% in front of it", flush=True)


if __name__ == "__main__":
    main()
