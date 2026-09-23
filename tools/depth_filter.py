#!/usr/bin/env python3
"""Flag depth pixels that other views prove too deep.

A depth map claims that everything in front of its surface is empty. If
another view has confidently observed a surface point inside that claimed
empty space, the claim is wrong: the first surface along the ray cannot be
behind a point the ray passes through first. That is the free-space test of
visibility-based depth fusion (Merrell et al. 2007; COLMAP's geometric
consistency check makes the same argument).

It matters here because the one error fusion cannot absorb is a view placing
its surface too deep. Too shallow is harmless -- the hull already bounds it,
and another view's empty space overrides it. Too deep carves real material.
Measured on ground truth, 0.7% to 7.2% of pixels per view sit more than three
voxels behind the truth.

For each view, the confident pixels of every other view are projected into it
and z-buffered to their nearest depth. A pixel is contradicted when some
confident point lands in front of its surface by more than `tol` voxels;
contradictions from at least `min_views` distinct views mark it. Only
confident source points are used, because an unanchored pixel is usually too
shallow, and a too-shallow point would accuse a correct one.

Run:
  python3 tools/depth_filter.py --views refs/lucy_gt --depth-dir out/depth_mv
"""
import argparse
import json
import os

import numpy as np

VOX = 1.0 / 1024


def cam(meta, view):
    m = np.array(meta["views"][view]["matrix_world"], dtype=np.float64)
    return m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3]


def points(meta, view, d, keep):
    res = d.shape[0]
    o = meta["ortho_scale"]
    right, up, back, loc = cam(meta, view)
    r, c = np.nonzero(keep)
    u = ((c + 0.5) / res - 0.5) * o
    v = (0.5 - (r + 0.5) / res) * o
    return loc + u[:, None] * right + v[:, None] * up - d[r, c][:, None] * back


def splat_min(meta, view, X, res):
    o = meta["ortho_scale"]
    right, up, back, loc = cam(meta, view)
    rel = X - loc
    col = np.floor((rel @ right / o + 0.5) * res).astype(np.int64)
    row = np.floor((0.5 - rel @ up / o) * res).astype(np.int64)
    w = -(rel @ back)
    ok = (col >= 0) & (col < res) & (row >= 0) & (row < res)
    buf = np.full(res * res, np.inf)
    np.minimum.at(buf, row[ok] * res + col[ok], w[ok])
    return buf.reshape(res, res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--tol", type=float, default=2.0, help="voxels")
    ap.add_argument("--min-views", type=int, default=1)
    ap.add_argument("--conf-dist", type=float, default=20.0,
                    help="source pixels within this many pixels of a live "
                         "anchor count as confident")
    ap.add_argument("--only-views", default=None)
    ap.add_argument("--write", action="store_true",
                    help="write <view>_contra.npy alongside each depth map")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    have = [v for v in meta["views"]
            if os.path.exists(os.path.join(args.depth_dir, f"{v}.npy"))]
    targets = args.only_views.split(",") if args.only_views else have
    D, P = {}, {}
    for v in have:
        d = np.load(os.path.join(args.depth_dir, f"{v}.npy")).astype(np.float64)
        dist = np.load(os.path.join(args.depth_dir, f"{v}_anchordist.npy"))
        keep = np.isfinite(d) & (np.nan_to_num(dist, nan=1e9) <= args.conf_dist)
        D[v] = d
        P[v] = points(meta, v, d, keep)
    print(f"views with depth: {len(have)}")
    for v in targets:
        d = D[v]
        res = d.shape[0]
        count = np.zeros(d.shape, dtype=np.int16)
        for u in have:
            if u == v:
                continue
            z = splat_min(meta, v, P[u], res)
            count += (z < d - args.tol * VOX)
        flag = np.isfinite(d) & (count >= args.min_views)
        note = ""
        gt_p = os.path.join(args.views, "depth_npy", f"{v}.npy")
        if os.path.exists(gt_p):
            gt = np.load(gt_p)
            m = np.isfinite(d) & (gt < 1e9)
            e = (d - gt) / VOX
            bad = m & (e > 3)
            good = m & (np.abs(e) < 1)
            note = (f"behind>3: {bad.sum():>7,} caught {100*(flag&bad).sum()/max(bad.sum(),1):5.1f}%"
                    f"   good(<1vox) flagged {100*(flag&good).sum()/max(good.sum(),1):5.2f}%"
                    f"   flagged total {100*flag[m].mean():5.2f}%")
        print(f"  {v:<13} {note}", flush=True)
        if args.write:
            np.save(os.path.join(args.depth_dir, f"{v}_contra.npy"), count)


if __name__ == "__main__":
    main()
