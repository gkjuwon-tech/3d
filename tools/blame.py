#!/usr/bin/env python3
"""Trace every hole back to the view that emptied it, and say why.

For each hole sample (a ground-truth surface point the fused field puts
outside, from tools/holes.py) the per-view terms of the fusion are recomputed
exactly as tools/fuse_field.py forms them. The view whose term pushed hardest
toward "empty" is blamed, and its evidence at that pixel is classified:

  interp     the nearest pixel's depth does not claim the point empty; only
             the bilinear blend with a neighbour across a depth jump does
  occluded   the view cannot see the point at all (something is in front of
             it), yet its depth is behind it: the occluder was placed too deep
  deep       the view sees the point, and its depth there is too deep
  other      none of the above (the view is right; the sum went wrong)

and, for the blamed pixels, whether they sit at a depth discontinuity, far
from any anchor, or pressed onto the hull floor.

Run:
  python3 tools/blame.py --holes out/holes.npz --grid out/hull/grid.npz \
      --depth-dir out/depth --hull-views out/hull/views
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_surface import REGIONS  # noqa: E402

BG = 1e9


def cam(meta, v):
    m = np.array(meta["views"][v]["matrix_world"], dtype=np.float64)
    return m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holes", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--views", default="refs/lucy_gt")
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--hull-views", required=True)
    ap.add_argument("--normals-dir", default="out/ps_normals")
    ap.add_argument("--trunc", type=float, default=3.0)
    ap.add_argument("--conf-len", type=float, default=40.0)
    ap.add_argument("--conf-floor", type=float, default=0.02)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    g = np.load(args.grid)
    h = float(g["h"])
    hz = np.load(args.holes)
    X = hz["p"].astype(np.float64)
    n_pts = len(X)
    views = list(meta["views"])
    res = meta["resolution"][0]
    o = meta["ortho_scale"]
    tau = args.trunc
    V = len(views)
    contrib = np.zeros((V, n_pts))
    info = {}
    for k, v in enumerate(views):
        d = np.load(os.path.join(args.depth_dir, f"{v}.npy")).astype(np.float64)
        hitv = np.isfinite(d)
        idx = ndimage.distance_transform_edt(~hitv, return_distances=False,
                                             return_indices=True)
        dfill = d[idx[0], idx[1]]
        dist = np.nan_to_num(np.load(os.path.join(args.depth_dir,
                                                  f"{v}_anchordist.npy")),
                             nan=1e6).astype(np.float64)
        conf = np.maximum(np.exp(-dist / args.conf_len), args.conf_floor)
        nzv = np.abs(np.load(os.path.join(args.normals_dir, f"{v}.npy"))[..., 2])
        conf = np.where(hitv, conf * nzv, 0)
        right, up, back, loc = cam(meta, v)
        rel = X - loc
        col = (rel @ right / o + 0.5) * res - 0.5
        row = (0.5 - rel @ up / o) * res - 0.5
        w = -(rel @ back)
        s = (ndimage.map_coordinates(dfill, [row, col], order=1, mode="nearest")
             - w) / h
        c = ndimage.map_coordinates(conf, [row, col], order=0, mode="nearest")
        ok = (s > -tau) & (c > 0)
        wt = np.where(ok, c * np.where(s < 0, 1 + s / tau, 1.0), 0)
        contrib[k] = wt * np.clip(s, -tau, tau)
        info[v] = (d, dist, row, col, w, s)
    blamed = np.argmax(contrib, axis=0)
    strength = contrib[blamed, np.arange(n_pts)]
    cats = np.full(n_pts, "other", dtype=object)
    at_jump = np.zeros(n_pts, bool)
    far = np.zeros(n_pts, bool)
    floor = np.zeros(n_pts, bool)
    err = np.full(n_pts, np.nan)
    for k, v in enumerate(views):
        sel = np.nonzero(blamed == k)[0]
        if len(sel) == 0:
            continue
        d, dist, row, col, w, s = info[v]
        ri = np.clip(np.rint(row[sel]).astype(int), 0, res - 1)
        ci = np.clip(np.rint(col[sel]).astype(int), 0, res - 1)
        gt = np.load(os.path.join(args.views, "depth_npy", f"{v}.npy")).astype(np.float64)
        hull = np.load(os.path.join(args.hull_views, "depth_npy", f"{v}.npy")).astype(np.float64)
        dn = d[ri, ci]
        s_nn = (dn - w[sel]) / h
        gtp = gt[ri, ci]
        visible = np.abs(gtp - w[sel]) / h < 1.5
        e = (dn - gtp) / h
        err[sel] = e
        interp = (s[sel] > 0) & ~(s_nn > 0)
        occl = ~interp & ~visible & (gtp < w[sel])
        deep = ~interp & ~occl & (e > 1)
        cat = np.where(interp, "interp", np.where(occl, "occluded",
                       np.where(deep, "deep", "other")))
        cats[sel] = cat
        g3 = np.where(gt < BG, gt, np.nan)
        win_max = ndimage.maximum_filter(np.nan_to_num(g3, nan=-1e9), 5)
        win_min = ndimage.minimum_filter(np.nan_to_num(g3, nan=1e9), 5)
        at_jump[sel] = (win_max[ri, ci] - win_min[ri, ci]) / h > 5
        far[sel] = dist[ri, ci] > 40
        floor[sel] = np.abs(dn - hull[ri, ci]) < 1e-6
    names = ["all"] + list(REGIONS)
    print(f"hole samples: {n_pts:,}")
    print(f"{'region':<11}{'holes':>8}  {'interp':>7}{'occluded':>9}{'deep':>7}"
          f"{'other':>7}   {'@jump':>6}{'unanch':>7}{'floor':>6}   top blamed views")
    for name in names:
        if name == "all":
            m = np.ones(n_pts, bool)
        else:
            c, r = REGIONS[name]
            m = np.linalg.norm(X - np.array(c), axis=1) < r
        if m.sum() == 0:
            continue
        fr = lambda a: 100 * a[m].mean()
        vc = np.bincount(blamed[m], minlength=V)
        top = ", ".join(f"{views[i]} {100*vc[i]/m.sum():.0f}%"
                        for i in np.argsort(vc)[::-1][:3] if vc[i])
        print(f"{name:<11}{m.sum():>8,}  {fr(cats=='interp'):>6.1f}%{fr(cats=='occluded'):>8.1f}%"
              f"{fr(cats=='deep'):>6.1f}%{fr(cats=='other'):>6.1f}%   {fr(at_jump):>5.1f}%"
              f"{fr(far):>6.1f}%{fr(floor):>5.1f}%   {top}")
    np.savez(args.holes.replace(".npz", "_blame.npz"), blamed=blamed, cat=cats.astype(str),
             err=err, jump=at_jump, far=far, floor=floor, strength=strength)


if __name__ == "__main__":
    main()
