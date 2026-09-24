#!/usr/bin/env python3
"""Score painted normals: accuracy vs truth, and agreement between views.

    python3 tools/genlab/score_paint.py --paint data/genlab/paint \
        --views data/genlab/proxy/views --truth data/genlab/appearance
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
import bands  # noqa: E402
from paint import View  # noqa: E402

PAIRS = [("01_front", "10_az315_up"), ("02_right", "07_az45_up"), ("03_back", "16_az90_up30"),
         ("04_left", "09_az225_up"), ("01_front", "18_az345_up30")]


def truth(truth_dir, meta, v, res):
    n = np.load(f"{truth_dir}/normal_npy/{v}.npy").astype(np.float32)
    n = np.stack([np.asarray(Image.fromarray(n[..., c]).resize((res, res), Image.BILINEAR))
                  for c in range(3)], -1).astype(np.float64)
    return n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-9)   # world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paint", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--res", type=int, default=1024)
    a = ap.parse_args()
    meta = json.load(open(f"{a.views}/cameras.json"))
    V = {v: View(a.views, meta, v, a.res) for v in meta["views"]}
    got = {}
    for v in V:
        p = f"{a.paint}/normals/{v}.npy"
        if os.path.exists(p):
            got[v] = V[v].world(np.load(p).astype(np.float64))
    print(f"{'view':15} {'proxy vs truth':>15} {'painted vs truth':>17}   detail r (1-8px) proxy / painted")
    rows = []
    for v in got:
        t = truth(a.truth, meta, v, a.res)
        m = V[v].mask & (np.linalg.norm(t, axis=-1) > 0.5)
        m = ndimage.binary_erosion(m, iterations=2)
        ep = bands.ang(V[v].nw[m], t[m]); eg = bands.ang(got[v][m], t[m])
        # detail correlation in the view's own frame (same projection, so fair)
        tc, pc, gc = V[v].cam(t), V[v].cam(V[v].nw), V[v].cam(got[v])
        det = lambda x: (bands.blur_n(x, m, 1) - bands.blur_n(x, m, 8))[m][:, :2].ravel()
        rp = np.corrcoef(det(pc), det(tc))[0, 1]; rg = np.corrcoef(det(gc), det(tc))[0, 1]
        rows.append((np.median(ep), np.median(eg), rp, rg))
        print(f"{v:15} {np.median(ep):13.1f}deg {np.median(eg):15.1f}deg   {rp:.2f} / {rg:.2f}")
    r = np.array(rows)
    print(f"{'mean':15} {r[:,0].mean():13.1f}deg {r[:,1].mean():15.1f}deg   {r[:,2].mean():.2f} / {r[:,3].mean():.2f}")
    print("cross-view agreement at shared surface points (median angle; truth floor in brackets):")
    for A, B in PAIRS:
        if A not in got or B not in got:
            continue
        va, vb = V[A], V[B]
        r_, c_, vis = vb.sees(va.X)
        vis &= va.mask
        vis = ndimage.binary_erosion(vis, iterations=3)
        nb = np.stack([ndimage.map_coordinates(got[B][..., k], [r_, c_], order=1) for k in range(3)], -1)
        nb /= np.linalg.norm(nb, axis=-1, keepdims=True).clip(1e-9)
        ta, tb = truth(a.truth, meta, A, a.res), truth(a.truth, meta, B, a.res)
        tb_ = np.stack([ndimage.map_coordinates(tb[..., k], [r_, c_], order=1) for k in range(3)], -1)
        tb_ /= np.linalg.norm(tb_, axis=-1, keepdims=True).clip(1e-9)
        e = bands.ang(got[A][vis], nb[vis]); et = bands.ang(ta[vis], tb_[vis])
        print(f"  {A} <-> {B}: {np.median(e):5.1f}deg  <5deg {100*(e<5).mean():4.1f}%   "
              f"[truth {np.median(et):.1f}deg, <5 {100*(et<5).mean():.1f}%]  n={vis.sum()}")


if __name__ == "__main__":
    main()
