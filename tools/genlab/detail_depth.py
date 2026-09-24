#!/usr/bin/env python3
"""Proxy depth + painted detail -> per-view depth (screened integration).

The proxy's depth is known exactly; only the detail is new. So per view solve
for a displacement h on top of the proxy depth whose gradient matches what the
painted normals say beyond what the proxy's own depth already slopes, with a
screening term that holds h to zero at scales above --scale pixels:

    min  sum_edges (h_b - h_a - (g_ab - dz0_ab))^2  +  (1/scale^2) sum_px h^2

Edges across the proxy's own depth jumps are cut (the proxy knows where they are).

    python3 tools/genlab/detail_depth.py --views data/genlab/proxy2/views \
        --normals data/genlab/paint2/normals --out data/genlab/paint2/depth \
        [--truth refs/lucy_gt]
"""
import argparse
import json
import os

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg


def grads(n, px, nz_min=0.2):
    nz = np.maximum(n[..., 2], nz_min)
    # depth grows away from the camera: along +col it changes by nx/nz, along
    # +row (image down, camera -y) by -ny/nz, per unit length
    return n[..., 0] / nz * px, -n[..., 1] / nz * px


def solve(z0, mask, n, px, scale, cut_px):
    H, W = mask.shape
    idx = -np.ones((H, W), int)
    idx[mask] = np.arange(mask.sum())
    gx, gy = grads(n, px)
    rows_a, rows_b, tgt = [], [], []
    for (dr, dc, g) in ((0, 1, gx), (1, 0, gy)):
        a = idx[:H - dr, :W - dc]; b = idx[dr:, dc:]
        ok = (a >= 0) & (b >= 0)
        dz0 = z0[dr:, dc:] - z0[:H - dr, :W - dc]
        ok &= np.abs(dz0) < cut_px * px                 # proxy depth jump: cut
        ga = g[:H - dr, :W - dc]; gb = g[dr:, dc:]
        gm = 0.5 * (ga + gb)
        ok &= np.abs(gm) < cut_px * px
        rows_a.append(a[ok]); rows_b.append(b[ok]); tgt.append((gm - dz0)[ok])
    a = np.concatenate(rows_a); b = np.concatenate(rows_b); t = np.concatenate(tgt) / px
    m, N = len(a), mask.sum()
    r = np.arange(m)
    A = sparse.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[r, r], np.r_[b, a])), shape=(m, N))
    M = (A.T @ A + sparse.identity(N) / scale ** 2).tocsr()
    h, _ = cg(M, A.T @ t, rtol=1e-6, maxiter=3000)
    out = np.full((H, W), np.nan)
    out[mask] = z0[mask] + h * px
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--normals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--cut-px", type=float, default=4.0)
    ap.add_argument("--truth", default=None, help="view set with depth_npy to score against")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    meta = json.load(open(f"{a.views}/cameras.json"))
    os.makedirs(a.out, exist_ok=True)
    tot = []
    for v in meta["views"]:
        if a.only and v not in a.only.split(","):
            continue
        d = np.load(f"{a.views}/depth_npy/{v}.npy").astype(np.float64)
        res = d.shape[0]
        px = meta["ortho_scale"] / res
        mask = d < 1e3
        mask = ndimage.binary_erosion(mask, iterations=1)
        n = np.load(f"{a.normals}/{v}.npy").astype(np.float64)
        z0 = np.where(mask, d, 0)
        z = solve(z0, mask, n, px, a.scale, a.cut_px)
        np.save(f"{a.out}/{v}.npy", z.astype(np.float32))
        if a.truth and os.path.exists(f"{a.truth}/depth_npy/{v}.npy"):
            g = np.load(f"{a.truth}/depth_npy/{v}.npy").astype(np.float64)
            f = g.shape[0] // res
            g = g.reshape(res, f, res, f)
            ok = (g < 1e3).all(axis=(1, 3))
            g = np.where(ok, g.mean(axis=(1, 3)), np.nan)
            # the same surface: truth within 10 px of the proxy
            same = mask & ok & (np.abs(g - z0) < 10 * px)
            e0 = np.abs(z0 - g)[same] / px; e1 = np.abs(z - g)[same] / px
            # detail only: remove each map's local mean (5 px) before comparing
            def hp(x):
                x = np.where(same, x, 0)
                w = ndimage.uniform_filter(same.astype(float), 5)
                return x - ndimage.uniform_filter(x, 5) / np.maximum(w, 1e-6)
            r0 = np.corrcoef(hp(z0)[same], hp(g)[same])[0, 1]
            r1 = np.corrcoef(hp(z)[same], hp(g)[same])[0, 1]
            tot.append((np.median(e0), np.median(e1), r0, r1))
            print(f"{v:15} depth err median px: proxy {np.median(e0):5.2f}  +detail {np.median(e1):5.2f}"
                  f"   detail corr with truth: proxy {r0:.2f}  +detail {r1:.2f}", flush=True)
        else:
            print(f"{v:15} done", flush=True)
    if tot:
        t = np.array(tot)
        print(f"{'mean':15} depth err median px: proxy {t[:,0].mean():5.2f}  +detail {t[:,1].mean():5.2f}"
              f"   detail corr with truth: proxy {t[:,2].mean():.2f}  +detail {t[:,3].mean():.2f}")


if __name__ == "__main__":
    main()
