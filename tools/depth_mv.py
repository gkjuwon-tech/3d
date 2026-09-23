#!/usr/bin/env python3
"""Per-view depth: local shape from normals, placement from other views.

Three stages per view:

  1. relief   robust normal integration (Cauchy IRLS over the edges the
              discontinuity detector left). Right locally, wrong by an unknown
              constant per piece, and wrong wherever a jump was bridged.
  2. anchors  normal-field plane sweep against every other view
              (normal_stereo.sweep). Where the match is tight the absolute
              depth is known to a fraction of a voxel: on the front view the
              best 40% of pixels by match cost sit within one voxel 91% of the
              time, spread across the whole figure.
  3. solve    the relief again, now screened toward the anchors, with both the
              edges and the anchors reweighted robustly. An anchor that
              disagrees with the relief and its neighbours loses its vote; an
              edge that disagrees with two well-anchored regions is a jump and
              gets cut.

Pieces that no anchor reaches are left where the weak pull toward the hull
puts them -- in front of the truth, which carves less, never more.

Run:
  python3 tools/depth_mv.py --views refs/lucy_gt --hull-views out/final_views \
      --normals-dir out/ps_normals --out out/depth_mv
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_contact as dcx  # noqa: E402
import normal_stereo as ns  # noqa: E402

BG = 1e9
VOX = 1.0 / 1024


def refine_parabola(cost, k, offsets):
    """Sub-step offset from a parabola through the best cost and its two
    neighbours; falls back to the grid value at the ends or on a flat fit."""
    K = cost.shape[0]
    km = np.clip(k - 1, 0, K - 1)
    kp = np.clip(k + 1, 0, K - 1)
    c0 = np.take_along_axis(cost, k[None], 0)[0].astype(np.float32)
    cm = np.take_along_axis(cost, km[None], 0)[0].astype(np.float32)
    cp = np.take_along_axis(cost, kp[None], 0)[0].astype(np.float32)
    den = cm - 2 * c0 + cp
    ok = np.isfinite(cm) & np.isfinite(cp) & (den > 1e-9) & (k > 0) & (k < K - 1)
    delta = np.where(ok, 0.5 * (cm - cp) / np.where(ok, den, 1), 0.0)
    step = offsets[1] - offsets[0]
    return offsets[k] + np.clip(delta, -0.5, 0.5) * step


def anchored_solve(a, b, grad, w_edge, N, hull_p, px, anc_idx, anc_z, anc_w,
                   iters, sigma, sigma_a, x0):
    """Robust screened integration toward sparse anchors."""
    from scipy import sparse
    import pyamg
    m = len(a)
    r = np.arange(m)
    z = x0
    wa = anc_w.copy()
    we = w_edge.copy()
    for it in range(iters + 1):
        sw = np.sqrt(we) / px
        A = sparse.csr_matrix((np.concatenate([sw, -sw]),
                               (np.concatenate([r, r]), np.concatenate([b, a]))),
                              shape=(m, N))
        AtA = (A.T @ A).tocsr()
        lam = 1e-9 * AtA.diagonal().mean()
        D = np.full(N, lam)
        rhs = lam * hull_p
        np.add.at(D, anc_idx, wa)
        np.add.at(rhs, anc_idx, wa * anc_z)
        AtA = AtA + sparse.diags(D)
        Atb = A.T @ (grad * sw) + rhs
        ml = pyamg.smoothed_aggregation_solver(AtA, symmetry="symmetric",
                                               max_coarse=500)
        z = ml.solve(Atb, x0=z, tol=1e-10, accel="cg", maxiter=400)
        if it == iters:
            break
        res_e = (z[b] - z[a]) - grad
        we = w_edge / (1.0 + (res_e / sigma) ** 2)
        res_a = z[anc_idx] - anc_z
        wa = anc_w / (1.0 + (res_a / sigma_a) ** 2)
    return z, we, wa


def process(views_dir, hull_dir, normals_dir, view, meta, args, nB, hitB):
    t0 = time.time()
    hull, n, rgb = dcx.load(views_dir, hull_dir, normals_dir, view, meta)
    hit = np.isfinite(hull)
    px = meta["ortho_scale"] / hull.shape[0]
    keep = dcx.cut_edges(n, rgb, hull, hit, args.budget)

    # 1. relief
    relief, _, _ = dcx.integrate(hull, n, hit, keep, px, args.nz_floor,
                                 args.irls, args.sigma)
    t1 = time.time()

    # 2. anchors by normal-field sweep. The sweep has to be centred close to
    # the truth everywhere, and one constant for the whole image is not: from
    # the top view the head and the plinth are hundreds of voxels apart in
    # depth, so with a global centre the true offset of whole regions fell
    # outside the swept range and they locked onto the least-wrong match
    # instead -- entire pieces placed 28 voxels too deep. The centre is now
    # local: the relief is shifted by a smoothed gap to the hull, so each
    # region starts on its own stretch of hull and the sweep only has to find
    # how far behind it the surface sits.
    gap = np.where(hit, hull - relief, 0.0)
    wts = ndimage.gaussian_filter(hit.astype(np.float64), args.center_sigma)
    local = ndimage.gaussian_filter(gap, args.center_sigma) / np.maximum(wts, 1e-9)
    relief = relief + local
    nA = ns.world_normals(views_dir, normals_dir, view, meta)
    offsets = (np.arange(-args.range_front, args.range_back + 1e-9, args.step)
               * VOX).astype(np.float32)
    srcs = [s for s in nB if s != view]
    cost = ns.sweep(meta, view, srcs, nA, nB, hitB, relief, hull, hit,
                    offsets, args.win)
    k = np.argmin(cost, axis=0)
    best = np.take_along_axis(cost, k[None], 0)[0].astype(np.float32)
    off = refine_parabola(cost, k, offsets)
    del cost
    z_st = relief + off
    t2 = time.time()

    # 3. anchored robust solve
    a, b, grad, idx = dcx.build_edges(n, hit, keep, px, args.nz_floor)
    N = int(hit.sum())
    anc = hit & np.isfinite(best) & (best < args.anchor_cost) \
        & (z_st >= hull - 1e-6)
    anc_idx = idx[anc]
    anc_z = z_st[anc]
    anc_w = np.full(len(anc_idx), (1.0 / px ** 2) / args.anchor_len ** 2)
    z, we, wa = anchored_solve(a, b, grad, np.ones(len(a)), N, hull[hit], px,
                               anc_idx, anc_z, anc_w, args.irls_final,
                               args.sigma, args.sigma_anchor * VOX,
                               x0=relief[hit])
    out = np.full(hit.shape, np.nan)
    out[hit] = z
    out = np.where(hit, np.maximum(out, hull), np.nan)

    # confidence: distance (in pixels) to the nearest surviving anchor
    live = np.zeros(hit.shape, bool)
    ar, ac = np.nonzero(anc)          # same row-major order as idx[anc]
    kept = wa > 0.5 * anc_w
    live[ar[kept], ac[kept]] = True
    dist = ndimage.distance_transform_edt(~live).astype(np.float32)
    dist[~hit] = np.nan
    t3 = time.time()
    return out.astype(np.float32), dist, best.astype(np.float32), \
        z_st.astype(np.float32), (t1 - t0, t2 - t1, t3 - t2), anc.sum(), \
        int((wa > 0.5 * anc_w).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--hull-views", required=True)
    ap.add_argument("--normals-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--nz-floor", type=float, default=0.15)
    ap.add_argument("--irls", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=2e-4)
    ap.add_argument("--range-front", type=float, default=20.0,
                    help="voxels swept toward the camera from the local centre")
    ap.add_argument("--range-back", type=float, default=60.0,
                    help="voxels swept away from it; the truth is behind the "
                         "hull, so the range is lopsided on purpose")
    ap.add_argument("--center-sigma", type=float, default=24.0,
                    help="pixels; smoothing of the gap that centres the sweep")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--win", type=int, default=11)
    ap.add_argument("--anchor-cost", type=float, default=0.012)
    ap.add_argument("--anchor-len", type=float, default=8.0,
                    help="screening length of an anchor, in pixels")
    ap.add_argument("--sigma-anchor", type=float, default=1.5,
                    help="Cauchy scale for anchor residuals, in voxels")
    ap.add_argument("--irls-final", type=int, default=5)
    ap.add_argument("--only-views", default=None)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    views = args.only_views.split(",") if args.only_views else list(meta["views"])
    os.makedirs(args.out, exist_ok=True)
    nB, hitB = {}, {}
    for s in meta["views"]:
        nB[s] = ns.world_normals(args.views, args.normals_dir, s, meta)
        hb = np.load(os.path.join(args.hull_views, "depth_npy", f"{s}.npy"))
        hitB[s] = hb < BG
    for v in views:
        z, dist, cost, z_st, tt, n_anc, n_live = process(
            args.views, args.hull_views, args.normals_dir, v, meta, args, nB, hitB)
        np.save(os.path.join(args.out, f"{v}.npy"), z)
        np.save(os.path.join(args.out, f"{v}_anchordist.npy"), dist)
        np.save(os.path.join(args.out, f"{v}_cost.npy"), cost)
        note = ""
        gt_path = os.path.join(args.views, "depth_npy", f"{v}.npy")
        if os.path.exists(gt_path):
            gt = np.load(gt_path)
            hull = np.load(os.path.join(args.hull_views, "depth_npy", f"{v}.npy"))
            m = np.isfinite(z) & (gt < BG)
            e = (z - gt)[m] / VOX
            eh = (hull - gt)[m] / VOX
            note = (f"MAE {np.abs(e).mean():5.2f} (hull {np.abs(eh).mean():5.2f}) "
                    f"med {np.median(np.abs(e)):4.2f}  w1 {100*(np.abs(e)<1).mean():4.1f}% "
                    f"w3 {100*(np.abs(e)<3).mean():4.1f}%  behind>1 {100*(e>1).mean():4.1f}% "
                    f">3 {100*(e>3).mean():4.1f}%")
        print(f"  {v:<13} anchors {n_anc:>7,} live {n_live:>7,}  {note}  "
              f"[{tt[0]:.0f}+{tt[1]:.0f}+{tt[2]:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
