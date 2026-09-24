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

A second pass (--pass1, --consensus) re-solves stage 3 only, with more
anchors: surface points where at least two anchored views agree in the
robust fusion of the first pass, rendered into this view by consensus.py.
Measured on Lucy, a view's depth is right to 1.5 voxels on 97.6% of pixels
within 2 px of one of its own anchors but only 39% at 10-40 px, and only
half the surface lies near an anchor of any view -- so where another view
has pinned the surface, this one should be pinned there too. It is the
geometric-consistency pass of multi-view stereo (Schoenberger et al. 2016)
in the form this pipeline can use: other views' agreed depths as anchors.

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
    from linsolve import spd_solve
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
        z = spd_solve(AtA.tocsr(), Atb, x0=z, tol=1e-6)
        if it == iters:
            break
        res_e = (z[b] - z[a]) - grad
        we = w_edge / (1.0 + (res_e / sigma) ** 2)
        res_a = z[anc_idx] - anc_z
        wa = anc_w / (1.0 + (res_a / sigma_a) ** 2)
    return z, we, wa


def place_orphans(z, a, b, we, N, hull_p, live_idx, cut_w, q):
    """Put pieces no anchor reaches against the hull instead of leaving them
    glued to their neighbours.

    Across a depth jump the robust weights fall to ~1e-5, not to zero, and a
    few hundred such edges along a finger's outline outweigh the whole
    finger's pull toward the hull by a factor of hundreds: an unanchored
    finger, knob or lock of hair ends up near the depth of whatever lies
    behind it -- 132 voxels too deep on Lucy's outstretched hand. The view
    then declares the finger's own volume empty, and where views disagree
    about it the fused surface shatters into fragments.

    So after the solve, edges below cut_w are treated as cut, and every piece
    with no surviving anchor keeps its relief but is moved toward the camera
    until it touches the hull -- the q-th percentile of its gap, so one bad
    pixel cannot set it. Small protrusions are exactly where the other views'
    silhouettes wrap the hull tightly, so that is close to right, and a piece
    is only ever moved forward: this can make a surface too shallow, never too
    deep.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    live = we >= cut_w
    adj = sparse.coo_matrix((np.ones(int(live.sum())), (a[live], b[live])),
                            shape=(N, N))
    n_lab, lab = connected_components(adj, directed=False)
    anchored = np.zeros(n_lab, bool)
    anchored[lab[live_idx]] = True
    gap = hull_p - z                        # > 0 where z is in front of hull
    order = np.lexsort((gap, lab))
    lab_s, gap_s = lab[order], gap[order]
    starts = np.searchsorted(lab_s, np.arange(n_lab))
    ends = np.searchsorted(lab_s, np.arange(n_lab), side="right")
    size = ends - starts
    k = np.clip(starts + np.floor((size - 1) * q / 100).astype(np.int64),
                starts, np.maximum(ends - 1, starts))
    c = np.where(size > 0, gap_s[np.minimum(k, len(gap_s) - 1)], 0.0)
    shift = np.where(anchored, 0.0, np.minimum(c, 0.0))   # forward only
    moved = shift[lab] < 0
    z = z + shift[lab]
    med = float(np.median(shift[lab][moved])) if moved.any() else 0.0
    return z, int(((~anchored) & (shift < 0)).sum()), int(moved.sum()), med


def coarse_relief(hull, n, rgb, hit, px, args):
    """The relief at 1/k resolution, brought back up.

    The relief only has to give the sweep the local shape inside its 11-pixel
    window; the final depth is re-solved at full resolution against the
    anchors. So it is integrated on k x k blocks -- normals averaged and
    renormalised, a block counting as hit when any of its pixels does -- with
    k^2 fewer unknowns, then upsampled bilinearly.
    """
    k = args.relief_scale
    H, W = hit.shape
    h2, w2 = H // k, W // k
    nb = n[:h2 * k, :w2 * k].reshape(h2, k, w2, k, 3).mean((1, 3))
    nb /= np.linalg.norm(nb, axis=2, keepdims=True).clip(1e-9)
    hb = hull[:h2 * k, :w2 * k].reshape(h2, k, w2, k)
    hitb = np.isfinite(hb).any((1, 3))
    hullb = np.where(hitb, np.nanmin(np.where(np.isfinite(hb), hb, np.inf),
                                     axis=(1, 3)), np.nan)
    rgbb = rgb[:h2 * k, :w2 * k].reshape(h2, k, w2, k).mean((1, 3))
    keepb = dcx.cut_edges(nb, rgbb, hullb, hitb, args.budget)
    rb, _, _ = dcx.integrate(hullb, nb, hitb, keepb, px * k, args.nz_floor,
                             args.irls, args.sigma * k)
    filled = np.where(hitb, rb, np.nanmean(rb[hitb]))
    rows = (np.arange(H) + 0.5) / k - 0.5
    cols = (np.arange(W) + 0.5) / k - 0.5
    up = ndimage.map_coordinates(filled, np.meshgrid(rows, cols, indexing="ij"),
                                 order=1, mode="nearest")
    return np.where(hit, up, np.nan)


def process(views_dir, hull_dir, normals_dir, view, meta, args, nB, hitB):
    t0 = time.time()
    hull, n, rgb = dcx.load(views_dir, hull_dir, normals_dir, view, meta)
    hit = np.isfinite(hull)
    px = meta["ortho_scale"] / hull.shape[0]
    keep = dcx.cut_edges(n, rgb, hull, hit, args.budget)
    if args.pass1:
        # second pass: relief and sweep as the first pass left them
        P = lambda k: os.path.join(args.pass1, f"{view}_{k}.npy")  # noqa: E731
        relief = np.load(P("relief")).astype(np.float64)
        z_st = np.load(P("sweep")).astype(np.float64)
        best = np.nan_to_num(np.load(P("cost")).astype(np.float32), nan=np.inf)
        margin = np.load(P("margin")).astype(np.float32) \
            if os.path.exists(P("margin")) else None
        if margin is not None:
            np.save(os.path.join(args.out, f"{view}_margin.npy"), margin.astype(np.float16))
        t1 = t2 = time.time()
        return finish(view, hull, n, hit, keep, px, relief, z_st, best, args,
                      t0, t1, t2, margin)

    # 1. relief
    if args.relief_scale > 1:
        relief = coarse_relief(hull, n, rgb, hit, px, args)
    else:
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
    srcs = [s for s in nB if s != view]
    if args.sweep == "fast":
        z_st, best, margin = ns.sweep_fast(
            meta, view, srcs, nA, nB, hitB, relief, hull, hit,
            -args.range_front, args.range_back, args.coarse, 1.0, args.win,
            masked=not args.sweep_unmasked, trunc=args.sweep_trunc,
            gap=args.unique_gap, return_margin=True)
        np.save(os.path.join(args.out, f"{view}_margin.npy"), margin.astype(np.float16))
    else:
        offsets = (np.arange(-args.range_front, args.range_back + 1e-9,
                             args.step) * VOX).astype(np.float32)
        cost = ns.sweep(meta, view, srcs, nA, nB, hitB, relief, hull, hit,
                        offsets, args.win)
        k = np.argmin(cost, axis=0)
        best = np.take_along_axis(cost, k[None], 0)[0].astype(np.float32)
        off = refine_parabola(cost, k, offsets)
        del cost
        z_st = relief + off
    t2 = time.time()
    return finish(view, hull, n, hit, keep, px, relief, z_st, best, args,
                  t0, t1, t2, margin if args.sweep == "fast" else None)


def finish(view, hull, n, hit, keep, px, relief, z_st, best, args, t0, t1, t2,
           margin=None):
    # 3. anchored robust solve
    a, b, grad, idx = dcx.build_edges(n, hit, keep, px, args.nz_floor)
    N = int(hit.sum())
    anc = hit & np.isfinite(best) & (best < args.anchor_cost) \
        & (z_st >= hull - 1e-6)
    if margin is not None and args.unique_min > 0:
        # an anchor only where the best depth beats every depth unique_gap
        # voxels away by unique_min: a flat surface matches everywhere
        anc &= np.nan_to_num(margin, nan=0.0) >= args.unique_min
    anc_idx = idx[anc]
    anc_z = z_st[anc]
    anc_w = np.full(len(anc_idx), (1.0 / px ** 2) / args.anchor_len ** 2)
    n_cons = 0
    if args.consensus:
        # other views' agreed surface, as further anchors. Where this view
        # has its own anchor too, both go in and the robust reweighting
        # below keeps whichever the relief and the neighbours agree with.
        zc = np.load(os.path.join(args.consensus, f"{view}.npy"))
        cm = hit & np.isfinite(zc) & (zc >= hull - 1e-6)
        n_cons = int(cm.sum())
        anc = anc | cm
        anc_idx = np.concatenate([anc_idx, idx[cm]])
        anc_z = np.concatenate([anc_z, zc[cm]])
        anc_w = np.concatenate([anc_w, np.full(n_cons, anc_w[0] if len(anc_w)
                                               else (1.0 / px ** 2) / args.anchor_len ** 2)
                                * args.consensus_w])
    if args.anchor_soft > 0:
        # graded rather than thresholded: a thin feature's window straddles
        # its outline, so even its right answer scores a middling cost, and a
        # hard cut at 0.012 left fingers and the torch knob with no anchor at
        # all -- free to be dragged off by their neighbours
        anc_w = anc_w * np.exp(-(best[anc] - best[anc].min()) / args.anchor_soft)
    z, we, wa = anchored_solve(a, b, grad, np.ones(len(a)), N, hull[hit], px,
                               anc_idx, anc_z, anc_w, args.irls_final,
                               args.sigma, args.sigma_anchor * VOX,
                               x0=relief[hit])
    kept = wa > 0.5 * anc_w
    orphan_note = ""
    if args.orphans == "contact":
        z, n_orph, n_px, med_shift = place_orphans(z, a, b, we, N, hull[hit],
                                                   anc_idx[kept], args.cut_w,
                                                   args.orphan_q)
        orphan_note = (f"  orphans {n_orph} pieces / {n_px:,} px, median "
                       f"shift {med_shift / VOX:+.1f} vox")
    out = np.full(hit.shape, np.nan)
    out[hit] = z
    out = np.where(hit, np.maximum(out, hull), np.nan)

    # confidence: distance (in pixels) to the nearest surviving anchor
    live = np.zeros(hit.shape, bool)
    hr, hc = np.nonzero(hit)          # idx numbers hit pixels row-major
    live[hr[anc_idx[kept]], hc[anc_idx[kept]]] = True
    dist = ndimage.distance_transform_edt(~live).astype(np.float32)
    dist[~hit] = np.nan
    t3 = time.time()
    if n_cons:
        orphan_note += f"  consensus anchors {n_cons:,}"
    return out.astype(np.float32), dist, best.astype(np.float32), \
        z_st.astype(np.float32), (t1 - t0, t2 - t1, t3 - t2), anc.sum(), \
        int((wa > 0.5 * anc_w).sum()), orphan_note, relief


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--hull-views", required=True)
    ap.add_argument("--normals-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--nz-floor", type=float, default=0.15)
    ap.add_argument("--irls", type=int, default=10)
    ap.add_argument("--relief-scale", type=int, default=2,
                    help="integrate the relief on k x k blocks (1 = full res). "
                         "2 is the confirmed setting: same accuracy, 4x fewer "
                         "unknowns")
    ap.add_argument("--sigma", type=float, default=2e-4)
    ap.add_argument("--range-front", type=float, default=20.0,
                    help="voxels swept toward the camera from the local centre")
    ap.add_argument("--range-back", type=float, default=160.0,
                    help="voxels swept away from it; the truth is behind the "
                         "hull, so the range is lopsided on purpose. 60 did "
                         "not reach the inside of Lucy's wing from the side, "
                         "100-120 voxels behind the hull: no anchor formed, "
                         "the solve dragged the piece onto the hull, and "
                         "every side view agreed on the same wrong slab. 160: "
                         "side-view MAE 3.00/5.12 -> 1.83/1.50 voxels, twice "
                         "the sweep time")
    ap.add_argument("--center-sigma", type=float, default=24.0,
                    help="pixels; smoothing of the gap that centres the sweep")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--win", type=int, default=11)
    ap.add_argument("--sweep", default="fast", choices=["fast", "full"],
                    help="fast: coarse-to-fine, 30 offsets; full: every step")
    ap.add_argument("--coarse", type=float, default=4.0,
                    help="voxels between offsets in the fast sweep's first pass")
    ap.add_argument("--sweep-trunc", type=float, default=0.02,
                    help="cap each pixel's matching cost at this before the "
                         "window average (truncated cost); 0 = uncapped")
    ap.add_argument("--sweep-unmasked", action="store_true",
                    help="old window: background counts as the worst match")
    ap.add_argument("--anchor-cost", type=float, default=0.012)
    ap.add_argument("--anchor-soft", type=float, default=0.0,
                    help="if > 0, anchor weight falls off as exp(-cost/this)")
    ap.add_argument("--anchor-len", type=float, default=8.0,
                    help="screening length of an anchor, in pixels")
    ap.add_argument("--sigma-anchor", type=float, default=1.5,
                    help="Cauchy scale for anchor residuals, in voxels")
    ap.add_argument("--irls-final", type=int, default=5)
    ap.add_argument("--orphans", default="contact", choices=["contact", "keep"],
                    help="contact: pieces with no anchor are moved forward onto "
                         "the hull instead of hanging off their neighbours")
    ap.add_argument("--cut-w", type=float, default=0.05,
                    help="robust edge weight below which an edge separates pieces")
    ap.add_argument("--orphan-q", type=float, default=99.5)
    ap.add_argument("--only-views", default=None)
    ap.add_argument("--pass1", default=None,
                    help="second pass: reuse relief, sweep and cost saved by "
                         "the first pass in this directory; re-solve only")
    ap.add_argument("--consensus", default=None,
                    help="directory of <view>.npy consensus depths "
                         "(consensus.py), added as anchors")
    ap.add_argument("--consensus-w", type=float, default=1.0,
                    help="weight of a consensus anchor relative to a sweep one")
    ap.add_argument("--unique-gap", type=float, default=12.0,
                    help="voxels: the sweep's second-best depth is looked for "
                         "at least this far from its best (<view>_margin.npy)")
    ap.add_argument("--unique-min", type=float, default=0.0,
                    help="an anchor needs its best cost to beat that second "
                         "best by this much; 0 off")
    ap.add_argument("--no-final-cost", dest="final_cost", action="store_false",
                    help="skip <view>_fcost.npy (cross-view normal cost at "
                         "the final depth, fuse_field's view weight)")
    ap.add_argument("--save-pass1", action="store_true",
                    help="save relief and sweep depth for a later --pass1")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    views = args.only_views.split(",") if args.only_views else list(meta["views"])
    os.makedirs(args.out, exist_ok=True)
    nB, hitB = {}, {}
    for s in ([] if args.pass1 and not args.final_cost else meta["views"]):
        nB[s] = ns.world_normals(args.views, args.normals_dir, s, meta)
        hb = np.load(os.path.join(args.hull_views, "depth_npy", f"{s}.npy"))
        hitB[s] = hb < BG
    for v in views:
        z, dist, cost, z_st, tt, n_anc, n_live, orphan_note, relief = process(
            args.views, args.hull_views, args.normals_dir, v, meta, args, nB, hitB)
        np.save(os.path.join(args.out, f"{v}.npy"), z)
        np.save(os.path.join(args.out, f"{v}_anchordist.npy"), dist)
        np.save(os.path.join(args.out, f"{v}_cost.npy"), cost)
        if args.final_cost:
            # how well the other views' normals agree with this view's FINAL
            # depth -- the same score the sweep minimises, at the answer
            # rather than at the sweep's guess. Where two views disagree
            # about a surface, this is what tells which one is right:
            # under Lucy's right ear it separates right from wrong depths
            # with AUC 0.83, against 0.55 for the anchor-distance weight
            # fuse_field used alone (fuse_field --fcost-s).
            hit = np.isfinite(z)
            _, fc = ns.sweep_fast(meta, v, [s for s in nB if s != v], nB[v], nB,
                                  hitB, z, z, hit, 0.0, 0.0, 1.0, 1.0, args.win,
                                  fixed=True, trunc=args.sweep_trunc)
            fc = np.where(hit & np.isfinite(fc), fc, np.nan)
            np.save(os.path.join(args.out, f"{v}_fcost.npy"), fc.astype(np.float16))
        if args.save_pass1:
            np.save(os.path.join(args.out, f"{v}_sweep.npy"), z_st)
            np.save(os.path.join(args.out, f"{v}_relief.npy"),
                    relief.astype(np.float32))
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
              f"[{tt[0]:.0f}+{tt[1]:.0f}+{tt[2]:.0f}s]{orphan_note}", flush=True)


if __name__ == "__main__":
    main()
