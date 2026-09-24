#!/usr/bin/env python3
"""Find where each piece of integrated relief actually sits, by matching normals
across views.

Integration gets local shape right and absolute depth wrong: every piece
between two discontinuities floats by an unknown constant, and a single missed
jump drags whole regions off. Resting pieces against the hull fixes the
constant only where a piece happens to touch it.

Other views fix it everywhere they overlap. A surface point has one normal in
world space, whichever camera sees it. Slide view A's relief along A's rays by
an offset c; at the right c every point lands where the other views observe
the same normal, and at a wrong c it lands on some other part of their
images. That is a plane sweep (Collins 1996) with the normal field standing in
for colour -- and a better one than colour for this job, because normals do
not depend on where the lights were.

Occlusion is handled the way multi-view stereo usually handles it: per pixel,
only the best two source views count. A source that cannot see the point just
does not make the cut. Costs are aggregated over a window in A assuming the
offset, not the depth, is locally constant -- the relief carries the shape.

Run (one target view):
  python3 tools/normal_stereo.py --views refs/lucy_gt --relief R.npy --view 01_front
"""
import argparse
import json
import os

import numpy as np
from scipy import ndimage

BG = 1e9


def cam(meta, view):
    m = np.array(meta["views"][view]["matrix_world"], dtype=np.float64)
    return m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3], m[:3, :3]


def world_normals(views_dir, normals_dir, view, meta):
    if normals_dir:
        n = np.load(os.path.join(normals_dir, f"{view}.npy")).astype(np.float32)
        R = cam(meta, view)[4].astype(np.float32)
        n = n @ R.T                                  # camera -> world
    else:
        n = np.load(os.path.join(views_dir, "normal_npy", f"{view}.npy")
                    ).astype(np.float32)
    n /= np.linalg.norm(n, axis=2, keepdims=True).clip(1e-9)
    return n


def pixel_rays(meta, view, res):
    right, up, back, loc, _ = cam(meta, view)
    o = meta["ortho_scale"]
    c = ((np.arange(res) + 0.5) / res - 0.5) * o
    u = c[None, :]
    v = -c[:, None]
    P0 = (loc[None, None, :] + u[..., None] * right + v[..., None] * up)
    return P0.astype(np.float32), (-back).astype(np.float32)


def sweep(meta, target, sources, nA, nB, hitB, relief, hull, hit, offsets,
          win, facing_min=0.15, top=2):
    """Aggregated cost per (offset, pixel) of target, as float32 [K, H, W]."""
    res = hit.shape[0]
    o = meta["ortho_scale"]
    P0, dA = pixel_rays(meta, target, res)
    K = len(offsets)
    # float16 halves the volume (81 offsets x 4M pixels) so three views can be
    # swept at once; costs live in [0, 2] and the anchor cut is 0.012, well
    # inside float16's precision there
    cost = np.full((K,) + hit.shape, np.inf, dtype=np.float16)
    rows, cols = np.nonzero(hit)
    P = P0[rows, cols]
    nAp = nA[rows, cols]
    base = relief[rows, cols].astype(np.float32)
    floor = hull[rows, cols].astype(np.float32)
    srcs = []
    for s in sources:
        right, up, back, loc, _ = cam(meta, s)
        facing = nAp @ back.astype(np.float32)       # >0: surface faces s
        srcs.append((s, right.astype(np.float32), up.astype(np.float32),
                     loc.astype(np.float32), facing))
    for k, c in enumerate(offsets):
        d = base + c
        X = P + d[:, None] * dA[None, :]
        best = np.full((top, len(rows)), 2.0, dtype=np.float32)
        wsum = np.zeros((top, len(rows)), dtype=np.float32)
        for s, right, up, loc, facing in srcs:
            rel = X - loc
            col = (rel @ right / o + 0.5) * res - 0.5
            row = (0.5 - rel @ up / o) * res - 0.5
            ci = np.clip(np.rint(col).astype(np.int32), 0, res - 1)
            ri = np.clip(np.rint(row).astype(np.int32), 0, res - 1)
            ok = hitB[s][ri, ci] & (facing > facing_min)
            dot = np.einsum("ij,ij->i", nAp, nB[s][ri, ci])
            c_s = np.where(ok, 1.0 - dot, 2.0).astype(np.float32)
            # aggregate over the window before choosing the best sources, so
            # the choice is made per neighbourhood, not per noisy pixel
            img = np.full(hit.shape, 2.0, dtype=np.float32)
            img[rows, cols] = c_s
            img = ndimage.uniform_filter(img, win, mode="nearest")[rows, cols]
            # insert into the running top-k (k=2 here)
            b0, b1 = best[0], best[1]
            new0 = np.minimum(b0, img)
            new1 = np.minimum(b1, np.maximum(b0, img))
            best[0], best[1] = new0, new1
        agg = best[:top].mean(axis=0)
        agg = np.where(d >= floor - 1e-6, agg, np.inf)   # never in front of hull
        cost[k][rows, cols] = agg
    return cost


def normal_cost(meta, target, sources, nA, nB, hitB, depth, hull, hit, win=11,
                facing_min=0.15, top=2):
    """Aggregated cross-view normal disagreement of a given depth map, per
    pixel: the same score sweep_fast minimises, evaluated at one depth. Lets
    a finished depth map be checked against the other views."""
    z, c = sweep_fast(meta, target, sources, nA, nB, hitB, depth, hull, hit,
                      0.0, 0.0, 1.0, 1.0, win, facing_min, top, fixed=True)
    return c


def sweep_fast(meta, target, sources, nA, nB, hitB, relief, hull, hit,
               lo, hi, coarse=4.0, fine=1.0, win=11, facing_min=0.15, top=2,
               vox=1.0 / 1024, fixed=False, masked=True, trunc=0.0,
               gap=12.0, return_margin=False):
    """Coarse-to-fine version of sweep(), on the xp backend.

    A pass every `coarse` voxels over [lo, hi] finds each pixel's basin, then
    a pass every `fine` voxels within +-coarse of it finds the minimum, which
    a parabola through its neighbours refines below the step. Offsets
    evaluated: (hi-lo)/coarse + 2*coarse/fine + 2 -- 30 instead of 81 for the
    default range. Returns (depth, cost) as NumPy arrays, NaN / inf off hit.

    masked: the window averages only over this view's own silhouette
    (normalised convolution). Unmasked, the background inside an 11-pixel
    window counted as the worst possible match, so near every outline -- and
    across the whole width of a finger, a toe, a torch knob -- the cost was
    mostly background, and the thin parts that most need an anchor could
    never get one.

    trunc: cap each pixel's cost before the window average (truncated
    absolute differences). Uncapped, one pixel the source view sees occluded
    costs 2.0 and adds 2/121 = 0.017 to its 11x11 window -- more than the
    whole anchor threshold -- so near any occlusion edge in any source view,
    even the true depth failed: scored at ground-truth depth on Lucy's front
    view only 45% of pixels passed. Capped at 0.02: 83% pass, and the true
    depth beats +-3 and +-8 voxels on 92.6% of pixels instead of 78.7%.
    """
    from xp import cpu, ndi, xp
    res = hit.shape[0]
    o = meta["ortho_scale"]
    P0, dA = pixel_rays(meta, target, res)
    rows, cols = np.nonzero(hit)
    R, C = xp.asarray(rows), xp.asarray(cols)
    P = xp.asarray(P0[rows, cols])
    dA = xp.asarray(dA)
    nAp = xp.asarray(nA[rows, cols])
    base = xp.asarray(relief[rows, cols].astype(np.float32))
    floor = xp.asarray(hull[rows, cols].astype(np.float32))
    srcs = []
    for s in sources:
        right, up, back, loc, _ = cam(meta, s)
        facing = nAp @ xp.asarray(back.astype(np.float32)) > facing_min
        if float(facing.astype(xp.float32).mean()) < 1e-3:
            continue                       # sees none of this view's surface
        srcs.append((xp.asarray(right.astype(np.float32)),
                     xp.asarray(up.astype(np.float32)),
                     xp.asarray(loc.astype(np.float32)), facing,
                     xp.asarray(nB[s]), xp.asarray(hitB[s])))

    if masked:
        hm = xp.zeros(hit.shape, dtype=xp.float32)
        hm[R, C] = 1.0
        cover = xp.maximum(ndi.uniform_filter(hm, win, mode="nearest")[R, C], 1e-3)

    def cost_at(d):
        X = P + d[:, None] * dA[None, :]
        b0 = xp.full(len(d), 2.0, dtype=xp.float32)
        b1 = xp.full(len(d), 2.0, dtype=xp.float32)
        for right, up, loc, facing, nBs, hBs in srcs:
            rel = X - loc
            ci = xp.clip(xp.rint((rel @ right / o + 0.5) * res - 0.5)
                         .astype(xp.int32), 0, res - 1)
            ri = xp.clip(xp.rint((0.5 - rel @ up / o) * res - 0.5)
                         .astype(xp.int32), 0, res - 1)
            ok = hBs[ri, ci] & facing
            dot = xp.sum(nAp * nBs[ri, ci], axis=1)
            img = xp.full(hit.shape, 0.0 if masked else 2.0, dtype=xp.float32)
            if trunc > 0:
                img[R, C] = xp.where(ok, xp.minimum(1.0 - dot, trunc), trunc)
            else:
                img[R, C] = xp.where(ok, 1.0 - dot, 2.0)
            c_s = ndi.uniform_filter(img, win, mode="nearest")[R, C]
            b1 = xp.minimum(b1, xp.maximum(b0, c_s))
            b0 = xp.minimum(b0, c_s)
        agg = (b0 + b1) / 2 if top == 2 else b0
        if masked:
            agg = agg / cover
        return xp.where(d >= floor - 1e-6, agg, xp.inf)

    if fixed:                       # just score the depth as given
        cost = np.full(hit.shape, np.inf, dtype=np.float32)
        cost[rows, cols] = cpu(cost_at(base))
        return relief, cost
    # coarse -- every offset's cost is kept, to measure afterwards how
    # unique the best one is
    offs = np.arange(lo, hi + 1e-9, coarse) * vox
    CC = xp.stack([cost_at(base + c) for c in offs])
    kb = xp.argmin(CC, axis=0)
    best_c = CC[kb, xp.arange(len(rows))]
    offs_x = xp.asarray(offs.astype(np.float32))
    best_o = offs_x[kb]
    # uniqueness: the best cost this pixel reaches at least `gap` voxels away
    # from its best offset. On a broad flat surface (the Buddha's pedestal
    # top) every offset matches the other views' normals about equally; the
    # minimum is then noise, and anchors 45-118 voxels too deep were taken
    far = xp.abs(offs_x[:, None] - best_o[None, :]) > gap * vox
    margin_px = xp.min(xp.where(far, CC, xp.inf), axis=0) - best_c
    del CC, far
    # fine, around each pixel's basin
    deltas = np.arange(-coarse, coarse + 1e-9, fine) * vox
    F = xp.stack([cost_at(base + best_o + dd) for dd in deltas])
    k = xp.argmin(F, axis=0)
    K = len(deltas)
    km, kp = xp.clip(k - 1, 0, K - 1), xp.clip(k + 1, 0, K - 1)
    ar = xp.arange(len(rows))
    c0, cm, cp = F[k, ar], F[km, ar], F[kp, ar]
    den = cm - 2 * c0 + cp
    okp = xp.isfinite(cm) & xp.isfinite(cp) & (den > 1e-9) & (k > 0) & (k < K - 1)
    delta = xp.where(okp, 0.5 * (cm - cp) / xp.where(okp, den, 1), 0.0)
    off = best_o + xp.asarray(deltas.astype(np.float32))[k] \
        + xp.clip(delta, -0.5, 0.5) * fine * vox
    z = np.full(hit.shape, np.nan, dtype=np.float32)
    cost = np.full(hit.shape, np.inf, dtype=np.float32)
    z[rows, cols] = cpu(base + off)
    cost[rows, cols] = cpu(c0)
    if return_margin:
        marg = np.full(hit.shape, np.nan, dtype=np.float32)
        marg[rows, cols] = cpu(margin_px)
        return z, cost, marg
    return z, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--normals-dir", default=None)
    ap.add_argument("--relief", required=True, help="npy depth up to offsets")
    ap.add_argument("--hull-views", default="out/final_views")
    ap.add_argument("--view", required=True)
    ap.add_argument("--range", type=float, default=40.0, help="+- voxels")
    ap.add_argument("--step", type=float, default=0.5, help="voxels")
    ap.add_argument("--win", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    VOX = 1.0 / 1024
    views = list(meta["views"])
    hull = np.load(os.path.join(args.hull_views, "depth_npy", f"{args.view}.npy"))
    hull = np.where(hull < BG, hull, np.nan)
    hit = np.isfinite(hull)
    relief = np.load(args.relief)
    # centre the sweep on a contact-ish placement: median gap to the hull
    gap = np.nanmedian((hull - relief)[hit])
    relief = relief + gap
    nA = world_normals(args.views, args.normals_dir, args.view, meta)
    nB, hitB = {}, {}
    for s in views:
        if s == args.view:
            continue
        nB[s] = world_normals(args.views, args.normals_dir, s, meta)
        hb = np.load(os.path.join(args.hull_views, "depth_npy", f"{s}.npy"))
        hitB[s] = hb < BG
    offsets = np.arange(-args.range, args.range + 1e-9, args.step) * VOX
    cost = sweep(meta, args.view, list(nB), nA, nB, hitB, relief, hull, hit,
                 offsets.astype(np.float32), args.win)
    k = np.argmin(cost, axis=0)
    best = np.take_along_axis(cost, k[None], 0)[0]
    z = relief + offsets[k]
    z = np.where(hit & np.isfinite(best), z, np.nan)
    np.savez(args.out, z=z.astype(np.float32), cost=best.astype(np.float32),
             k=k.astype(np.int16))
    gt_path = os.path.join(args.views, "depth_npy", f"{args.view}.npy")
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
        m = hit & (gt < BG) & np.isfinite(z)
        e = (z - gt)[m] / VOX
        e0 = (relief - gt)[m] / VOX
        print(f"relief(median-placed): MAE {np.abs(e0).mean():.2f}  "
              f"stereo: MAE {np.abs(e).mean():.2f} vox  median |e| "
              f"{np.median(np.abs(e)):.2f}  within1 {100*(np.abs(e)<1).mean():.1f}%"
              f"  within3 {100*(np.abs(e)<3).mean():.1f}%  "
              f"behind>3 {100*(e>3).mean():.1f}%")


if __name__ == "__main__":
    main()
