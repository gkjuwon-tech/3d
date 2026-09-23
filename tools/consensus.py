#!/usr/bin/env python3
"""Render the first pass's agreed surface into every view, as new anchors.

A view's depth is only as good as its anchors: on Lucy it is right to 1.5
voxels on 97.6% of the pixels within 2 px of one of its own anchors, and on
39% of those 10-40 px away, where the integrated relief has drifted. But the
views see the same surface. A patch this view could not pin down by the
normal-field sweep has often been pinned by two others, and the robust
fusion of the first pass already knows where they agree.

So: march each view's pixel rays through the first-pass fused field, find
the first zero crossing, and keep it only where the fusion's support there
(the number of agreeing views that are anchored at that point, from
fuse_field.py --support-out) is at least --min-support. Those depths go back
to depth_mv.py --consensus as extra anchors for a second solve.

And only where this view agrees: the fused surface's normal there (the field's
gradient) must match this view's own photometric normal to --max-angle. The
normals are right to a degree; a depth is not. Without this check the passes
fed on themselves: a small pocket that pass 2 carved inside Lucy's head was
rendered back into the views as "agreed surface", adopted, and carved deeper,
until pass 3's face held a cavity 3-20 voxels under the skin (F@1 in the
face 93 -> 64) though it looked perfect from outside.

Run:
  python3 tools/consensus.py --field recon/fuse1_field.npy \
      --support recon/fuse1_support.npy --grid recon/fuse1_occ.npz \
      --views data/lucy/views --hull-views recon/hull/views --out recon/consensus
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xp import cpu, ndi, xp  # noqa: E402
import normal_stereo as ns  # noqa: E402

BG = 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--support", required=True)
    ap.add_argument("--grid", required=True, help="npz with lo, h, dims")
    ap.add_argument("--views", required=True)
    ap.add_argument("--hull-views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--only-views", default=None)
    ap.add_argument("--normals-dir", default=None,
                    help="photometric normals; with it, a consensus pixel is "
                         "kept only where the fused surface's normal agrees")
    ap.add_argument("--max-angle", type=float, default=15.0)
    ap.add_argument("--max-depth", type=float, default=160.0,
                    help="voxels behind the hull to search for the surface")
    args = ap.parse_args()

    t0 = time.time()
    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    g = np.load(args.grid)
    lo, h = g["lo"].astype(np.float64), float(g["h"])
    res = meta["resolution"][0]
    o = meta["ortho_scale"]
    F = xp.asarray(np.load(args.field).astype(np.float32))
    # support where the surface passes between voxel centres: the most any
    # neighbour has
    Sup = ndi.maximum_filter(xp.asarray(np.load(args.support)), size=3)
    lo_x = xp.asarray(lo, dtype=xp.float32)
    os.makedirs(args.out, exist_ok=True)
    print(f"loaded field: {time.time()-t0:.0f}s", flush=True)

    views = args.only_views.split(",") if args.only_views else list(meta["views"])
    for v in views:
        hull = np.load(os.path.join(args.hull_views, "depth_npy", f"{v}.npy"))
        hit = hull < BG
        r, c = np.nonzero(hit)
        m = np.array(meta["views"][v]["matrix_world"], dtype=np.float64)
        right, up, back, loc = (xp.asarray(m[:3, k], dtype=xp.float32)
                                for k in range(4))
        x = xp.asarray(((c + 0.5) / res - 0.5) * o, dtype=xp.float32)
        y = xp.asarray((0.5 - (r + 0.5) / res) * o, dtype=xp.float32)
        base = loc[None] + x[:, None] * right[None] + y[:, None] * up[None]
        t = xp.asarray(hull[hit], dtype=xp.float32) - h     # voxel in front
        t_end = t + args.max_depth * h
        n = len(r)
        found = xp.full(n, xp.nan, dtype=xp.float32)
        sup = xp.zeros(n, dtype=xp.uint8)

        def sample(P, field, order):
            idx = ((P - lo_x) / h - 0.5).T
            return ndi.map_coordinates(field, idx, order=order, mode="constant",
                                       cval=10.0 if order else 0)

        act = xp.arange(n)
        f_prev = sample(base - t[:, None] * back[None], F, 1)
        while len(act):
            # step by the field where it is a distance, never more than 2
            # voxels (a thin part must not be stepped over), never less than
            # a half
            step = xp.clip(0.7 * f_prev, 0.5, 2.0) * h
            t_new = t[act] + step
            P = base[act] - t_new[:, None] * back[None]
            f_new = sample(P, F, 1)
            cross = (f_new < 0) & (f_prev >= 0)
            if cross.any():
                a = act[cross]
                fp, fn = f_prev[cross], f_new[cross]
                ts = t[a] + step[cross] * fp / xp.maximum(fp - fn, 1e-6)
                found[a] = ts
                Ps = base[a] - ts[:, None] * back[None]
                sup[a] = sample(Ps, Sup, 0).astype(xp.uint8)
            t[act] = t_new
            keep = ~cross & (t_new < t_end[act])
            act = act[keep]
            f_prev = f_new[keep]
        okx = xp.isfinite(found) & (sup >= args.min_support)
        n_rej = 0
        if args.normals_dir:
            a = xp.nonzero(okx)[0]
            Ps = base[a] - found[a][:, None] * back[None]
            g = xp.stack([sample(Ps + h * e[None], F, 1) - sample(Ps - h * e[None], F, 1)
                          for e in xp.eye(3, dtype=xp.float32)], -1)
            g = g / xp.maximum(xp.linalg.norm(g, axis=1, keepdims=True), 1e-9)
            nw = xp.asarray(ns.world_normals(args.views, args.normals_dir, v, meta)
                            [r, c].astype(np.float32))[a]
            agree = (g * nw).sum(1) > np.cos(np.radians(args.max_angle))
            n_rej = int((~agree).sum())
            okx[a[~agree]] = False
        ok = cpu(okx)
        depth = np.full(hit.shape, np.nan, dtype=np.float32)
        depth[r[ok], c[ok]] = cpu(found)[ok]
        np.save(os.path.join(args.out, f"{v}.npy"), depth)
        note = ""
        gt_p = os.path.join(args.views, "depth_npy", f"{v}.npy")
        if os.path.exists(gt_p):
            gt = np.load(gt_p)
            mm = np.isfinite(depth) & (gt < BG)
            e = (depth - gt)[mm] / h
            note = (f"  vs GT: |e|<1.5 {100*(np.abs(e) < 1.5).mean():5.1f}%  "
                    f"deep>1.5 {100*(e > 1.5).mean():4.1f}%")
        print(f"  {v:<13} consensus {int(ok.sum()):>9,} px (normal check "
              f"dropped {n_rej:,}) "
              f"({100*ok.sum()/max(n, 1):4.1f}% of silhouette){note}  "
              f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
