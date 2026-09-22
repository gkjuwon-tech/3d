#!/usr/bin/env python3
"""Bring per-view depth estimates into one metric space, and measure what that
costs when the anchor is the visual hull instead of the truth.

A monocular depth network predicts depth only up to an unknown scale and shift,
and the unknowns differ per view. Six such maps back-projected without solving
for them do not land in a common space at all, which is the first reason a
six-view fusion explodes. The plan's answer is to anchor the solve on the
visual hull, which is available without ground truth.

This measures three versions of the same depth maps:

  raw        normalized per view to [0,1]; what naive fusion would use
  hull-fit   affine solved per view against the hull's rendered depth
  gt-fit     affine solved per view against the true depth (an oracle)

The gap between hull-fit and gt-fit is the price of not having ground truth,
and is the number that decides whether the plan's anchor is good enough.

Run:
  python3 tools/calibrate_depth.py --est out/est --views refs/lucy_gt \
      --hull-views out/hull_views_1024 --out out/m2
"""
import argparse
import json
import os

import numpy as np

BG = 1e9


def robust_affine(x, y, iters=12, huber=1.345):
    """Least squares a*x+b ~= y, reweighted so concavities -- where the hull
    is far from any real surface -- stop dragging the fit."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    w = np.ones_like(x)
    a, b = 1.0, 0.0
    for _ in range(iters):
        sw = w.sum()
        mx = (w * x).sum() / sw
        my = (w * y).sum() / sw
        vxx = (w * (x - mx) ** 2).sum()
        vxy = (w * (x - mx) * (y - my)).sum()
        a = vxy / vxx if vxx > 1e-20 else 1.0
        b = my - a * mx
        r = y - (a * x + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        u = np.abs(r) / s
        w = np.where(u <= huber, 1.0, huber / np.maximum(u, 1e-12))
    return float(a), float(b)


def view_rays(info, ortho, res):
    """Per-pixel world origin and view direction for an orthographic camera."""
    m = np.array(info["matrix_world"], dtype=np.float64)
    right, up, back, loc = m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3]
    fwd = -back
    j = (np.arange(res) + 0.5) / res
    u = (j - 0.5) * ortho
    v = (0.5 - j) * ortho
    origin = (loc[None, None, :]
              + v[:, None, None] * up[None, None, :]
              + u[None, :, None] * right[None, None, :])
    return origin, fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True, help="per-view estimates from the kernel")
    ap.add_argument("--views", required=True, help="ground-truth view set")
    ap.add_argument("--hull-views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--erode", type=int, default=6,
                    help="pixels eroded from the mask before fitting; depth at "
                         "a silhouette edge is the least trustworthy part of "
                         "any estimate")
    args = ap.parse_args()

    from scipy import ndimage

    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    manifest = json.load(open(os.path.join(args.est, "manifest.json")))
    ortho, res = meta["ortho_scale"], meta["resolution"][0]
    os.makedirs(args.out, exist_ok=True)

    depth_models = [k for k, v in manifest["models"].items()
                    if v["kind"] == "depth"]
    print("depth models:", depth_models)

    report = {"models": {}}
    for model in depth_models:
        spec = manifest["models"][model]
        # A disparity model is affine in 1/z, not in z; fitting it in the wrong
        # space bakes in a systematic bend.
        disparity = "inverse" in spec.get("note", "").lower()
        rows = {}
        clouds = {"raw": [], "hull": [], "gt": []}

        for view, info in meta["views"].items():
            gt = np.load(os.path.join(args.views, "depth_npy", f"{view}.npy"))
            hull = np.load(os.path.join(args.hull_views, "depth_npy",
                                        f"{view}.npy"))
            est = np.load(os.path.join(args.est, model, f"{view}.npy")
                          ).astype(np.float64)

            # Some estimators mark regions they decline to predict (MoGe uses
            # NaN); those pixels are excluded everywhere rather than poisoning
            # the mean.
            valid = np.isfinite(est) if est.ndim == 2 else \
                np.isfinite(est).all(-1)
            hit = (gt < BG) & valid
            fit_mask = ndimage.binary_erosion(hit, iterations=args.erode) \
                & (hull < BG)

            target_gt = 1.0 / gt if disparity else gt
            target_hull = 1.0 / hull if disparity else hull

            a_g, b_g = robust_affine(est[fit_mask], target_gt[fit_mask])
            a_h, b_h = robust_affine(est[fit_mask], target_hull[fit_mask])

            def to_depth(a, b):
                z = a * est + b
                return 1.0 / np.clip(z, 1e-6, None) if disparity else z

            z_gt = to_depth(a_g, b_g)
            z_hull = to_depth(a_h, b_h)
            # raw: per-view min/max normalization onto the hull's depth range,
            # which is the best a naive pipeline does without solving anything
            est = np.where(valid, est, np.nan)
            lo, hi = est[hit].min(), est[hit].max()
            span = hull[fit_mask]
            z_raw = (est - lo) / max(hi - lo, 1e-9)
            z_raw = z_raw * (span.max() - span.min()) + span.min()
            if disparity:
                z_raw = span.max() + span.min() - z_raw

            # The hull's own rendered depth is the baseline to beat. If a
            # calibrated estimate cannot improve on the thing used to calibrate
            # it, the estimator is contributing nothing on this view.
            hull_hit = hull < BG
            both = hit & hull_hit

            def err(z):
                d = np.abs(z[hit] - gt[hit])
                return {"mae": float(d.mean()),
                        "rmse": float(np.sqrt((d ** 2).mean())),
                        "p95": float(np.percentile(d, 95))}

            hull_err = float(np.abs(hull[both] - gt[both]).mean())
            if hit.sum() == 0:
                raise RuntimeError(f"{model}/{view}: estimator produced no "
                                   f"finite values inside the mask")
            rows[view] = {"gt_fit": err(z_gt), "hull_fit": err(z_hull),
                          "raw": err(z_raw),
                          "hull_baseline_mae": hull_err,
                          "coverage": float(hit.mean()),
                          "declined_frac": float(1.0 - valid[gt < BG].mean()),
                          "affine_gt": [a_g, b_g], "affine_hull": [a_h, b_h]}

            origin, fwd = view_rays(info, ortho, res)
            for key, z in (("raw", z_raw), ("hull", z_hull), ("gt", z_gt)):
                pts = origin[hit] + np.asarray(fwd)[None, :] * z[hit][:, None]
                sel = np.random.default_rng(0).choice(
                    len(pts), size=min(60000, len(pts)), replace=False)
                clouds[key].append(pts[sel])

        report["models"][model] = {"per_view": rows, "disparity": disparity}
        for key in clouds:
            np.save(os.path.join(args.out, f"{model}_cloud_{key}.npy"),
                    np.concatenate(clouds[key]).astype(np.float32))

        print(f"\n=== {model}"
              f"{'  (disparity space)' if disparity else ''}")
        print(f"{'view':<11}{'cover':>7}{'raw':>9}{'hull-fit':>10}"
              f"{'gt-fit':>9}{'HULL ITSELF':>13}   verdict")
        for v, r in rows.items():
            base = r["hull_baseline_mae"]
            win = "estimator" if r["hull_fit"]["mae"] < base else "hull"
            print(f"{v:<11}{100*r['coverage']:>6.1f}%{r['raw']['mae']:>9.5f}"
                  f"{r['hull_fit']['mae']:>10.5f}{r['gt_fit']['mae']:>9.5f}"
                  f"{base:>13.5f}   {win}")
        hb = np.mean([r["hull_baseline_mae"] for r in rows.values()])
        print(f"{'mean':<11}{'':>7}"
              f"{np.mean([r['raw']['mae'] for r in rows.values()]):>9.5f}"
              f"{np.mean([r['hull_fit']['mae'] for r in rows.values()]):>10.5f}"
              f"{np.mean([r['gt_fit']['mae'] for r in rows.values()]):>9.5f}"
              f"{hb:>13.5f}")

    with open(os.path.join(args.out, "depth_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}/depth_report.json")


if __name__ == "__main__":
    main()
