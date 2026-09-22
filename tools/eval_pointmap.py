#!/usr/bin/env python3
"""Score a multi-view model's joint point map against the true surface.

A multi-view network that emits world points has already performed the
integration this project is about. Its output arrives in an arbitrary frame
and at an arbitrary scale, so it is aligned to the truth by a similarity
transform first (Umeyama), then scored. The alignment is a gift -- it removes
seven degrees of freedom the model would otherwise have to get right -- so the
result is an upper bound on what the network's reconstruction is worth, and a
bad number after alignment cannot be blamed on the frame.

Run:
  python3 tools/eval_pointmap.py --points out/est_dl/vggt_world_points.npy \
      --gt-mesh assets/lucy_le.ply --views refs/lucy_gt --out out/m2b
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity transform mapping src onto dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    C = d.T @ s / len(src)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s ** 2).sum() / len(src)
    c = (D * np.diag(S)).sum() / var if with_scale else 1.0
    t = mu_d - c * R @ mu_s
    return c, R, t


def read_ply(path):
    with open(path, "rb") as f:
        head = f.read(4096)
        end = head.find(b"end_header")
        offset = head.find(b"\n", end) + 1
        n_vert = n_face = 0
        for ln in head[:end].decode("ascii", "replace").splitlines():
            p = ln.split()
            if len(p) >= 3 and p[0] == "element":
                if p[1] == "vertex":
                    n_vert = int(p[2])
                elif p[1] == "face":
                    n_face = int(p[2])
        f.seek(offset)
        v = np.fromfile(f, dtype="<f4", count=n_vert * 3).reshape(n_vert, 3)
        fa = np.fromfile(f, dtype=np.dtype([("n", "u1"), ("v", "<i4", 3)]),
                         count=n_face)["v"]
    return v, fa


def sample_surface(v, f, n, rng, chunk=4_000_000):
    areas = np.empty(len(f), dtype=np.float64)
    for i in range(0, len(f), chunk):
        g = f[i:i + chunk]
        a, b, c = v[g[:, 0]], v[g[:, 1]], v[g[:, 2]]
        areas[i:i + chunk] = 0.5 * np.linalg.norm(np.cross(b - a, c - a), 1)
    cdf = np.cumsum(areas)
    idx = np.clip(np.searchsorted(cdf, rng.random(n) * cdf[-1]), 0, len(f) - 1)
    g = f[idx]
    a, b, c = v[g[:, 0]], v[g[:, 1]], v[g[:, 2]]
    u, w = rng.random((n, 1)), rng.random((n, 1))
    flip = (u + w) > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    return (a + u * (b - a) + w * (c - a)).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="(S,H,W,3) world point map")
    ap.add_argument("--conf", default=None)
    ap.add_argument("--conf-quantile", type=float, default=0.3,
                    help="drop the least confident fraction before aligning")
    ap.add_argument("--gt-mesh", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=300_000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    norm = meta["normalization"]
    views = list(meta["views"])

    wp = np.load(args.points).astype(np.float64)
    print(f"point map {wp.shape}")
    S, H, W = wp.shape[:3]

    # Masks come from the reference renders, resized to the model's grid.
    keep = np.zeros((S, H, W), dtype=bool)
    for i, v in enumerate(views[:S]):
        m = Image.open(os.path.join(args.views, "mask", f"{v}.png")).convert("L")
        keep[i] = np.asarray(m.resize((W, H), Image.NEAREST)) > 127
    print(f"masked points: {keep.sum():,} of {S*H*W:,}")

    if args.conf and os.path.exists(args.conf):
        c = np.load(args.conf).astype(np.float64)
        thr = np.quantile(c[keep], args.conf_quantile)
        keep &= c > thr
        print(f"after confidence cut (>{thr:.3f}): {keep.sum():,}")

    pts = wp[keep]
    pts = pts[np.isfinite(pts).all(1)]

    # ground truth in the reference frame
    gv, gf = read_ply(args.gt_mesh)
    yaw = np.radians(meta.get("yaw_deg", 0.0))
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    gv = (gv.astype(np.float64) @ R.T) * norm["applied_scale"] \
        + np.array(norm["applied_offset"])
    rng = np.random.default_rng(0)
    gp = sample_surface(gv.astype(np.float32), gf, args.samples, rng)

    sub = pts[rng.choice(len(pts), size=min(args.samples, len(pts)),
                         replace=False)]

    # Align by nearest-neighbour correspondence, refined a few times. The
    # first fit only has to be close enough for the correspondences to improve.
    tree_gt = cKDTree(gp)
    cur = sub.copy()
    for it in range(6):
        _, j = tree_gt.query(cur, workers=-1)
        c, Rm, t = umeyama(sub, gp[j])
        cur = (c * (Rm @ sub.T).T + t)
        d = np.linalg.norm(cur - gp[j], axis=1)
        print(f"  align {it}: scale {c:.4f}  mean residual {d.mean():.5f}")

    d_pg, _ = tree_gt.query(cur, workers=-1)
    d_gp, _ = cKDTree(cur).query(gp, workers=-1)
    res = {
        "points_used": int(len(pts)),
        "similarity_scale": float(c),
        "chamfer_mean": float((d_pg.mean() + d_gp.mean()) / 2),
        "pred_to_gt_mean": float(d_pg.mean()),
        "pred_to_gt_p95": float(np.percentile(d_pg, 95)),
        "gt_to_pred_mean": float(d_gp.mean()),
        "gt_to_pred_p95": float(np.percentile(d_gp, 95)),
    }
    print("\n" + "=" * 56)
    print(f"points used      {res['points_used']:,}")
    print(f"Chamfer (sym)    {res['chamfer_mean']:.6f}   "
          f"object height == 1.0")
    print(f"  pred -> gt     mean {res['pred_to_gt_mean']:.6f}  "
          f"p95 {res['pred_to_gt_p95']:.6f}")
    print(f"  gt -> pred     mean {res['gt_to_pred_mean']:.6f}  "
          f"p95 {res['gt_to_pred_p95']:.6f}")
    print("=" * 56)
    print("compare against the M1 visual hull: Chamfer 0.019788")

    with open(os.path.join(args.out, "pointmap_eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.save(os.path.join(args.out, "pointmap_aligned.npy"),
            cur.astype(np.float32))


if __name__ == "__main__":
    main()
