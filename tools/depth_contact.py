#!/usr/bin/env python3
"""Robust normal integration, and a placement rule that did not survive testing.

Normal integration recovers a surface up to a constant per connected piece:
wherever a discontinuity cut separates two regions, nothing in the normals
relates their depths. Something else has to set each constant, and what set
it before was a weak pull toward the hull. That settles every piece at the
hull's *average* depth over it -- and the hull is in front of the true surface
everywhere except where it touches it, so every piece landed too close to the
camera. Measured on the front view: 15 voxels in front of the truth on average
(the hull itself is 20), with 57% of the pixels then clamped flat onto the
hull. The relief the photometric normals carry was computed and thrown away.

What is kept here:

  integrate()  least-squares integration solved by algebraic multigrid, with
               optional Cauchy IRLS on the edge residuals. Conjugate gradient
               with a Jacobi preconditioner leaves the lowest frequencies of a
               770k-unknown Poisson problem unconverged, and one missed jump is
               enough to wreck a whole piece: with *ground-truth* normals the
               largest piece's relief is off by 9.07 voxels after its best
               offset, against 0.28 for a piece whose boundary was cut
               cleanly. Ten IRLS rounds bring the 9.07 to 4.19.

What did not work -- place(), kept so the measurement can be repeated:

  Push each piece toward the camera until it touches the hull, on the grounds
  that the hull is exact along other views' silhouette rims. With ground-truth
  normals this put the front view 7 voxels *behind* the truth, 84% of pixels
  more than a voxel behind. One constant per piece is set by the piece's worst
  region, and a piece large enough to contain a rim contact is large enough to
  contain a leak. Placement now comes from other views instead: see
  normal_stereo.py and depth_mv.py.

Run:
  python3 tools/depth_contact.py --views refs/lucy_gt --hull-views out/final_views \
      --normals-dir out/ps_normals --out /tmp/depth_contact --irls 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pyamg
from PIL import Image
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discontinuity as dc  # noqa: E402

BG = 1e9


def load(views_dir, hull_dir, normals_dir, view, meta):
    hull = np.load(os.path.join(hull_dir, "depth_npy", f"{view}.npy"))
    hull = np.where(hull < BG, hull, np.nan).astype(np.float64)
    if normals_dir:
        n = np.load(os.path.join(normals_dir, f"{view}.npy")).astype(np.float64)
    else:  # ground-truth world normals, rotated into the camera
        n = np.load(os.path.join(views_dir, "normal_npy", f"{view}.npy"))
        R = np.array(meta["views"][view]["matrix_world"])[:3, :3]
        n = (n.reshape(-1, 3) @ R).reshape(n.shape).astype(np.float64)
    n /= np.linalg.norm(n, axis=2, keepdims=True).clip(1e-9)
    rgb = np.asarray(Image.open(os.path.join(views_dir, "rgb", f"{view}.png"))
                     .convert("L"), dtype=np.float64) / 255.0
    return hull, n, rgb


def cut_edges(n, rgb, hull, hit, budget):
    """Per-axis boolean arrays, True where an edge may be integrated across."""
    keep = []
    for axis in (0, 1):
        det, valid = dc.detectors(n, rgb, hull, hit, axis)
        s = np.zeros_like(det["graze"])
        for key in ("graze", "turn"):
            s = np.maximum(s, dc.rank_normalise(det[key], valid))
        k = valid.copy()
        if budget > 0:
            k &= s < np.percentile(s[valid], 100 - budget)
        keep.append(k)
    return keep  # [rows (axis 0), cols (axis 1)], shapes (H-1,W) and (H,W-1)


def build_edges(n, hit, keep, px, nz_floor):
    """Every integrable edge as (a, b, gradient), a and b pixel indices.

    The gradient on an edge is the mean of its two ends' when both normals are
    usable, else whichever one is.
    """
    H, W = hit.shape
    idx = -np.ones((H, W), dtype=np.int64)
    idx[hit] = np.arange(int(hit.sum()))
    nz = n[..., 2]
    ok = hit & (np.abs(nz) > nz_floor)
    safe = np.where(ok, nz, 1.0)
    g = [np.where(ok, -(n[..., 1] / safe) * px, 0.0),   # d depth / d row
         np.where(ok, (n[..., 0] / safe) * px, 0.0)]    # d depth / d col
    A_, B_, G_ = [], [], []
    for axis in (0, 1):
        oa, ob = dc.edge_pairs(ok, axis)
        ga, gb = dc.edge_pairs(g[axis], axis)
        ia, ib = dc.edge_pairs(idx, axis)
        use = keep[axis] & (oa | ob)
        A_.append(ia[use])
        B_.append(ib[use])
        G_.append(np.where(oa & ob, 0.5 * (ga + gb), np.where(oa, ga, gb))[use])
    return np.concatenate(A_), np.concatenate(B_), np.concatenate(G_), idx


def solve(a, b, grad, w, N, prior, px, reg=1e-9, x0=None, tol=1e-10):
    """Weighted least squares  sum w (z_b - z_a - grad)^2 / px^2, screened
    very weakly toward a plausible depth so every piece is well defined."""
    m = len(a)
    r = np.arange(m)
    sw = np.sqrt(w) / px
    A = sparse.csr_matrix((np.concatenate([sw, -sw]),
                           (np.concatenate([r, r]), np.concatenate([b, a]))),
                          shape=(m, N))
    AtA = (A.T @ A).tocsr()
    lam = reg * AtA.diagonal().mean()
    AtA = AtA + lam * sparse.identity(N, format="csr")
    Atb = A.T @ (grad * sw) + lam * prior
    ml = pyamg.smoothed_aggregation_solver(AtA, symmetry="symmetric",
                                           max_coarse=500)
    return ml.solve(Atb, x0=x0, tol=tol, accel="cg", maxiter=400)


def integrate(hull, n, hit, keep, px, nz_floor, irls=0, sigma=2e-4,
              cut_w=0.05):
    """Depth from gradients, robust to the jumps the cut detector missed.

    One missed occlusion edge is enough to wreck a piece: every path through
    it carries the wrong step to everything beyond. So after a plain solve the
    edges are reweighted by a Cauchy M-estimator on their residual -- an edge
    whose step disagrees with its normals by much more than normal noise is
    almost certainly a jump -- and the solve repeats. Edges whose weight ends
    below cut_w are treated as cut when the pieces are counted.

    Returns depth (up to a constant per piece), a piece label per pixel, and
    the number of pieces.
    """
    a, b, grad, idx = build_edges(n, hit, keep, px, nz_floor)
    N = int(hit.sum())
    prior = hull[hit]
    w = np.ones(len(a))
    z = solve(a, b, grad, w, N, prior, px)
    for _ in range(irls):
        res = (z[b] - z[a]) - grad
        w = 1.0 / (1.0 + (res / sigma) ** 2)
        z = solve(a, b, grad, w, N, prior, px, x0=z)
    live = w >= cut_w
    adj = sparse.coo_matrix((np.ones(int(live.sum())), (a[live], b[live])),
                            shape=(N, N))
    n_pieces, lab = connected_components(adj, directed=False)
    H, W = hit.shape
    depth = np.full((H, W), np.nan)
    depth[hit] = z
    label = -np.ones((H, W), dtype=np.int64)
    label[hit] = lab
    return depth, label, n_pieces


def place(depth, label, hull, q=99.5, min_piece=50):
    """Shift each piece until it touches the hull from behind."""
    hit = label >= 0
    lab = label[hit]
    gap = (hull - depth)[hit]              # >0 where the piece is in front
    order = np.lexsort((gap, lab))
    lab_s, gap_s = lab[order], gap[order]
    starts = np.searchsorted(lab_s, np.arange(lab.max() + 1))
    ends = np.searchsorted(lab_s, np.arange(lab.max() + 1), side="right")
    size = ends - starts
    k = np.clip(starts + np.floor((size - 1) * q / 100).astype(np.int64),
                starts, np.maximum(ends - 1, starts))
    shift = np.where(size > 0, gap_s[np.minimum(k, len(gap_s) - 1)], 0.0)
    out = depth.copy()
    out[hit] = depth[hit] + shift[lab]
    small = np.zeros_like(hit)
    small[hit] = size[lab] < min_piece
    out = np.where(small, hull, out)
    out = np.where(hit, np.maximum(out, hull), np.nan)  # never in front
    return out, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--hull-views", required=True)
    ap.add_argument("--normals-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--nz-floor", type=float, default=0.15)
    ap.add_argument("--q", type=float, default=99.5)
    ap.add_argument("--min-piece", type=int, default=50)
    ap.add_argument("--irls", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=2e-4)
    ap.add_argument("--cut-w", type=float, default=0.05,
                    help="edges whose robust weight ends below this separate "
                         "pieces, each placed against the hull on its own")
    ap.add_argument("--only-views", default=None)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    views = args.only_views.split(",") if args.only_views else list(meta["views"])
    os.makedirs(args.out, exist_ok=True)
    VOX = 1.0 / 1024
    for v in views:
        t0 = time.time()
        hull, n, rgb = load(args.views, args.hull_views, args.normals_dir, v, meta)
        hit = np.isfinite(hull)
        px = meta["ortho_scale"] / hull.shape[0]
        keep = cut_edges(n, rgb, hull, hit, args.budget)
        depth, label, n_pieces = integrate(hull, n, hit, keep, px, args.nz_floor,
                                           args.irls, args.sigma, args.cut_w)
        z, size = place(depth, label, hull, args.q, args.min_piece)
        np.save(os.path.join(args.out, f"{v}.npy"), z.astype(np.float32))
        note = ""
        gt_path = os.path.join(args.views, "depth_npy", f"{v}.npy")
        if os.path.exists(gt_path):
            gt = np.load(gt_path)
            m = hit & (gt < BG)
            e = (z - gt)[m] / VOX
            eh = (hull - gt)[m] / VOX
            note = (f"  MAE {np.abs(e).mean():6.2f} vox (hull {np.abs(eh).mean():6.2f})"
                    f"  mean {e.mean():+6.2f}  behind>1 {100*(e>1).mean():5.2f}%"
                    f"  >3 {100*(e>3).mean():5.2f}%"
                    f"  on-hull {100*np.mean(np.abs(z[m]-hull[m])<1e-7):4.1f}%")
        print(f"  {v:<13} pieces {n_pieces:>6} (>= {args.min_piece}px: "
              f"{int((size >= args.min_piece).sum()):>4}){note}  "
              f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
