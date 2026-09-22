#!/usr/bin/env python3
"""Recover a view's depth by integrating its surface normals, anchored on the
hull.

A normal map gives the surface's tilt, which determines its shape but not its
position: integrating tilt leaves an unknown offset per connected region. The
visual hull supplies that offset. Along each silhouette's boundary the hull
touches the true surface by definition -- that boundary is where the object
stops being visible from this direction -- so the hull's depth there *is* the
true depth, and the integration has a Dirichlet condition for free.

For an orthographic camera the relation is exact and linear:

    dz/dx = s * n_x / n_z        dz/dy = s * n_y / n_z

with no focal length and no per-pixel ray direction. The sign s depends on the
estimator's axis convention and is resolved once, against ground truth, by
--calibrate.

The solve is a screened Poisson problem:

    min  w_grad |grad z - g|^2  +  w_anchor |z - h|^2  +  w_smooth |lap z|^2

with w_anchor large on the rim and small inside, which keeps a wide flat region
far from any silhouette edge from drifting. Being screened rather than pure
Poisson also bounds the condition number, so conjugate gradients converge.

Run:
  python3 tools/integrate_normals.py --views refs/lucy_gt \
      --hull-views out/final_views --normals GT --view 01_front --res 1024
"""
import argparse
import json
import os
import time

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

BG = 1e9


def load_view(views_dir, hull_dir, view, normals_dir, res):
    """Depth, hull depth, mask and camera-space normals at one resolution."""
    gt = np.load(os.path.join(views_dir, "depth_npy", f"{view}.npy"))
    hull = np.load(os.path.join(hull_dir, "depth_npy", f"{view}.npy"))
    meta = json.load(open(os.path.join(views_dir, "cameras.json")))
    R = np.array(meta["views"][view]["matrix_world"], dtype=np.float64)[:3, :3]

    if normals_dir == "GT":
        n = np.load(os.path.join(views_dir, "normal_npy", f"{view}.npy"))
        n = (n.reshape(-1, 3) @ R).reshape(n.shape)      # world -> camera
    else:
        n = np.load(os.path.join(normals_dir, f"{view}.npy")).astype(np.float64)

    full = gt.shape[0]
    if res != full:
        k = full // res
        if k * res != full:
            raise ValueError(f"{full} is not a multiple of {res}")
        # Block-average the float fields and take any covered pixel for the
        # mask, so the rim survives downsampling.
        def blk(a):
            sh = (res, k, res, k) + a.shape[2:]
            return a.reshape(sh).mean(axis=(1, 3))
        hit_full = gt < BG
        hit = hit_full.reshape(res, k, res, k).any(axis=(1, 3))
        gt = np.where(hit_full, gt, np.nan)
        hull_full = hull < BG
        hull = np.where(hull_full, hull, np.nan)
        import warnings
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # blocks lying entirely outside the mask average to nan by design
            warnings.simplefilter("ignore", RuntimeWarning)
            gt = np.nanmean(gt.reshape(res, k, res, k), axis=(1, 3))
            hull = np.nanmean(hull.reshape(res, k, res, k), axis=(1, 3))
            n = blk(n)
    else:
        hit = gt < BG
        hull = np.where(hull < BG, hull, np.nan)

    nn = np.linalg.norm(n, axis=2, keepdims=True)
    n = n / np.clip(nn, 1e-9, None)
    return gt, hull, hit, n, meta


def gradients_from_normals(n, sign, pixel_size, nz_floor=0.15):
    """Depth gradient implied by the normals, in depth units *per pixel*.

    n_x/n_z is a change in depth per unit of world distance, while the solve's
    finite differences are per pixel, so the ratio is scaled by the pixel size.
    Leaving that out inflates every gradient by the resolution -- 465x at
    res 512 -- and the solve chases a surface hundreds of times too steep.

    The row axis is negated because image rows increase downward while the
    camera's up axis points the other way.

    Where the surface turns edge-on, n_z approaches zero and the ratio
    diverges. Those pixels are dropped from the gradient term rather than
    clamped, and the solve interpolates across them.
    """
    nz = n[..., 2]
    ok = np.abs(nz) > nz_floor
    safe = np.where(ok, nz, 1.0)
    gx = sign * (n[..., 0] / safe) * pixel_size
    gy = -sign * (n[..., 1] / safe) * pixel_size
    return np.where(ok, gx, 0.0), np.where(ok, gy, 0.0), ok


def build_rim(hit, width=3):
    """Pixels within `width` of the silhouette boundary."""
    inner = ndimage.binary_erosion(hit, iterations=width)
    return hit & ~inner


def solve_view(hull, hit, gx, gy, gok, *, w_rim, w_inner, w_grad, w_smooth,
               anchor_blur, tol, maxiter, pixel_size=1.0, x0=None,
               pinned=None):
    """Screened Poisson solve over the masked pixels.

    Gradient equations are scaled by 1/pixel_size so their residuals are in
    world units per world unit rather than per pixel. Without that the two
    terms sit on different scales -- gradient residuals around 0.002 against
    anchor residuals around 0.02 at res 512 -- and the anchor quietly wins a
    contest the weights say it should lose, at a strength that also changes
    with resolution.
    """
    w_grad = w_grad / max(pixel_size, 1e-12)
    H, W = hit.shape
    idx = -np.ones((H, W), dtype=np.int64)
    n_unk = int(hit.sum())
    idx[hit] = np.arange(n_unk)

    # The hull is not an equality anchor anywhere, including the rim.
    #
    # The original design pinned depth to the hull along the silhouette
    # boundary, on the theory that the hull touches the true surface there.
    # Measured: the hull's depth error is 0.0231 on the rim against 0.0194
    # inside -- the rim is the *worst* place, not the best, because near a
    # silhouette the surface is nearly parallel to the view ray, so one voxel
    # of 3D slack becomes a large depth error. Nailing the boundary to a value
    # wrong by 0.023 caps the whole solve at 0.023, which is exactly what
    # feeding it exact gradients produced.
    #
    # What the hull is, per-pixel and verified on every pixel of this view, is
    # a *lower bound*: hull depth never exceeds true depth. So it enters as an
    # inequality plus a weak pull, and the answer settles as close to the hull
    # as the gradients allow -- touching it where it is tight, standing behind
    # it where it is not.
    anchor = np.where(np.isfinite(hull), hull, np.nan)
    filled = np.where(np.isfinite(anchor), anchor, np.nanmean(anchor))
    target = ndimage.gaussian_filter(filled, anchor_blur) if anchor_blur > 0 \
        else filled
    w_a = np.full(hit.shape, w_inner, dtype=np.float64)
    if pinned is not None:
        # active-set pass: pixels the previous round pushed through the floor
        # are held on it while the rest re-solve around them
        w_a = np.where(pinned, w_rim, w_a)
        target = np.where(pinned, filled, target)
    w_a = np.where(np.isfinite(anchor), w_a, w_inner * 0.05)

    rows, cols, vals, rhs = [], [], [], []
    eq = 0

    def add(pairs, b, weight):
        nonlocal eq
        for c, v in pairs:
            rows.append(eq)
            cols.append(c)
            vals.append(v * weight)
        rhs.append(b * weight)
        eq += 1

    # gradient equations, vectorised per direction
    def grad_block(shift_axis, g, name):
        nonlocal eq
        a = hit.copy()
        b = np.roll(hit, -1, axis=shift_axis)
        if shift_axis == 0:
            b[-1] = False
        else:
            b[:, -1] = False
        both = a & b & gok
        ia = idx[both]
        ib = idx[np.roll(both, 1, axis=shift_axis)] if False else None
        # index of the shifted neighbour
        sh = np.roll(idx, -1, axis=shift_axis)
        ib = sh[both]
        m = len(ia)
        r = np.arange(eq, eq + m)
        rows.extend(np.repeat(r, 2).tolist())
        cols.extend(np.stack([ib, ia], 1).ravel().tolist())
        vals.extend(np.tile([w_grad, -w_grad], m).tolist())
        rhs.extend((g[both] * w_grad).tolist())
        eq += m
        return m

    m_y = grad_block(0, gy, "y")
    m_x = grad_block(1, gx, "x")

    # anchor equations
    ia = idx[hit]
    wa = w_a[hit]
    r = np.arange(eq, eq + len(ia))
    rows.extend(r.tolist())
    cols.extend(ia.tolist())
    vals.extend(wa.tolist())
    rhs.extend((target[hit] * wa).tolist())
    eq += len(ia)

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(eq, n_unk)).tocsr()
    b = np.asarray(rhs, dtype=np.float64)
    AtA = (A.T @ A).tocsr()
    Atb = A.T @ b

    if w_smooth > 0:
        lap = laplacian_operator(idx, hit) * w_smooth
        AtA = AtA + (lap.T @ lap)

    d = AtA.diagonal()
    d[d == 0] = 1.0
    M = sparse.diags(1.0 / d)
    guess = x0 if x0 is not None else target[hit]
    z, info = cg(AtA, Atb, x0=guess, rtol=tol, maxiter=maxiter, M=M)

    out = np.full(hit.shape, np.nan)
    out[hit] = z
    return out, info, {"unknowns": n_unk, "grad_eqs": m_x + m_y}


def laplacian_operator(idx, hit):
    """Five-point Laplacian restricted to the mask."""
    H, W = hit.shape
    n = int(hit.sum())
    rows, cols, vals = [], [], []
    centre = idx[hit]
    r = np.arange(n)
    rows.extend(r.tolist())
    cols.extend(centre.tolist())
    deg = np.zeros(n)
    for ax, sgn in ((0, 1), (0, -1), (1, 1), (1, -1)):
        nb = np.roll(idx, -sgn, axis=ax)
        nbm = np.roll(hit, -sgn, axis=ax)
        if ax == 0:
            (nbm[-1:] if sgn == 1 else nbm[:1]).fill(False)
        else:
            (nbm[:, -1:] if sgn == 1 else nbm[:, :1]).fill(False)
        good = hit & nbm
        gi = idx[good]
        gj = nb[good]
        rows.extend(gi.tolist())
        cols.extend(gj.tolist())
        vals.extend([-1.0] * len(gi))
        np.add.at(deg, gi, 1.0)
    vals = deg.tolist() + vals
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def score(z, gt, hull, hit):
    """Compare a solved depth against the truth, and against the hull it has
    to beat to be worth anything."""
    m = hit & np.isfinite(z) & np.isfinite(gt)
    mh = m & np.isfinite(hull)
    return {
        "mae": float(np.abs(z[m] - gt[m]).mean()),
        "rmse": float(np.sqrt(((z[m] - gt[m]) ** 2).mean())),
        "p95": float(np.percentile(np.abs(z[m] - gt[m]), 95)),
        "hull_mae": float(np.abs(hull[mh] - gt[mh]).mean()),
        "in_front_of_hull_frac": float((z[mh] < hull[mh] - 1e-6).mean()),
        "pixels": int(m.sum()),
    }


def calibrate_sign(views_dir, hull_dir, view, normals_dir, res):
    """Estimators differ in axis convention; resolve it once against truth.

    Both sides go through load_view so they are downsampled identically --
    comparing a full-resolution reference against a reduced mask is how this
    went wrong the first time.
    """
    _, _, hit, n, _ = load_view(views_dir, hull_dir, view, normals_dir, res)
    _, _, _, ref, _ = load_view(views_dir, hull_dir, view, "GT", res)
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                cand = n * np.array([sx, sy, sz])
                c = np.clip((cand[hit] * ref[hit]).sum(1), -1, 1)
                err = float(np.degrees(np.arccos(c)).mean())
                if best is None or err < best[0]:
                    best = (err, (sx, sy, sz))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--hull-views", required=True)
    ap.add_argument("--normals", required=True,
                    help="directory of per-view normal .npy, or GT")
    ap.add_argument("--view", default="01_front")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--sign", type=float, default=1.0)
    ap.add_argument("--w-rim", type=float, default=40.0)
    ap.add_argument("--w-inner", type=float, default=0.05)
    ap.add_argument("--w-grad", type=float, default=1.0)
    ap.add_argument("--w-smooth", type=float, default=0.0)
    ap.add_argument("--anchor-blur", type=float, default=6.0)
    ap.add_argument("--nz-floor", type=float, default=0.15)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--active-set", type=int, default=6,
                   help="rounds of solve-and-pin against the hull floor")
    ap.add_argument("--clamp", action="store_true",
                    help="clamp the result to lie behind the hull")
    ap.add_argument("--budget", type=float, default=None,
                    help="maximum distance the solve may run behind the hull")
    ap.add_argument("--out", default=None)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    if args.calibrate:
        err, signs = calibrate_sign(args.views, args.hull_views, args.view,
                                    args.normals, args.res)
        print(f"best axis signs {signs}  mean angular error {err:.2f} deg")
        return

    t0 = time.time()
    gt, hull, hit, n, meta = load_view(args.views, args.hull_views, args.view,
                                    args.normals, args.res)
    px = meta["ortho_scale"] / args.res
    gx, gy, gok = gradients_from_normals(n, args.sign, px, args.nz_floor)
    print(f"view {args.view}  res {args.res}  masked {hit.sum():,} px  "
          f"pixel {px:.6f}  usable gradient {100*gok[hit].mean():.1f}%")

    # Active set on the floor d >= hull: solve, see who fell through, hold
    # those on the floor, solve again. The set is monotone in practice and
    # settles in a handful of rounds.
    pinned = None
    z = None
    floor = np.where(np.isfinite(hull), hull, -np.inf)
    for it in range(args.active_set):
        z, info, stats = solve_view(
            hull, hit, gx, gy, gok,
            w_rim=args.w_rim, w_inner=args.w_inner, w_grad=args.w_grad,
            w_smooth=args.w_smooth, anchor_blur=args.anchor_blur,
            tol=args.tol, maxiter=args.maxiter, pixel_size=px,
            x0=(z[hit] if z is not None else None), pinned=pinned)
        below = hit & np.isfinite(z) & (z < floor - 1e-7)
        new_pinned = below if pinned is None else (pinned | below)
        n_new = int(below.sum() if pinned is None
                    else (below & ~pinned).sum())
        print(f"  pass {it}: cg {info}  below floor {100*below[hit].mean():5.2f}%"
              f"  newly pinned {n_new:,}")
        pinned = new_pinned
        if n_new == 0:
            break
    z = np.maximum(z, floor)
    print(f"solved {stats['unknowns']:,} unknowns from "
          f"{stats['grad_eqs']:,} gradient equations  {time.time()-t0:.1f}s")

    if args.clamp or args.budget is not None:
        lo = hull
        hi = hull + args.budget if args.budget is not None else np.inf
        before = z.copy()
        z = np.clip(z, lo, hi)
        moved = np.isfinite(before) & np.isfinite(z) & (np.abs(before - z) > 1e-9)
        print(f"clamped {100*moved[hit].mean():.2f}% of pixels into "
              f"[hull, hull+{args.budget}]")

    s = score(z, gt, hull, hit)
    print(f"\n{'':<16}{'MAE':>10}{'RMSE':>10}{'p95':>10}")
    print(f"{'integrated':<16}{s['mae']:>10.5f}{s['rmse']:>10.5f}{s['p95']:>10.5f}")
    print(f"{'hull baseline':<16}{s['hull_mae']:>10.5f}")
    gain = 100 * (1 - s["mae"] / s["hull_mae"])
    verdict = "BEATS the hull" if s["mae"] < s["hull_mae"] else "loses to the hull"
    print(f"\n{verdict}: {gain:+.1f}% vs the depth it was anchored on")
    print(f"in front of the hull (should be 0): "
          f"{100*s['in_front_of_hull_frac']:.3f}%")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.save(args.out, z.astype(np.float32))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
