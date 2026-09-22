#!/usr/bin/env python3
"""Carve the hull using each view's surface relief, with absolute depth
resolved by agreement between views rather than within one view.

S2's gate test established that a view's integrated depth is trustworthy in
*shape* and not in *offset*: dropping the edge-on pixels fragments the domain,
and a fragment's absolute depth is not observable from the view it lives in.

So it is never asserted. Each fragment is placed at the most forward position
the hull permits -- touching the floor at exactly one point and standing behind
it everywhere else. That position is provably at or in front of the true
surface, so carving to it can only remove voxels that are genuinely outside:

    d = r + min{ c : r + c >= hull }  <=  d_true

The carve is therefore conservative by construction, and containment survives
without a budget parameter. Each round tightens the hull, which raises the
floor, which lets the next round place fragments further back and carve more.
Views resolve each other's offsets by iteration.

Run:
  python3 tools/carve_relief.py --occ out/final_mesh_occ.npz \
      --views refs/lucy_gt --normals GT --rounds 4 --out out/relief
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np
from scipy import ndimage

BG = 1e9
_spec = importlib.util.spec_from_file_location(
    "iz", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "integrate_normals.py"))
iz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iz)


def axis_view(matrix_world):
    """(view axis, sign, u axis, u sign, v axis, v sign) for an axis-aligned
    orthographic camera, or None if the camera is diagonal."""
    m = np.array(matrix_world, dtype=np.float64)
    out = []
    for col in (0, 1, 2):
        v = m[:3, col]
        a = int(np.argmax(np.abs(v)))
        if abs(abs(v[a]) - 1.0) > 1e-6:
            return None
        out.append((a, float(np.sign(v[a]))))
    return out  # [right, up, back]


def view_world_coords(centers, cam_loc, basis):
    """World coordinates of each grid-view pixel, in image order.

    The grid view spans the voxel grid's extent; the rendered images span the
    camera's ortho_scale over a square frame. Resampling one onto the other by
    index pastes background over the subject, which is how the first run found
    zero usable gradient.
    """
    (a_u, s_u), (a_v, s_v), _ = basis
    u = centers[a_u].copy()
    if s_u < 0:
        u = u[::-1]
    v = centers[a_v].copy()
    if s_v > 0:                 # image rows run opposite the up axis
        v = v[::-1]
    return u, v, (a_u, s_u), (a_v, s_v)


def sample_image(field, centers, cam_loc, basis, ortho, res_img):
    """Sample a rendered res_img-square field at the grid view's pixels."""
    u, v, (a_u, s_u), (a_v, s_v) = view_world_coords(centers, cam_loc, basis)
    u_cam = s_u * (u - cam_loc[a_u])
    v_cam = s_v * (v - cam_loc[a_v])
    col = np.clip(((u_cam / ortho + 0.5) * res_img).astype(int),
                  0, res_img - 1)
    row = np.clip(((0.5 - v_cam / ortho) * res_img).astype(int),
                  0, res_img - 1)
    return field[np.ix_(row, col)]


def depth_from_grid(occ, centers, cam_loc, basis):
    """Depth of the first occupied voxel along the view direction, straight off
    the grid. Rendering this would need Blender once per view per round; for an
    axis-aligned camera it is an argmax."""
    (a_u, s_u), (a_v, s_v), (a_b, s_b) = basis
    # march along the axis the camera looks down, from the camera inwards
    order = np.arange(occ.shape[a_b])
    if s_b > 0:          # camera sits at +axis, looks toward -axis
        order = order[::-1]
    moved = np.moveaxis(occ, (a_u, a_v, a_b), (0, 1, 2))[:, :, order]
    any_hit = moved.any(axis=2)
    first = moved.argmax(axis=2)
    coord = centers[a_b][order][np.clip(first, 0, len(order) - 1)]
    depth = np.where(any_hit, np.abs(coord - cam_loc[a_b]), np.inf)
    if s_u < 0:
        depth = depth[::-1]
        any_hit = any_hit[::-1]
    if s_v > 0:                    # image rows run opposite the up axis
        depth = depth[:, ::-1]
        any_hit = any_hit[:, ::-1]
    return depth.T, any_hit.T      # (v, u) == (row, col)


def resample(a, shape):
    """Nearest-neighbour resample of a 2D or 3D-channel field."""
    H, W = shape
    ri = (np.arange(H) * (a.shape[0] / H)).astype(int).clip(0, a.shape[0] - 1)
    ci = (np.arange(W) * (a.shape[1] / W)).astype(int).clip(0, a.shape[1] - 1)
    return a[np.ix_(ri, ci)] if a.ndim == 2 else a[np.ix_(ri, ci)]


def place_fragments(relief, hull, valid, min_size=64, quantile=98.0):
    """Offset each connected fragment to the most forward legal position.

    A fragment's own view says nothing about its absolute depth, so nothing is
    asserted: it is pushed forward until one pixel touches the hull. That is a
    lower bound on the true surface, which is what makes carving to it safe.
    """
    lab, n = ndimage.label(valid)
    out = np.full(relief.shape, np.nan)
    placed = 0
    for i in range(1, n + 1):
        sel = lab == i
        if sel.sum() < min_size:
            continue
        h = hull[sel]
        r = relief[sel]
        good = np.isfinite(h) & np.isfinite(r)
        if not good.any():
            continue
        # The smallest offset that keeps the whole fragment behind the hull is
        # max(h - r), but a max is decided by one pixel: a single outlier drags
        # the entire fragment backwards and everything in front of it gets
        # carved. A high quantile is used instead, and the floor is restored by
        # clamping, which bounds one bad pixel's influence to itself.
        c = np.percentile(h[good] - r[good], quantile)
        out[sel] = np.maximum(relief[sel] + c, hull[sel])
        placed += 1
    return out, placed, n


def carve_with_depth(occ, centers, cam_loc, basis, depth, margin):
    """Remove voxels lying in front of the observed surface.

    Voxels only ever leave. There is no branch that adds one, so spikes, holes
    and runaway geometry have no mechanism regardless of how wrong the depth is.
    """
    (a_u, s_u), (a_v, s_v), (a_b, s_b) = basis
    d = depth.T                                  # back to (u, v)
    if s_v > 0:
        d = d[:, ::-1]
    if s_u < 0:
        d = d[::-1]
    # voxel depth along the view axis
    vd = np.abs(centers[a_b] - cam_loc[a_b])
    shape = [1, 1, 1]
    shape[a_b] = len(vd)
    vd = vd.reshape(shape)

    # No estimate means carve nothing. Mapping "unknown" to +inf instead
    # reads as "the surface is infinitely far back", which carves the entire
    # grid -- the default has to fail closed, not open.
    dd = np.where(np.isfinite(d), d, -np.inf)
    ds = [1, 1, 1]
    ds[a_u] = d.shape[0]
    ds[a_v] = d.shape[1]
    axes = sorted([(a_u, 0), (a_v, 1)])
    arr = dd if axes[0][1] == 0 else dd.T
    front = vd < (arr.reshape(ds) - margin)
    removed = int((occ & front).sum())
    occ &= ~front
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--normals", default="GT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="extra clearance left in front of the surface, in "
                         "object units")
    ap.add_argument("--nz-floor", type=float, default=0.15)
    ap.add_argument("--w-inner", type=float, default=0.05)
    ap.add_argument("--anchor-blur", type=float, default=4.0)
    ap.add_argument("--min-fragment", type=int, default=64)
    ap.add_argument("--quantile", type=float, default=98.0,
                   help="percentile of (hull - relief) used as a fragment's "
                        "offset; 100 is the exact floor and is outlier-driven")
    ap.add_argument("--maxiter", type=int, default=3000)
    args = ap.parse_args()

    z = np.load(args.occ)
    dims = z["dims"]
    occ = np.unpackbits(z["occ"])[:int(np.prod(dims))].reshape(dims).astype(bool)
    lo, h = z["lo"], float(z["h"])
    centers = [lo[a] + (np.arange(dims[a]) + 0.5) * h for a in range(3)]
    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    ortho = meta["ortho_scale"]

    usable = {}
    for v, info in meta["views"].items():
        b = axis_view(info["matrix_world"])
        if b is not None:
            usable[v] = (b, np.array(info["location"], dtype=np.float64))
    print(f"grid {tuple(dims)}  {occ.sum():,} voxels")
    print(f"axis-aligned views usable: {len(usable)} of {len(meta['views'])}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    start = occ.sum()
    for rnd in range(args.rounds):
        t0 = time.time()
        total = 0
        for v, (basis, cam_loc) in usable.items():
            hull_d, hull_hit = depth_from_grid(occ, centers, cam_loc, basis)
            H, W = hull_d.shape

            if args.normals == "GT":
                n = np.load(os.path.join(args.views, "normal_npy", f"{v}.npy"))
                R = np.array(meta["views"][v]["matrix_world"])[:3, :3]
                n = (n.reshape(-1, 3) @ R).reshape(n.shape)
            else:
                n = np.load(os.path.join(args.normals, f"{v}.npy")).astype(float)
            n = sample_image(n, centers, cam_loc, basis, ortho, n.shape[0])
            n /= np.linalg.norm(n, axis=2, keepdims=True).clip(1e-9)

            px = ortho / max(H, W)
            gx, gy, gok = iz.gradients_from_normals(n, 1.0, px, args.nz_floor)
            hit = np.isfinite(hull_d) & hull_hit
            gok = gok & hit

            hull_nan = np.where(hit, hull_d, np.nan)
            relief, info, _ = iz.solve_view(
                hull_nan, hit, gx, gy, gok,
                w_rim=1e5, w_inner=args.w_inner, w_grad=1.0, w_smooth=0.0,
                anchor_blur=args.anchor_blur, tol=1e-8, maxiter=args.maxiter,
                pixel_size=px, pinned=None)

            placed, n_placed, n_frag = place_fragments(
                relief, hull_nan, gok, args.min_fragment, args.quantile)
            removed = carve_with_depth(occ, centers, cam_loc, basis, placed,
                                       args.margin)
            total += removed
            print(f"  r{rnd} {v:<11} {H}x{W}  fragments {n_placed}/{n_frag}  "
                  f"removed {removed:>9,}")
        print(f"round {rnd}: removed {total:,}  remaining {occ.sum():,}  "
              f"{time.time()-t0:.0f}s")
        if total == 0:
            break

    print(f"\ntotal removed {start - occ.sum():,} "
          f"({100*(start-occ.sum())/start:.2f}%)")
    np.savez_compressed(args.out + "_occ.npz", occ=np.packbits(occ),
                        dims=dims, lo=lo, h=h)
    from skimage import measure
    padded = np.pad(occ.astype(np.float32), 1)
    padded = ndimage.gaussian_filter(padded, 1.0, truncate=3.0)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5,
                                                spacing=(h, h, h))
    verts += lo - h
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from visual_hull import write_ply
    write_ply(args.out + ".ply", verts.astype(np.float32),
              faces.astype(np.int32))
    print(f"wrote {args.out}.ply  {len(faces):,} faces")


if __name__ == "__main__":
    main()
