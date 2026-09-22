#!/usr/bin/env python3
"""Carve a visual hull from a set of axis-aligned orthographic silhouettes.

This is the floor of the reconstruction: the hull is a proven outer bound on
the true surface, it is watertight by construction, and no amount of bad input
can make it fly apart. It is fat and it cannot see concavities. That is the
trade being made on purpose -- everything downstream refines inside this cage
rather than starting from nothing.

Because the cameras are axis-aligned and orthographic, projecting a voxel into
a view does not need a matrix multiply per voxel: a voxel's pixel in the front
view depends only on its X and Z indices. Each view therefore collapses to a
2D lookup broadcast along the remaining axis, so carving a 26M voxel grid is
six array ANDs instead of 156M projections.

Run:
  python3 tools/visual_hull.py --views refs/lucy_gt --out out/lucy_hull \
      --res 512
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure


def axis_and_sign(vec, tol=1e-6):
    """An axis-aligned unit vector as (axis index, sign), or None."""
    a = int(np.argmax(np.abs(vec)))
    if abs(abs(vec[a]) - 1.0) > tol or np.abs(np.delete(vec, a)).max() > tol:
        return None
    return a, float(np.sign(vec[a]))


def view_basis(matrix_world):
    """Camera right/up axes in world space, plus the camera location."""
    m = np.array(matrix_world, dtype=np.float64)
    return m[:3, 0], m[:3, 1], m[:3, 3]


def carve_axis_aligned(occ, sil, centers, right, up, loc, ortho, res_px):
    """Fast path for a camera whose axes are world axes.

    A voxel's pixel then depends on only two of its three indices, so the view
    reduces to a 2D lookup broadcast along the third -- no per-voxel projection
    at all.
    """
    (a_u, s_u), (a_v, s_v) = axis_and_sign(right), axis_and_sign(up)
    u = s_u * centers[a_u] - s_u * loc[a_u]
    v = s_v * centers[a_v] - s_v * loc[a_v]
    col = np.clip(((u / ortho + 0.5) * res_px).astype(np.int64), 0, res_px - 1)
    row = np.clip(((0.5 - v / ortho) * res_px).astype(np.int64), 0, res_px - 1)
    lut = sil[np.ix_(row, col)]
    axes = [a_v, a_u]
    if a_v > a_u:
        lut = lut.T
        axes = [a_u, a_v]
    shape = [1, 1, 1]
    shape[axes[0]] = lut.shape[0]
    shape[axes[1]] = lut.shape[1]
    occ &= lut.reshape(shape)


def carve_general(occ, sil, centers, right, up, loc, ortho, res_px, slab=24):
    """General path for a diagonal camera.

    A voxel's image coordinate is still separable -- u = rx*x + ry*y + rz*z is
    a sum of three one-dimensional terms -- so it is built by broadcasting
    rather than by projecting points. Slabs along x bound the peak memory,
    which otherwise runs to several hundred megabytes of float per view.
    """
    nx, ny, nz = occ.shape
    ux = right[0] * centers[0] - float(right @ loc)
    uy = right[1] * centers[1]
    uz = right[2] * centers[2]
    vx = up[0] * centers[0] - float(up @ loc)
    vy = up[1] * centers[1]
    vz = up[2] * centers[2]
    for i0 in range(0, nx, slab):
        i1 = min(i0 + slab, nx)
        u = (ux[i0:i1, None, None] + uy[None, :, None] + uz[None, None, :])
        v = (vx[i0:i1, None, None] + vy[None, :, None] + vz[None, None, :])
        col = np.clip(((u / ortho + 0.5) * res_px).astype(np.int32),
                      0, res_px - 1)
        row = np.clip(((0.5 - v / ortho) * res_px).astype(np.int32),
                      0, res_px - 1)
        occ[i0:i1] &= sil[row, col]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True, help="directory with cameras.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512,
                    help="voxels along the longest object axis")
    ap.add_argument("--mask-threshold", type=float, default=0.0,
                    help="alpha above which a pixel counts as silhouette; the "
                         "default counts any coverage, which keeps the hull a "
                         "true outer bound at the cost of one voxel of fat")
    ap.add_argument("--center-test", action="store_true",
                   help="mark a voxel occupied only if its centre projects "
                        "inside every silhouette. Faster, but half a voxel of "
                        "the true surface can fall outside the result, which "
                        "costs the outer-bound guarantee")
    ap.add_argument("--use-views", default=None,
                   help="comma-separated subset of view names to carve with")
    ap.add_argument("--smooth", type=float, default=0.0,
                   help="gaussian sigma in voxels applied to the occupancy "
                        "before the isosurface. Marching cubes on a binary "
                        "grid terraces, and that terracing is high-frequency "
                        "noise any normal-driven refinement would chase")
    ap.add_argument("--level", type=float, default=0.5,
                   help="isosurface level. Below 0.5 dilates, which is how "
                        "smoothing keeps the containment guarantee")
    ap.add_argument("--pad", type=float, default=0.02,
                    help="fraction of the bounding box added around the grid")
    args = ap.parse_args()

    root = os.path.abspath(args.views)
    meta = json.load(open(os.path.join(root, "cameras.json")))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    ortho = meta["ortho_scale"]
    res_px = meta["resolution"][0]
    box = np.array(meta["normalization"]["normalized_size"], dtype=np.float64)

    # Cubic voxels, sized off the longest axis, over a slightly padded box.
    h = box.max() / args.res
    lo = -box / 2 - box.max() * args.pad
    hi = box / 2 + box.max() * args.pad
    dims = np.maximum(np.ceil((hi - lo) / h).astype(int), 2)
    centers = [lo[a] + (np.arange(dims[a]) + 0.5) * h for a in range(3)]

    print(f"grid        : {dims[0]} x {dims[1]} x {dims[2]} "
          f"= {np.prod(dims)/1e6:.1f}M voxels, {h:.6f} per side")
    print(f"box         : {box.round(4).tolist()}  pad {args.pad}")
    print(f"mask thresh : alpha > {args.mask_threshold}")

    px = ortho / res_px
    # A voxel spans several pixels, so testing only its centre can drop a voxel
    # that the true surface passes through. Dilating the silhouette by the
    # voxel footprint makes the test conservative: a voxel survives if any part
    # of it projects inside. That is what keeps the hull a real outer bound.
    span = int(np.ceil(h / px)) + 1
    if not args.center_test:
        print(f"conservative: voxel spans {h/px:.2f} px, "
              f"dilating silhouettes by {span} px")

    wanted = (args.use_views.split(",") if args.use_views
              else list(meta["views"]))
    occ = np.ones(tuple(dims), dtype=bool)
    t0 = time.time()

    for view in wanted:
        info = meta["views"][view]
        right, up, loc = view_basis(info["matrix_world"])
        mask = np.asarray(Image.open(os.path.join(root, "mask", f"{view}.png"))
                          .convert("L"), dtype=np.float32) / 255.0
        sil = mask > args.mask_threshold
        if not args.center_test:
            sil = ndimage.maximum_filter(sil, size=span, mode="constant")

        aligned = (axis_and_sign(right) is not None
                   and axis_and_sign(up) is not None)
        before = int(occ.sum())
        if aligned:
            carve_axis_aligned(occ, sil, centers, right, up, loc, ortho, res_px)
        else:
            carve_general(occ, sil, centers, right, up, loc, ortho, res_px)
        after = int(occ.sum())
        print(f"  carve {view:<13}{'axis' if aligned else 'diag':<5}"
              f"{before:>12,} -> {after:>12,} "
              f"({100.0*(before-after)/max(before,1):5.1f}% removed)")

    n_occ = int(occ.sum())
    vol = n_occ * h ** 3
    print(f"carved in   : {time.time()-t0:.1f}s")
    print(f"occupied    : {n_occ:,} voxels  volume {vol:.6f}")

    # Pad so the isosurface closes if the hull touches the grid boundary.
    padded = np.pad(occ.astype(np.float32), 1)
    if args.smooth > 0:
        padded = ndimage.gaussian_filter(padded, args.smooth, truncate=3.0)
        print(f"smoothed    : sigma {args.smooth} voxels, level {args.level}")
        # The surface no longer bounds the original voxels, so the volume the
        # mesh actually encloses is what containment must be measured against.
        occ = padded[1:-1, 1:-1, 1:-1] >= args.level
        n_occ = int(occ.sum())
        vol = n_occ * h ** 3
        print(f"re-occupied : {n_occ:,} voxels  volume {vol:.6f}")
    verts, faces, normals, _ = measure.marching_cubes(
        padded, level=args.level, spacing=(h, h, h))
    # grid index i (after the one-voxel pad) is voxel i-1, centred at
    # lo + (i - 0.5) h; subtracting a full h here put every mesh half a voxel
    # off along all three axes
    verts += lo - 0.5 * h

    print(f"mesh        : {len(verts):,} verts / {len(faces):,} faces")

    out_ply = args.out + ".ply"
    write_ply(out_ply, verts.astype(np.float32), faces.astype(np.int32))
    print(f"wrote       : {out_ply}")

    np.savez_compressed(args.out + "_occ.npz", occ=np.packbits(occ),
                        dims=dims, lo=lo, h=h)
    print(f"wrote       : {args.out}_occ.npz")

    stats = {
        "grid_dims": dims.tolist(),
        "voxel_size": h,
        "grid_lo": lo.tolist(),
        "occupied_voxels": n_occ,
        "hull_volume": vol,
        "mesh_vertices": int(len(verts)),
        "mesh_faces": int(len(faces)),
        "mask_threshold": args.mask_threshold,
        "smooth_sigma": args.smooth,
        "iso_level": args.level,
        "conservative": not args.center_test,
        "dilation_px": None if args.center_test else span,
        "views": wanted,
        "source_views": root,
    }
    with open(args.out + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def write_ply(path, verts, faces):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    rec = np.empty(len(faces), dtype=np.dtype([("n", "u1"), ("v", "<i4", 3)]))
    rec["n"] = 3
    rec["v"] = faces
    with open(path, "wb") as f:
        f.write(header)
        verts.astype("<f4").tofile(f)
        rec.tofile(f)


if __name__ == "__main__":
    main()
