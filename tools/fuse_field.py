#!/usr/bin/env python3
"""Fuse the hull and the per-view depth maps into one continuous field.

Every surface so far came out of marching cubes run over a binary grid, and
that is where the terracing came from: a voxel is in or out, so a gently
sloping plane becomes a staircase of one-voxel treads, and a Gaussian blur of
width one voxel softens the treads without removing them. The fix is not more
blur. It is to never make the surface binary in the first place.

Two continuous fields, combined by a max (an intersection, for signed
distances that are positive outside):

  H   the visual hull, exactly. For an orthographic camera the distance from a
      point to the silhouette's extruded cone is the 2D distance from the
      point's projection to the silhouette outline, so H is the max over
      views of a bilinearly sampled 2D signed distance map. Its zero set is
      made of ruled surfaces, not voxel faces.
  D   the depth maps, as a truncated signed distance (Curless & Levoy 1996).
      For each view, s = depth - distance along the ray: positive in front of
      the observed surface (empty), negative just behind it, unknown beyond the
      truncation band. Samples are averaged with weights for how well anchored
      that pixel's depth is and how squarely the view faces the surface.

F = max(H, D) where some view has an opinion, H elsewhere; marching cubes at
level zero then places vertices between voxel centres by linear interpolation
of a field that really is linear there.

Run:
  python3 tools/fuse_field.py --occ out/final_mesh_occ.npz --views refs/lucy_gt \
      --depth-dir out/depth_mv --normals-dir out/ps_normals --out out/fused
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visual_hull import write_ply  # noqa: E402

BG = 1e9


def cam(meta, view):
    m = np.array(meta["views"][view]["matrix_world"], dtype=np.float64)
    return m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3]


def project(meta, view, X, res):
    """Continuous (row, col) with integers at pixel centres, and ray depth."""
    right, up, back, loc = cam(meta, view)
    o = meta["ortho_scale"]
    rel = X - loc.astype(np.float32)
    col = (rel @ right.astype(np.float32) / o + 0.5) * res - 0.5
    row = (0.5 - rel @ up.astype(np.float32) / o) * res - 0.5
    w = -(rel @ back.astype(np.float32))
    return row, col, w


def silhouette_sdf(mask, px, up=4, level=0.5):
    """Signed distance to the silhouette outline in world units, >0 outside,
    sampled on a grid `up` times finer than the image.

    The outline is the `level` contour of the anti-aliased coverage, found on
    the finer grid. Thresholding at pixel resolution instead leaves the outline
    a staircase of whole pixels, and a cone extruded from a staircase is a
    stack of ridges: every near-vertical silhouette edge becomes horizontal
    terraces across the body, and the diagonal views' terraces cross into the
    concentric squares that were on the chest. That was never a voxel
    artefact; the continuous field reproduced it exactly until this changed.
    """
    fine = ndimage.zoom(mask, up, order=1, grid_mode=True, mode="nearest")
    inside = fine >= level
    d_out = ndimage.distance_transform_edt(~inside)
    d_in = ndimage.distance_transform_edt(inside)
    return ((d_out - d_in) * (px / up)).astype(np.float32)


def sample(img, row, col, order=1):
    return ndimage.map_coordinates(img, [row, col], order=order,
                                   mode="nearest", prefilter=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ", required=True, help="grid definition (and a hull "
                    "occupancy used only to bound where depth is fused)")
    ap.add_argument("--views", required=True)
    ap.add_argument("--depth-dir", default=None)
    ap.add_argument("--normals-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trunc", type=float, default=3.0, help="voxels")
    ap.add_argument("--conf-len", type=float, default=40.0,
                    help="pixels from the nearest anchor at which a depth "
                         "sample's weight has fallen to 1/e")
    ap.add_argument("--conf-floor", type=float, default=0.02)
    ap.add_argument("--hull-cache", default=None,
                    help="npy to load the hull field from, or save it to")
    ap.add_argument("--sil-up", type=int, default=4)
    ap.add_argument("--sil-level", type=float, default=0.5,
                    help="coverage level taken as the silhouette outline")
    ap.add_argument("--hull-offset", type=float, default=0.0,
                    help="voxels added outward to H")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="gaussian sigma in voxels on the final field")
    ap.add_argument("--band", type=float, default=2.0,
                    help="voxels outside the hull still fused (anti-alias)")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    z = np.load(args.occ)
    dims = tuple(int(d) for d in z["dims"])
    lo, h = z["lo"].astype(np.float64), float(z["h"])
    res = meta["resolution"][0]
    px = meta["ortho_scale"] / res
    views = list(meta["views"])
    t0 = time.time()

    # --- H: exact hull distance, max over views --------------------------------
    cx = (lo[0] + (np.arange(dims[0]) + 0.5) * h).astype(np.float32)
    cy = (lo[1] + (np.arange(dims[1]) + 0.5) * h).astype(np.float32)
    cz = (lo[2] + (np.arange(dims[2]) + 0.5) * h).astype(np.float32)
    if args.hull_cache and os.path.exists(args.hull_cache):
        H = np.load(args.hull_cache).astype(np.float32)
    else:
        H = np.full(dims, -np.inf, dtype=np.float32)
        up = args.sil_up
        slab = 16
        for v in views:
            mask = np.asarray(Image.open(os.path.join(args.views, "mask",
                                                      f"{v}.png"))
                              .convert("L"), dtype=np.float32) / 255.0
            sd = silhouette_sdf(mask, px, up, args.sil_level)
            for i0 in range(0, dims[0], slab):
                i1 = min(i0 + slab, dims[0])
                X = np.stack(np.meshgrid(cx[i0:i1], cy, cz, indexing="ij"),
                             -1).reshape(-1, 3)
                row, col, _ = project(meta, v, X, res)
                # coarse pixel coordinate c sits at (c + 0.5) up - 0.5 finely
                H[i0:i1] = np.maximum(
                    H[i0:i1], sample(sd, (row + 0.5) * up - 0.5,
                                     (col + 0.5) * up - 0.5)
                    .reshape(i1 - i0, dims[1], dims[2]))
            del sd
        H /= h                               # voxel units
        if args.hull_cache:
            np.save(args.hull_cache, H.astype(np.float16))
    H -= args.hull_offset
    print(f"hull field: {time.time()-t0:.0f}s, inside {int((H < 0).sum()):,} "
          f"voxels", flush=True)

    F = H.copy()
    if args.depth_dir:
        # --- D: truncated signed distance from the depth maps ------------------
        live = H < args.band
        ii = np.nonzero(live)
        X = np.stack([cx[ii[0]], cy[ii[1]], cz[ii[2]]], -1)
        print(f"fusing depth over {len(X):,} voxels", flush=True)
        num = np.zeros(len(X), dtype=np.float32)
        den = np.zeros(len(X), dtype=np.float32)
        tau = args.trunc
        for v in views:
            d = np.load(os.path.join(args.depth_dir, f"{v}.npy"))
            hitv = np.isfinite(d)
            idx = ndimage.distance_transform_edt(~hitv, return_distances=False,
                                                 return_indices=True)
            dfill = d[idx[0], idx[1]]
            conf = np.ones_like(d)
            dist_p = os.path.join(args.depth_dir, f"{v}_anchordist.npy")
            if os.path.exists(dist_p):
                dist = np.nan_to_num(np.load(dist_p), nan=1e6)
                conf = np.maximum(np.exp(-dist / args.conf_len), args.conf_floor)
            if args.normals_dir:
                nzv = np.abs(np.load(os.path.join(args.normals_dir, f"{v}.npy"))
                             [..., 2]).astype(np.float32)
                conf = conf * nzv
            conf = np.where(hitv, conf, 0).astype(np.float32)
            for j0 in range(0, len(X), 20_000_000):
                j1 = min(j0 + 20_000_000, len(X))
                row, col, w = project(meta, v, X[j0:j1], res)
                s = (sample(dfill, row, col) - w) / h      # voxels, >0 empty
                c = sample(conf, row, col, order=0)
                ok = (s > -tau) & (c > 0)
                wt = np.where(ok, c * np.where(s < 0, 1 + s / tau, 1.0), 0)
                num[j0:j1] += wt * np.clip(s, -tau, tau)
                den[j0:j1] += wt
            print(f"  {v}: {time.time()-t0:.0f}s", flush=True)
        seen = den > 1e-6
        D = np.where(seen, num / np.maximum(den, 1e-12), -tau).astype(np.float32)
        F[ii] = np.maximum(H[ii], D)
        del X, num, den, D
    if args.smooth > 0:
        F = ndimage.gaussian_filter(F, args.smooth, truncate=3.0)

    occ = F < 0
    np.savez_compressed(args.out + "_occ.npz", occ=np.packbits(occ),
                        dims=np.array(dims), lo=lo, h=h)
    np.save(args.out + "_field.npy", F.astype(np.float16))
    from skimage import measure
    Fp = np.pad(F, 1, constant_values=float(tau if args.depth_dir else 10))
    verts, faces, _, _ = measure.marching_cubes(Fp, level=0.0,
                                                spacing=(h, h, h))
    verts += lo + 0.5 * h - h      # voxel centres, undo the pad
    faces = faces[:, ::-1]         # outward-facing for a field positive outside
    write_ply(args.out + ".ply", verts.astype(np.float32),
              faces.astype(np.int32))
    print(f"wrote {args.out}.ply  {len(faces):,} faces  inside "
          f"{int(occ.sum()):,} voxels  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
