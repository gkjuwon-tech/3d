#!/usr/bin/env python3
"""The visual hull as a distance field, from silhouettes and nothing else.

Writes, into --out:
  grid.npz       voxel grid (dims, lo, h), sized from the silhouettes
  H.npy          hull distance field, voxels, >0 outside (float16)
  hull.ply       its zero level set
  views/         hull depth per view (depth_npy/<view>.npy) plus cameras.json,
                 the floor every depth recovery rests on

Three things make this cheaper and more honest than the voxel carver it
replaces.

The grid comes from the silhouettes. Their extents in the axis-aligned views
bound the object on every world axis, so no bounding box has to be read from
the renderer's metadata -- that would be a quiet leak of ground truth into a
pipeline that is supposed to see images only.

H is computed exactly only where it matters. For an orthographic camera the
distance to a silhouette's cone is the 2D distance of the projection to the
outline, and the max over views of 1-Lipschitz functions is 1-Lipschitz. So
H is first evaluated on every fourth voxel; a voxel whose coarse neighbour
reads |H| >= band + 3.5 is then guaranteed to have the same sign and to be
at least `band` from the surface, and only the rest are evaluated exactly.

Hull depth is found by sphere tracing H: along each pixel's ray, step by the
distance H guarantees is empty, until it is under a twentieth of a voxel.

Run:
  python3 tools/hull_field.py --views refs/lucy_gt --out out/hull
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuse_field import cam, project, sample, silhouette_sdf  # noqa: E402
from visual_hull import axis_and_sign, write_ply  # noqa: E402

BG = 1e10


def load_mask(views, v):
    return np.asarray(Image.open(os.path.join(views, "mask", f"{v}.png"))
                      .convert("L"), dtype=np.float32) / 255.0


def grid_from_silhouettes(views, meta, res_vox, pad):
    """World-axis bounds from the axis-aligned views' silhouette extents."""
    o = meta["ortho_scale"]
    lo = np.full(3, -np.inf)
    hi = np.full(3, np.inf)
    for v in meta["views"]:
        right, up, back, loc = cam(meta, v)
        ar, au = axis_and_sign(right), axis_and_sign(up)
        if ar is None or au is None:
            continue
        m = load_mask(views, v) > 0
        rows, cols = np.nonzero(m)
        res = m.shape[0]
        for (axis, sign), a0, a1, flip in ((ar, cols.min(), cols.max() + 1, False),
                                           (au, rows.min(), rows.max() + 1, True)):
            if flip:   # rows run opposite to up
                c0, c1 = (0.5 - a1 / res) * o, (0.5 - a0 / res) * o
            else:
                c0, c1 = (a0 / res - 0.5) * o, (a1 / res - 0.5) * o
            base = loc[axis]
            w0, w1 = sorted((base + sign * c0, base + sign * c1))
            lo[axis] = max(lo[axis], w0)
            hi[axis] = min(hi[axis], w1)
    size = hi - lo
    h = size.max() / res_vox
    lo = lo - size.max() * pad
    hi = hi + size.max() * pad
    dims = np.maximum(np.ceil((hi - lo) / h).astype(int), 2)
    return dims, lo, h


def sdf_cached(views, meta, v, up, level, cache):
    """The fine silhouette distance map, computed once per view.

    It is needed twice -- for the coarse pass and for the band -- and the
    distance transform at four times the image resolution is most of the cost
    of the whole hull, so the first pass leaves it on disk.
    """
    path = os.path.join(cache, f"{v}_sdf.npy") if cache else None
    if path and os.path.exists(path):
        return np.load(path).astype(np.float32)
    px = meta["ortho_scale"] / meta["resolution"][0]
    sd = silhouette_sdf(load_mask(views, v), px, up, level)
    if path:
        np.save(path, sd.astype(np.float16))
    return sd


def eval_H(views, meta, pts, up, level, cache=None):
    """Exact hull distance (world units) at an (N, 3) float32 array."""
    res = meta["resolution"][0]
    H = np.full(len(pts), -np.inf, dtype=np.float32)
    for v in meta["views"]:
        sd = sdf_cached(views, meta, v, up, level, cache)
        for j0 in range(0, len(pts), 8_000_000):
            j1 = min(j0 + 8_000_000, len(pts))
            row, col, _ = project(meta, v, pts[j0:j1], res)
            H[j0:j1] = np.maximum(H[j0:j1], sample(sd, (row + 0.5) * up - 0.5,
                                                    (col + 0.5) * up - 0.5))
        del sd
    return H


def centres(dims, lo, h, stride=1):
    return [(lo[a] + (np.arange(0, dims[a], stride) + 0.5) * h).astype(np.float32)
            for a in range(3)]


def hull_depth(meta, view, Hf, lo, h, dims, mask, max_iter=400):
    """Sphere-trace the hull field along every silhouette pixel's ray."""
    res = mask.shape[0]
    o = meta["ortho_scale"]
    right, up, back, loc = cam(meta, view)
    r, c = np.nonzero(mask > 0)
    u = ((c + 0.5) / res - 0.5) * o
    v = (0.5 - (r + 0.5) / res) * o
    P0 = loc + u[:, None] * right + v[:, None] * up
    d = -back
    # enter the grid's box along the ray
    box_lo, box_hi = lo, lo + dims * h
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (box_lo - P0) / d
        t1 = (box_hi - P0) / d
    tmin = np.nanmax(np.minimum(t0, t1), axis=1)
    tmax = np.nanmin(np.maximum(t0, t1), axis=1)
    t = np.maximum(tmin, 0.0)
    depth = np.full(len(r), np.nan)
    live = np.arange(len(r))
    for _ in range(max_iter):
        X = P0[live] + t[live, None] * d
        idx = ((X - lo) / h - 0.5).T
        f = ndimage.map_coordinates(Hf, idx, order=1, mode="nearest")
        done = f < 0.05
        depth[live[done]] = t[live[done]] + np.maximum(f[done], 0) * h
        step = np.maximum(f, 0.25) * h
        t[live] += np.where(done, 0, step)
        live = live[~done & (t[live] < tmax[live])]
        if len(live) == 0:
            break
    out = np.full(mask.shape, BG, dtype=np.float32)
    ok = np.isfinite(depth)
    out[r[ok], c[ok]] = depth[ok]
    return out, len(live)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024, help="voxels on the longest axis")
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--sil-up", type=int, default=4)
    ap.add_argument("--sil-level", type=float, default=0.5)
    ap.add_argument("--offset", type=float, default=0.4,
                    help="voxels the hull is moved outward; covers the 0.35 "
                         "voxel worst case measured at contact points")
    ap.add_argument("--band", type=float, default=6.0)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    t00 = time.time()
    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    os.makedirs(args.out, exist_ok=True)
    dims, lo, h = grid_from_silhouettes(args.views, meta, args.res, args.pad)
    dims_t = tuple(int(x) for x in dims)
    print(f"grid {dims_t} = {np.prod(dims)/1e6:.0f}M voxels, h {h:.6f}", flush=True)

    # coarse pass
    s = args.stride
    cx, cy, cz = centres(dims, lo, h, s)
    P = np.stack(np.meshgrid(cx, cy, cz, indexing="ij"), -1).reshape(-1, 3)
    cache = os.path.join(args.out, "sdf_cache")
    os.makedirs(cache, exist_ok=True)
    Hc = (eval_H(args.views, meta, P, args.sil_up, args.sil_level, cache) / h
          ).reshape(len(cx), len(cy), len(cz))
    del P
    print(f"coarse {Hc.shape}: {time.time()-t00:.0f}s", flush=True)
    # nearest coarse sample of every fine voxel, then the band
    ix = [np.minimum(np.arange(dims[a]) // s, Hc.shape[a] - 1) for a in range(3)]
    H = Hc[np.ix_(ix[0], ix[1], ix[2])].astype(np.float32)
    lip = 0.5 * s * np.sqrt(3) + 0.1
    band = np.abs(H) < args.band + lip
    ii = np.nonzero(band)
    print(f"band: {len(ii[0]):,} voxels ({100*len(ii[0])/H.size:.1f}%)", flush=True)
    fx, fy, fz = centres(dims, lo, h)
    P = np.stack([fx[ii[0]], fy[ii[1]], fz[ii[2]]], -1)
    H[ii] = eval_H(args.views, meta, P, args.sil_up, args.sil_level, cache) / h
    shutil.rmtree(cache)
    del P, band, ii
    H -= args.offset
    print(f"fine: {time.time()-t00:.0f}s  inside {int((H < 0).sum()):,}", flush=True)

    np.save(os.path.join(args.out, "H.npy"), H.astype(np.float16))
    np.savez_compressed(os.path.join(args.out, "grid.npz"),
                        occ=np.packbits(H < 0), dims=np.array(dims_t), lo=lo, h=h)

    from skimage import measure
    verts, faces, _, _ = measure.marching_cubes(
        np.pad(H, 1, constant_values=10.0), level=0.0, spacing=(h, h, h))
    verts += lo - 0.5 * h
    write_ply(os.path.join(args.out, "hull.ply"), verts.astype(np.float32),
              faces[:, ::-1].astype(np.int32))
    print(f"hull.ply {len(faces):,} faces: {time.time()-t00:.0f}s", flush=True)

    vdir = os.path.join(args.out, "views")
    os.makedirs(os.path.join(vdir, "depth_npy"), exist_ok=True)
    shutil.copy(os.path.join(args.views, "cameras.json"), vdir)
    for v in meta["views"]:
        d, missed = hull_depth(meta, v, H, lo, h, dims, load_mask(args.views, v))
        np.save(os.path.join(vdir, "depth_npy", f"{v}.npy"), d)
        note = ""
        gp = os.path.join(args.views, "depth_npy", f"{v}.npy")
        if os.path.exists(gp):
            gt = np.load(gp)
            m = (gt < 1e9) & (d < 1e9)
            e = (d[m] - gt[m]) * 1024
            note = (f"  vs GT: in front {100*(e < 0).mean():.2f}%, "
                    f"behind max {e.max():.2f} vox, mean gap {-e.mean():.2f} vox")
        print(f"  {v:<13} unconverged rays {missed}{note}", flush=True)
    print(f"done {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
