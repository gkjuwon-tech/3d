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

How D is formed matters more than anything else here. A weighted *mean*
(--fusion mean, the original) loses to one wrong view: at a true surface
point every view that got it right says s = 0, which pulls on the mean not
at all, so a single view whose depth landed too deep -- saying "empty" at
full weight -- moves the surface inward on its own. That, measured on Lucy,
was what punched the holes: 72% of hole points were emptied by one view
that saw the point and put it too deep. --fusion robust (the default) takes
the weighted median of the views' truncated distances and then averages
only the views within one voxel of it, so a minority cannot carve through
what the others agree on, and the agreeing views still average their noise
away. Holes on Lucy: 6.71% -> 4.75% of the surface (face 5.75 -> 1.38,
hand 48.0 -> 19.8), from the same depth maps.

It also counts, per voxel, how many *anchored* views are among those that
agree (--support-out): the input to the second, cross-view depth pass in
consensus.py.

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
from xp import GPU, cpu, ndi, xp  # noqa: E402

CHUNK = 40_000_000 if GPU else 20_000_000

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


def geodesic_anchor_dist(d, anchordist_path, h, jump_vox, limit):
    """Pixels from the nearest live anchor, walking only between neighbours
    whose depths differ by less than jump_vox voxels."""
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra
    hit = np.isfinite(d)
    live = np.nan_to_num(np.load(anchordist_path), nan=1e9) == 0
    H_, W_ = d.shape
    idx = -np.ones(d.shape, dtype=np.int64)
    idx[hit] = np.arange(int(hit.sum()))
    rows, cols = [], []
    for sa, sb in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                   ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        a, b = idx[sa], idx[sb]
        ok = (a >= 0) & (b >= 0)
        ok &= np.abs(np.nan_to_num(d[sa]) - np.nan_to_num(d[sb])) < jump_vox * h
        rows.append(a[ok])
        cols.append(b[ok])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    n = int(hit.sum())
    G = sparse.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    src = idx[live & hit]
    out = np.full(d.shape, np.inf)
    if len(src):
        dist = dijkstra(G, directed=False, indices=src, min_only=True,
                        limit=limit, unweighted=True)
        out[hit] = dist
    return out


def drop_specks(d, h, min_px, jump_vox):
    """NaN out pieces of a depth map smaller than min_px, pixels joined where
    neighbours differ by less than jump_vox voxels.

    Where the depth solve cut its edges along a crease it leaves thousands of
    one- and two-pixel islands per view (Lucy: ~120k pixels over 14 views),
    each floating to whatever depth its last surviving edge suggested. Fused,
    every island is a needle along its ray -- a spike where it is too shallow,
    a crack where too deep; the shards and pits under Lucy's right ear. With
    no depth, the island's pixels abstain and the views around fill in:
    F@1 there 86.3 -> 89.5."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    hit = np.isfinite(d)
    idx = -np.ones(d.shape, np.int64)
    idx[hit] = np.arange(int(hit.sum()))
    dd = np.nan_to_num(d)
    r, c = [], []
    for sa, sb in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                   ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        ia, ib = idx[sa], idx[sb]
        ok = (ia >= 0) & (ib >= 0) & (np.abs(dd[sa] - dd[sb]) < jump_vox * h)
        r.append(ia[ok])
        c.append(ib[ok])
    r = np.concatenate(r)
    c = np.concatenate(c)
    n = int(hit.sum())
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)),
                                             shape=(n, n)), directed=False)
    small = np.bincount(lab)[lab] < min_px
    out = d.copy()
    hr, hc = np.nonzero(hit)
    out[hr[small], hc[small]] = np.nan
    return out


def robust_fuse(X, per_view, meta, res, h, tau, inlier):
    """Weighted median of the views' truncated distances, then the weighted
    mean of the views within `inlier` voxels of it. Returns D (voxels, -tau
    where no view has an opinion) and, per voxel, the number of agreeing
    views that are anchored at the pixel they see it through.

    Voxels are processed in chunks with every view resident at once, so the
    median sees all fourteen samples of a voxel together.
    """
    V = len(per_view)
    o = meta["ortho_scale"]
    D = np.empty(len(X), dtype=np.float32)
    sup = np.empty(len(X), dtype=np.uint8)
    chunk = max(CHUNK // (4 * V), 100_000)
    for j0 in range(0, len(X), chunk):
        j1 = min(j0 + chunk, len(X))
        Xc = xp.asarray(X[j0:j1])
        n = j1 - j0
        S = xp.empty((V, n), dtype=xp.float32)
        W = xp.empty((V, n), dtype=xp.float32)      # median weight
        Wd = xp.empty((V, n), dtype=xp.float32)     # mean weight (TSDF ramp)
        A = xp.empty((V, n), dtype=xp.uint8)
        for k, (dg, cg_, anc, right, up_, back, loc) in enumerate(per_view):
            Xg = Xc - loc
            col = (Xg @ right / o + 0.5) * res - 0.5
            row = (0.5 - Xg @ up_ / o) * res - 0.5
            w = -(Xg @ back)
            rc = xp.stack([row, col])
            s = (ndi.map_coordinates(dg, rc, order=1, mode="nearest") - w) / h
            c = ndi.map_coordinates(cg_, rc, order=0, mode="nearest")
            A[k] = ndi.map_coordinates(anc, rc, order=0, mode="nearest")
            ok = (s > -tau) & (c > 0)
            S[k] = xp.clip(s, -tau, tau)
            W[k] = xp.where(ok, c, 0)
            Wd[k] = xp.where(ok, c * xp.where(s < 0, 1 + s / tau, 1.0), 0)
        order = xp.argsort(xp.where(W > 0, S, xp.inf), axis=0)
        Ss = xp.take_along_axis(S, order, 0)
        cw = xp.cumsum(xp.take_along_axis(W, order, 0), 0)
        tot = cw[-1]
        km = xp.clip((cw < 0.5 * tot).sum(0), 0, V - 1)
        med = xp.take_along_axis(Ss, km[None], 0)[0]
        inl = (xp.abs(S - med) <= inlier) & (Wd > 0)
        d2 = (Wd * inl).sum(0)
        Dc = xp.where(d2 > 1e-6, (Wd * inl * S).sum(0) / xp.maximum(d2, 1e-12),
                      med)
        D[j0:j1] = cpu(xp.where(tot > 1e-6, Dc, -tau))
        sup[j0:j1] = cpu((inl & (A > 0)).sum(0).astype(xp.uint8))
    return D, sup


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
    ap.add_argument("--conf-mode", default="anchor", choices=["anchor", "geodesic"],
                    help="geodesic: distance to an anchor measured without "
                         "crossing a depth jump, so a finger does not borrow "
                         "the confidence of the torso behind it")
    ap.add_argument("--jump-vox", type=float, default=4.0,
                    help="a step larger than this between neighbouring pixels "
                         "is a depth jump for the geodesic distance")
    ap.add_argument("--quorum", type=int, default=1,
                    help="views that must call a voxel empty before it may be "
                         "carved against what the other views say")
    ap.add_argument("--empty-vox", type=float, default=1.0,
                    help="how far in front of a view's surface counts as that "
                         "view calling the voxel empty")
    ap.add_argument("--contra-views", type=int, default=0,
                    help="drop depth pixels contradicted by at least this many "
                         "other views (tools/depth_filter.py --write)")
    ap.add_argument("--hull-cache", default=None,
                    help="npy to load the hull field from, or save it to")
    ap.add_argument("--sil-up", type=int, default=4)
    ap.add_argument("--sil-level", type=float, default=0.5,
                    help="coverage level taken as the silhouette outline")
    ap.add_argument("--hull-offset", type=float, default=0.0,
                    help="voxels added outward to H")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="gaussian sigma in voxels on the final field")
    ap.add_argument("--fusion", default="robust", choices=["robust", "mean"],
                    help="robust: weighted median, then the mean of the views "
                         "within --inlier voxels of it; mean: the original")
    ap.add_argument("--inlier", type=float, default=1.0,
                    help="voxels; views this close to the median are averaged")
    ap.add_argument("--support-out", default=None,
                    help="write per-voxel count of agreeing anchored views "
                         "(uint8) here, for consensus.py")
    ap.add_argument("--support-px", type=float, default=2.0,
                    help="a view counts as anchored at a pixel this close to "
                         "one of its live anchors")
    ap.add_argument("--edge-len", type=float, default=4.0,
                    help="pixels; a view's confidence falls as 1-exp(-d/this) "
                         "with distance d to its own depth discontinuities "
                         "(jumps over --edge-vox). 0 disables")
    ap.add_argument("--edge-vox", type=float, default=4.0)
    ap.add_argument("--edge-floor", type=float, default=0.05,
                    help="the edge factor never falls below this. At the jump "
                         "pixel itself it was exactly zero, so in a crevice -- "
                         "all jumps -- every view fell silent at once, the "
                         "voxels there defaulted to solid, and the boundary of "
                         "that silence meshed as crumbs: under Lucy's right ear "
                         "F@1 77.7 -> 86.3, loose pieces 35 -> 18")
    ap.add_argument("--speckle-px", type=int, default=30,
                    help="depth pieces smaller than this abstain (drop_specks); "
                         "0 keeps them")
    ap.add_argument("--clamp-dir", default=None,
                    help="consensus depths (consensus.py): a view may not claim "
                         "to see further than this verified surface along its "
                         "ray plus --clamp-margin voxels")
    ap.add_argument("--clamp-margin", type=float, default=2.0)
    ap.add_argument("--no-mesh", action="store_true",
                    help="write the field (and support) only")
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
        if args.quorum > 1:
            num_n = np.zeros(len(X), dtype=np.float32)   # surface/inside only
            den_n = np.zeros(len(X), dtype=np.float32)
            n_pos = np.zeros(len(X), dtype=np.int16)     # views calling it empty
        tau = args.trunc
        per_view = []
        for v in views:
            d = np.load(os.path.join(args.depth_dir, f"{v}.npy"))
            if args.speckle_px > 0:
                d = drop_specks(d.astype(np.float64), h, args.speckle_px,
                                args.edge_vox)
            if args.clamp_dir:
                # Inside a solid, the views that got its skin right are past
                # their truncation band and abstain; a view that put its
                # depth too deep is then the only voice there, and the
                # median of one hollows the interior out behind an intact
                # skin. Its line of sight is blocked at the verified surface.
                cl = np.load(os.path.join(args.clamp_dir, f"{v}.npy"))
                d = np.where(np.isfinite(cl), np.minimum(d, cl + args.clamp_margin * h), d)
            hitv = np.isfinite(d)
            idx = ndimage.distance_transform_edt(~hitv, return_distances=False,
                                                 return_indices=True)
            dfill = d[idx[0], idx[1]]
            conf = np.ones_like(d)
            dist_p = os.path.join(args.depth_dir, f"{v}_anchordist.npy")
            if args.conf_mode == "geodesic":
                dist = geodesic_anchor_dist(d, os.path.join(args.depth_dir,
                                                            f"{v}_anchordist.npy"),
                                            h, args.jump_vox, args.conf_len * 6)
                conf = np.maximum(np.exp(-dist / args.conf_len), args.conf_floor)
            elif os.path.exists(dist_p):
                dist = np.nan_to_num(np.load(dist_p), nan=1e6)
                conf = np.maximum(np.exp(-dist / args.conf_len), args.conf_floor)
            if args.contra_views > 0:
                cp = os.path.join(args.depth_dir, f"{v}_contra.npy")
                conf = np.where(np.load(cp) >= args.contra_views, 0.0, conf)
            if args.normals_dir:
                nzv = np.abs(np.load(os.path.join(args.normals_dir, f"{v}.npy"))
                             [..., 2]).astype(np.float32)
                conf = conf * nzv
            if args.edge_len > 0:
                # depth is least reliable next to its own jumps: measured on
                # Lucy, 44% of pixels within 2 px of one are right to 1.5
                # voxels, 98% of those 20 px away
                dd = np.nan_to_num(d.astype(np.float64), nan=1e3)
                jump = np.zeros(d.shape, bool)
                jr = np.abs(np.diff(dd, axis=0)) > args.edge_vox * h
                jc = np.abs(np.diff(dd, axis=1)) > args.edge_vox * h
                jump[1:] |= jr
                jump[:-1] |= jr
                jump[:, 1:] |= jc
                jump[:, :-1] |= jc
                jd = ndimage.distance_transform_edt(~jump)
                conf = conf * np.maximum(1.0 - np.exp(-jd / args.edge_len),
                                         args.edge_floor)
            conf = np.where(hitv, conf, 0).astype(np.float32)
            # the per-voxel sampling below is the whole cost of fusion, and
            # runs on the xp backend: the GPU when THREED_GPU=1
            dg = xp.asarray(dfill.astype(np.float32))
            cg_ = xp.asarray(conf)
            right, up_, back, loc = (xp.asarray(q, dtype=xp.float32)
                                     for q in cam(meta, v))
            o = meta["ortho_scale"]
            if args.fusion == "robust":
                ad = np.load(dist_p) if os.path.exists(dist_p) else \
                    np.zeros_like(d)
                anc = xp.asarray((np.nan_to_num(ad, nan=1e9)
                                  <= args.support_px).astype(np.uint8))
                per_view.append((dg, cg_, anc, right, up_, back, loc))
                continue
            for j0 in range(0, len(X), CHUNK):
                j1 = min(j0 + CHUNK, len(X))
                Xg = xp.asarray(X[j0:j1]) - loc
                col = (Xg @ right / o + 0.5) * res - 0.5
                row = (0.5 - Xg @ up_ / o) * res - 0.5
                w = -(Xg @ back)
                rc = xp.stack([row, col])
                s = (ndi.map_coordinates(dg, rc, order=1, mode="nearest") - w) / h
                c = ndi.map_coordinates(cg_, rc, order=0, mode="nearest")
                ok = (s > -tau) & (c > 0)
                wt = xp.where(ok, c * xp.where(s < 0, 1 + s / tau, 1.0), 0)
                num[j0:j1] += cpu(wt * xp.clip(s, -tau, tau))
                den[j0:j1] += cpu(wt)
                if args.quorum > 1:
                    neg = wt * (s <= args.empty_vox)
                    num_n[j0:j1] += cpu(neg * xp.clip(s, -tau, tau))
                    den_n[j0:j1] += cpu(neg)
                    n_pos[j0:j1] += cpu((wt > 0) & (s > args.empty_vox)).astype(np.int16)
            del dg, cg_
            print(f"  {v}: {time.time()-t0:.0f}s", flush=True)
        if args.fusion == "robust":
            D, sup = robust_fuse(X, per_view, meta, res, h, tau, args.inlier)
            del per_view
            print(f"robust fusion: {time.time()-t0:.0f}s", flush=True)
            if args.support_out:
                S = np.zeros(dims, dtype=np.uint8)
                S[ii] = sup
                np.save(args.support_out, S)
                del S
            del sup
        else:
            seen = den > 1e-6
            D = np.where(seen, num / np.maximum(den, 1e-12), -tau).astype(np.float32)
        if args.quorum > 1:
            # a voxel only one view calls empty keeps what the others say about
            # it: one view with a wrong depth cannot carve on its own
            lone = (n_pos < args.quorum) & (den_n > 1e-6)
            D = np.where(lone, np.minimum(D, num_n / np.maximum(den_n, 1e-12)), D)
            print(f"quorum {args.quorum}: {int(lone.sum()):,} voxels held by "
                  f"their other views", flush=True)
        F[ii] = np.maximum(H[ii], D)
        del X, num, den, D
    if args.smooth > 0:
        F = ndimage.gaussian_filter(F, args.smooth, truncate=3.0)

    if args.no_mesh:
        np.save(args.out + "_field.npy", F.astype(np.float16))
        print(f"wrote {args.out}_field.npy  {time.time()-t0:.0f}s")
        return
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
