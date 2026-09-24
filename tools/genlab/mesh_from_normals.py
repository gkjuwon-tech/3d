#!/usr/bin/env python3
"""One mesh from several generated normal maps: integrate on the surface.

A single generated view's normal map integrates into a convincing relief, but
two views disagree by tens of pixels in depth, so fusing per-view depth maps by
agreement (fuse_field) breaks the object apart. Here the unknown is one mesh,
and every surface point follows the normals of the views that see it most
squarely (weights cos^p over the views that see it: effectively one view per
point, blended across the seams). No voting, so nothing is carved away where
views disagree -- the surface simply takes one view's word there.

Each outer iteration: rotate every face toward its target normal, then solve
for the vertex positions whose edges best match the rotated edges (a sparse
Poisson solve over the whole mesh, so a relief propagates in one step), held
lightly to the previous positions and firmly on the silhouette contours.

    python3 tools/genlab/mesh_from_normals.py --mesh data/cat/proxy8v2/proxy.ply \
        --views data/catgen/views --normals data/catgen/normals --out data/cat/mfn/mesh.ply
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from eval_hull import read_ply  # noqa: E402
from displace import write_ply, subdivide  # noqa: E402


class Cam:
    def __init__(self, meta, v, normals, views):
        M = np.array(meta["views"][v]["matrix_world"])
        self.right, self.up, self.back, self.loc, self.R = M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3], M[:3, :3]
        self.o = meta["ortho_scale"]
        n = np.load(f"{normals}/{v}.npy").astype(np.float64)
        self.res = n.shape[0]
        self.nw = n @ self.R.T                                  # camera -> world
        self.mask = np.asarray(Image.open(f"{views}/mask/{v}.png").convert("L")) > 127
        self.valid = self.mask & (np.abs(n).sum(-1) > 0)
        self.px = self.o / self.res

    def project(self, X):
        d = X - self.loc
        return ((-(d @ self.up) / self.o + 0.5) * self.res - 0.5,
                ((d @ self.right) / self.o + 0.5) * self.res - 0.5, d @ (-self.back))

    def visible(self, X, tol_px=3.0, splat=2):
        r, c, z = self.project(X)
        ri, ci = np.round(r).astype(int), np.round(c).astype(int)
        inb = (ri >= 0) & (ri < self.res) & (ci >= 0) & (ci < self.res)
        zb = np.full((self.res, self.res), np.inf)
        np.minimum.at(zb, (ri[inb], ci[inb]), z[inb])
        zb = ndimage.grey_erosion(zb, size=2 * splat + 1)          # close the gaps between vertices
        vis = np.zeros(len(X), bool)
        vis[inb] = z[inb] <= zb[ri[inb], ci[inb]] + tol_px * self.px
        return r, c, vis

    def sample(self, r, c):
        """masked bilinear sample of the world normals; (values, ok)"""
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        out = np.zeros((len(r), 3)); ws = np.zeros(len(r))
        for dr in (0, 1):
            for dc in (0, 1):
                rr, cc = (r0 + dr).clip(0, self.res - 1), (c0 + dc).clip(0, self.res - 1)
                w = (1 - np.abs(r - (r0 + dr))) * (1 - np.abs(c - (c0 + dc))) * self.valid[rr, cc]
                out += w[:, None] * self.nw[rr, cc]; ws += w
        return out / np.maximum(ws, 1e-9)[:, None], ws > 0.5


def face_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    return fn / np.linalg.norm(fn, axis=1, keepdims=True).clip(1e-12)


def targets(v, f, cams, power):
    """per-face target normal from the views that see the face most squarely"""
    cen = v[f].mean(1)
    fn = face_normals(v, f)
    acc = np.zeros((len(f), 3)); wsum = np.zeros(len(f)); best = np.zeros(len(f))
    for cam in cams:
        r, c, vis = cam.visible(cen)
        cos = fn @ cam.back
        val, ok = cam.sample(r, c)
        use = vis & ok & (cos > 0.05)
        w = np.where(use, np.clip(cos, 0, 1) ** power, 0.0)
        acc += w[:, None] * val; wsum += w
        best = np.maximum(best, np.where(use, cos, 0))
    t = acc / np.maximum(wsum, 1e-12)[:, None]
    t /= np.linalg.norm(t, axis=1, keepdims=True).clip(1e-12)
    has = wsum > 0
    t[~has] = fn[~has]
    return t, has


def rotate_edges(v, f, tgt):
    """edge vectors of each face after the minimal rotation taking its normal to tgt"""
    n = face_normals(v, f)
    k = np.cross(n, tgt); s = np.linalg.norm(k, axis=1); cth = (n * tgt).sum(1)
    k = k / np.maximum(s, 1e-12)[:, None]
    out = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        e = v[f[:, b]] - v[f[:, a]]
        # Rodrigues
        er = (e * cth[:, None] + np.cross(k, e) * s[:, None]
              + k * (k * e).sum(1, keepdims=True) * (1 - cth)[:, None])
        out.append(er)
    return out


def contour_vertices(v, f, cams, cos_max=0.2, band_px=2.0):
    """vertices on some view's silhouette: grazing there and next to the mask edge"""
    vn = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-12)
    on = np.zeros(len(v), bool)
    for cam in cams:
        edge = ndimage.distance_transform_edt(cam.mask) <= band_px
        edge &= cam.mask
        r, c, _ = cam.project(v)
        ri, ci = np.round(r).astype(int).clip(0, cam.res - 1), np.round(c).astype(int).clip(0, cam.res - 1)
        on |= (np.abs(vn @ cam.back) < cos_max) & edge[ri, ci]
    return on


def silhouette_pull(v, cams, tol_px=0.5):
    """3D offset per vertex that brings its projection inside every mask it
    falls outside of (averaged over those views), and whether it needed one:
    the object lies inside every silhouette, from every camera"""
    off = np.zeros_like(v); cnt = np.zeros(len(v))
    for cam in cams:
        if not hasattr(cam, "_near"):
            d, idx = ndimage.distance_transform_edt(~cam.mask, return_indices=True)
            cam._near, cam._dist = idx, d
        r, c, _ = cam.project(v)
        ri, ci = np.round(r).astype(int).clip(0, cam.res - 1), np.round(c).astype(int).clip(0, cam.res - 1)
        out = cam._dist[ri, ci] > tol_px
        tr, tc = cam._near[0][ri, ci], cam._near[1][ri, ci]
        dr, dc = (tr - r), (tc - c)
        o = (dc[:, None] * cam.right - dr[:, None] * cam.up) * cam.px
        off[out] += o[out]; cnt[out] += 1
    need = cnt > 0
    off[need] /= cnt[need][:, None]
    return off, need


def solve(v, f, edges_t, mu, target=None):
    nV = len(v)
    rows, cols, vals, rhs = [], [], [], []
    m = 0
    E = []
    for (a, b), et in zip(((0, 1), (1, 2), (2, 0)), edges_t):
        E.append((f[:, a], f[:, b], et))
    ia = np.concatenate([e[0] for e in E]); ib = np.concatenate([e[1] for e in E])
    et = np.concatenate([e[2] for e in E])
    m = len(ia); r = np.arange(m)
    A = sparse.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[r, r], np.r_[ib, ia])), shape=(m, nV))
    M = (A.T @ A + sparse.diags(mu)).tocsr()
    out = np.empty_like(v)
    target = v if target is None else target
    for k in range(3):
        b = A.T @ et[:, k] + mu * target[:, k]
        out[:, k], _ = cg(M, b, x0=v[:, k], rtol=1e-7, maxiter=2000)
    return out


def silhouette_iou(v, cams):
    res = []
    for cam in cams:
        r, c, _ = cam.project(v)
        ri, ci = np.round(r).astype(int).clip(0, cam.res - 1), np.round(c).astype(int).clip(0, cam.res - 1)
        s = np.zeros_like(cam.mask); s[ri, ci] = True
        s = ndimage.binary_closing(ndimage.binary_dilation(s, iterations=1), iterations=3)
        s = ndimage.binary_fill_holes(s)
        res.append((s & cam.mask).sum() / (s | cam.mask).sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--views", required=True, help="cameras.json + mask/")
    ap.add_argument("--normals", required=True, help="<view>.npy camera-space normals")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--power", type=float, default=8.0, help="view weight cos^p")
    ap.add_argument("--mu", type=float, default=0.02, help="hold toward the previous positions")
    ap.add_argument("--mu-contour", type=float, default=2.0, help="hold on silhouette contours")
    ap.add_argument("--mu-sil", type=float, default=2.0,
                    help="pull of a vertex outside some silhouette toward it")
    ap.add_argument("--subdivide-at", type=int, default=-1, help="subdivide once before this iteration")
    a = ap.parse_args()
    meta = json.load(open(f"{a.views}/cameras.json"))
    cams = [Cam(meta, v, a.normals, a.views) for v in meta["views"]
            if os.path.exists(f"{a.normals}/{v}.npy")]
    v, f = read_ply(a.mesh)
    v = v.astype(np.float64); f = f.astype(np.int64)
    vol = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6
    if vol < 0:
        f = f[:, ::-1].copy()      # faces must point outward
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"{len(v):,} verts {len(f):,} faces, {len(cams)} views; silhouette IoU "
          + " ".join(f"{x:.3f}" for x in silhouette_iou(v, cams)), flush=True)
    for it in range(a.iters):
        if it == a.subdivide_at:
            v, f = subdivide(v, f)
            print(f"  subdivided: {len(v):,} verts", flush=True)
        t, has = targets(v, f, cams, a.power)
        err0 = np.degrees(np.arccos(np.clip((face_normals(v, f) * t).sum(1), -1, 1)))[has]
        et = rotate_edges(v, f, t)
        mu = np.full(len(v), a.mu)
        mu[contour_vertices(v, f, cams)] = a.mu_contour
        off, need = silhouette_pull(v, cams)
        mu[need] = a.mu_sil
        v = solve(v, f, et, mu, v + off)
        err1 = np.degrees(np.arccos(np.clip((face_normals(v, f) * t).sum(1), -1, 1)))[has]
        print(f"  iter {it}: face vs target normal median {np.median(err0):5.1f} -> {np.median(err1):5.1f} deg"
              f"  (covered {100*has.mean():.0f}% of faces; {100*need.mean():.0f}% of vertices outside a silhouette)",
              flush=True)
        write_ply(a.out, v.astype(np.float32), f.astype(np.int32))
    print("silhouette IoU " + " ".join(f"{x:.3f}" for x in silhouette_iou(v, cams)))


if __name__ == "__main__":
    main()
