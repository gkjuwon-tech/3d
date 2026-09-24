#!/usr/bin/env python3
"""Proxy mesh + per-view detail depth -> detailed mesh.

Every vertex of the (subdivided) proxy takes its displacement from exactly one
view, the one that sees it most squarely -- the same ownership rule the painter
used, so the detail a vertex gets is the detail its owner painted. The move is
along that view's ray, by the owner's integrated depth minus the proxy depth.

    python3 tools/genlab/displace.py --proxy data/genlab/proxy2/lucy_proxy_tri.ply \
        --views data/genlab/proxy2/views --depth data/genlab/paint2/depth \
        --out data/genlab/paint2/mesh.ply
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_hull import read_ply  # noqa: E402


def subdivide(v, f):
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.ravel()
    mid = len(v) + np.arange(len(uniq))
    v2 = np.concatenate([v, 0.5 * (v[uniq[:, 0]] + v[uniq[:, 1]])])
    m = mid[inv].reshape(3, -1).T          # midpoints of edges 01, 12, 20
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    f2 = np.concatenate([np.stack([a, m[:, 0], m[:, 2]], 1), np.stack([m[:, 0], b, m[:, 1]], 1),
                         np.stack([m[:, 2], m[:, 1], c], 1), np.stack([m[:, 0], m[:, 1], m[:, 2]], 1)])
    return v2, f2


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)


def write_ply(path, v, f):
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  f"element face {len(f)}\nproperty list uchar int vertex_indices\n"
                  "end_header\n").encode())
        v.astype("<f4").tofile(fh)
        rec = np.empty(len(f), dtype=[("n", "u1"), ("v", "<i4", 3)])
        rec["n"] = 3; rec["v"] = f
        rec.tofile(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--depth", required=True, help="per-view detail depth (detail_depth.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--edge-px", type=float, default=1.0, help="subdivide to about this edge length")
    a = ap.parse_args()
    meta = json.load(open(f"{a.views}/cameras.json"))
    v, f = read_ply(a.proxy)
    v = v.astype(np.float64); f = f.astype(np.int64)
    names = list(meta["views"])
    res = np.load(f"{a.views}/depth_npy/{names[0]}.npy").shape[0]
    px = meta["ortho_scale"] / res
    while True:
        el = np.median(np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1))
        if el < a.edge_px * px * 1.5:
            break
        v, f = subdivide(v, f)
    print(f"subdivided: {len(v):,} verts {len(f):,} faces, edge {el/px:.2f} px", flush=True)
    n = vertex_normals(v, f)
    best = np.full(len(v), -2.0); disp = np.zeros(len(v)); dirs = np.zeros_like(v)
    for name in names:
        if not os.path.exists(f"{a.depth}/{name}.npy"):
            continue
        m = np.array(meta["views"][name]["matrix_world"])
        right, up, back, loc = m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3]
        d0 = np.load(f"{a.views}/depth_npy/{name}.npy").astype(np.float64)
        dz = np.load(f"{a.depth}/{name}.npy").astype(np.float64)
        h = np.where(np.isfinite(dz) & (d0 < 1e3), dz - d0, np.nan)
        rel = v - loc
        col = (rel @ right / meta["ortho_scale"] + 0.5) * res - 0.5
        row = (-(rel @ up) / meta["ortho_scale"] + 0.5) * res - 0.5
        z = rel @ (-back)
        ri, ci = np.round(row).astype(int), np.round(col).astype(int)
        inb = (ri >= 0) & (ri < res) & (ci >= 0) & (ci < res)
        ri, ci = ri.clip(0, res - 1), ci.clip(0, res - 1)
        vis = inb & (d0[ri, ci] < 1e3) & (np.abs(d0[ri, ci] - z) < 2 * px)
        hv = ndimage.map_coordinates(np.nan_to_num(h, nan=0.0), [row, col], order=1)
        okh = ndimage.map_coordinates(np.isfinite(h).astype(float), [row, col], order=1) > 0.99
        cos = n @ back
        take = vis & okh & (cos > best)
        best[take] = cos[take]; disp[take] = hv[take]; dirs[take] = -back
    moved = best > -2
    v = v + disp[:, None] * dirs
    print(f"displaced {100*moved.mean():.1f}% of vertices; |h| median {np.median(np.abs(disp[moved]))/px:.2f} px, "
          f"p99 {np.percentile(np.abs(disp[moved]), 99)/px:.2f} px", flush=True)
    write_ply(a.out, v.astype(np.float32), f.astype(np.int32))


if __name__ == "__main__":
    main()
