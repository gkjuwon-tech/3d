#!/usr/bin/env python3
"""Proxy from generated silhouettes: a tolerant visual hull.

Generated views agree only roughly (mirror IoU ~0.9), so each silhouette is
dilated before carving -- a voxel is removed only where a view is sure it is
empty -- and the hull is then smoothed into a mesh. Reports, per view, how well
the hull's own silhouette matches the input (the check that the views agree).

    python3 tools/genlab/carve.py --views data/cat/views --out data/cat/proxy
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure as skm

sys.path[:0] = [os.path.dirname(__file__)]
from displace import write_ply  # noqa: E402
import measure  # noqa: E402


def cam(meta, v):
    M = np.array(meta["views"][v]["matrix_world"])
    return M[:3, 0], M[:3, 1], M[:3, 2]


def project(meta, v, X, res):
    r, u, _ = cam(meta, v)
    o = meta["ortho_scale"]
    return (-(X @ u) / o + 0.5) * res - 0.5, ((X @ r) / o + 0.5) * res - 0.5


def grid(n, h):
    c = (np.arange(n) + 0.5) / n * 2 * h - h
    return c, np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)


def carve(meta, masks, X, dilate, votes=1):
    """a voxel goes when `votes` views see it as empty (1: plain intersection)"""
    empty = np.zeros(len(X), np.uint8)
    for v, m in masks.items():
        res = m.shape[0]
        md = ndimage.distance_transform_edt(~m) <= dilate
        r, cc = project(meta, v, X, res)
        ri, ci = np.round(r).astype(int), np.round(cc).astype(int)
        inb = (ri >= 0) & (ri < res) & (ci >= 0) & (ci < res)
        keep = np.zeros(len(X), bool)
        keep[inb] = md[ri[inb], ci[inb]]
        empty += ~keep
    return empty < votes


def silhouette(meta, v, pts, res, vox_px):
    r, cc = project(meta, v, pts, res)
    s = np.zeros((res, res), bool)
    s[np.round(r).astype(int).clip(0, res - 1), np.round(cc).astype(int).clip(0, res - 1)] = True
    # each voxel covers about vox_px pixels: fill its footprint
    return ndimage.binary_dilation(s, iterations=max(1, int(np.ceil(vox_px / 2))))


def register(meta, masks, ref, h, dilate, rounds=2, n=160):
    """Per-view scale + shift so each view agrees with the hull of the others
    (the reference view stays put). Generated views place the turntable axis a
    few pixels differently each; left alone, every carve cuts those pixels off."""
    _, X = grid(n, h)
    vox_px = 2 * h / n / meta["ortho_scale"] * next(iter(masks.values())).shape[0]
    for rd in range(rounds):
        for v in masks:
            if v == ref:
                continue
            others = {k: m for k, m in masks.items() if k != v}
            occ = carve(meta, others, X, dilate)
            sil = silhouette(meta, v, X[occ], masks[v].shape[0], vox_px)
            iou0 = measure.iou(masks[v], sil)
            iou, s, ty, tx = measure.align(masks[v], sil)
            if iou > iou0 + 0.002:
                masks[v] = measure.warp(masks[v], s, ty, tx) > 0.5
            print(f"  register round {rd} {v:12}: IoU vs others' hull {iou0:.3f} -> {max(iou, iou0):.3f}"
                  f"  (scale {s:.3f} shift {ty:+.1f},{tx:+.1f})", flush=True)
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True, help="cameras.json + mask/<view>.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=320, help="voxels per axis")
    ap.add_argument("--dilate", type=float, default=6.0, help="px of tolerance per silhouette")
    ap.add_argument("--smooth", type=float, default=1.5, help="voxels of smoothing before meshing")
    ap.add_argument("--register", type=int, default=2, help="registration rounds (0: off)")
    ap.add_argument("--ref", default="01_front", help="view that defines the frame")
    ap.add_argument("--votes", type=int, default=1,
                    help="views that must see a voxel as empty to remove it; 2 lets any "
                         "one generated view be wrong without cutting into the others")
    a = ap.parse_args()
    meta = json.load(open(f"{a.views}/cameras.json"))
    views = [v for v in meta["views"] if os.path.exists(f"{a.views}/mask/{v}.png")]
    h = meta["ortho_scale"] / 2
    c, X = grid(a.n, h)
    masks = {v: np.asarray(Image.open(f"{a.views}/mask/{v}.png").convert("L")) > 127 for v in views}
    if a.register:
        masks = register(meta, masks, a.ref, h, a.dilate, a.register)
        os.makedirs(f"{a.views}/mask_reg", exist_ok=True)
        for v, m in masks.items():
            Image.fromarray((m * 255).astype(np.uint8)).save(f"{a.views}/mask_reg/{v}.png")
    occ = carve(meta, masks, X, a.dilate, a.votes).reshape(a.n, a.n, a.n)
    lab, k = ndimage.label(occ)
    if k > 1:
        occ = lab == 1 + np.argmax(ndimage.sum(occ, lab, range(1, k + 1)))
    print(f"hull: {occ.sum():,} voxels of {a.n}^3 from {len(views)} views", flush=True)
    # per-view agreement: the hull's silhouette vs the input silhouette
    pts = X[occ.ravel()]
    for v in views:
        m = masks[v]; res = m.shape[0]
        s = silhouette(meta, v, pts, res, 2 * h / a.n / meta["ortho_scale"] * res)
        miss = (m & ~s).sum() / m.sum(); extra = (s & ~m).sum() / m.sum()
        print(f"  {v:15} IoU {(m & s).sum() / (m | s).sum():.3f}  input not covered {100*miss:.1f}%"
              f"  hull beyond input {100*extra:.1f}%", flush=True)
    os.makedirs(a.out, exist_ok=True)
    f = ndimage.gaussian_filter(occ.astype(np.float32), a.smooth)
    vs, fs, _, _ = skm.marching_cubes(np.pad(f, 1), 0.5)
    vs = (vs - 1 + 0.5) / a.n * 2 * h - h
    write_ply(f"{a.out}/proxy.ply", vs.astype(np.float32), fs.astype(np.int32))
    np.savez_compressed(f"{a.out}/hull.npz", occ=occ, lo=-h, hi=h)
    print(f"mesh: {len(vs):,} verts {len(fs):,} faces -> {a.out}/proxy.ply")


if __name__ == "__main__":
    main()
