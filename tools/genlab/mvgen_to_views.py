#!/usr/bin/env python3
"""MV-Adapter output -> a stage2 view set (views/{rgb,mask,cameras.json} + normals/).

Cameras: orthographic, width 1.1 (MV-Adapter's frustum is +-0.55, the same
ortho_scale this pipeline renders with), elevation 0, azimuths as generated.
--az-sign picks how MV-Adapter's azimuth maps onto ours (checked by eye once:
which side the second view shows).

Normals: each estimator has its own axis conventions, so they are fixed from
the silhouette: on the outline a surface faces straight outward in the image
plane, which fixes the signs of x and y; z is flipped if the object would face
away from the camera. The seed whose views agree best on their outlines (the
share of each silhouette the strict hull keeps) is chosen.

    python3 tools/genlab/mvgen_to_views.py --src data/catmv/mvgen --out data/catmv
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path[:0] = [os.path.dirname(__file__)]
from more_views import camera  # noqa: E402
from carve import grid, carve, silhouette  # noqa: E402

NAMES = {0: "01_front", 45: "d315", 90: "02_right", 180: "03_back", 270: "04_left", 315: "d225"}


def mask_of(img):
    a = np.asarray(img.convert("RGB"), np.float64)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, 0)
    d = np.abs(a - bg).max(-1)
    near = d < 12
    lab, _ = ndimage.label(near)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    m = ~np.isin(lab, edge[edge > 0])
    m = ndimage.binary_opening(ndimage.binary_fill_holes(m), iterations=1)
    lab, n = ndimage.label(m)
    return lab == 1 + np.argmax(ndimage.sum(m, lab, range(1, n + 1))) if n else m


def calibrate(n, m):
    """sign flips (sx, sy, sz) that make the rim normals point outward"""
    d_in = ndimage.distance_transform_edt(m)
    rim = m & (d_in <= 2) & (d_in > 0)
    g = np.stack(np.gradient(ndimage.gaussian_filter(m.astype(float), 2)), -1)  # (d/drow, d/dcol)
    out_x = -g[..., 1]; out_y = g[..., 0]          # outward in (x right, y up)
    sx = np.sign(np.sum(n[..., 0][rim] * out_x[rim])) or 1
    sy = np.sign(np.sum(n[..., 1][rim] * out_y[rim])) or 1
    inner = m & (d_in > 10)
    sz = np.sign(np.mean(n[..., 2][inner])) or 1
    return sx, sy, sz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--az-sign", type=int, default=1, help="+1: MV-Adapter +az orbits to the right")
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    files = glob.glob(f"{a.src}/view_s*_*.png")
    seeds = sorted({int(re.search(r"view_s(\d+)_", f).group(1)) for f in files})
    azs = sorted({int(re.search(r"_(\d+)\.png", f).group(1)) for f in files})
    meta = {"ortho_scale": 1.1, "resolution": [a.res, a.res], "views": {}}
    for z in azs:
        ours = (270 + a.az_sign * z) % 360           # ours: 270 = front, 0 = the right-side camera
        meta["views"][NAMES.get(z, f"az{z:03d}")] = {"matrix_world": camera(ours, 0)}
    h = meta["ortho_scale"] / 2
    _, X = grid(256, h)
    best = None
    for s in seeds:
        masks = {NAMES.get(z, f"az{z:03d}"): mask_of(Image.open(f"{a.src}/view_s{s}_{z:03d}.png")) for z in azs}
        occ = carve(meta, masks, X, 2.0, 1)
        cov = []
        for v, m in masks.items():
            sv = silhouette(meta, v, X[occ], m.shape[0], 2 * h / 256 / meta["ortho_scale"] * m.shape[0])
            cov.append((m & sv).sum() / m.sum())
        print(f"seed {s}: share of each silhouette all views agree on: "
              + " ".join(f"{c:.3f}" for c in cov) + f"  (min {min(cov):.3f})")
        if (a.seed is None and (best is None or min(cov) > best[0])) or a.seed == s:
            best = (min(cov), s, masks)
    _, s, masks = best
    print(f"using seed {s}")
    for d in ("views/rgb", "views/mask", "normals"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    json.dump(meta, open(f"{a.out}/views/cameras.json", "w"), indent=1)
    for z in azs:
        v = NAMES.get(z, f"az{z:03d}")
        Image.open(f"{a.src}/view_s{s}_{z:03d}.png").convert("RGB").save(f"{a.out}/views/rgb/{v}.png")
        Image.fromarray((masks[v] * 255).astype(np.uint8)).save(f"{a.out}/views/mask/{v}.png")
        npath = f"{a.src}/normal_s{s}_{z:03d}.npy"
        if os.path.exists(npath):
            n = np.load(npath).astype(np.float64)
            if n.shape[:2] != masks[v].shape:
                n = np.stack([np.asarray(Image.fromarray(n[..., c].astype(np.float32)).resize(
                    masks[v].shape[::-1], Image.BILINEAR)) for c in range(3)], -1)
            n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-9)
            sx, sy, sz = calibrate(n, masks[v])
            n *= np.array([sx, sy, sz])
            n[~masks[v]] = 0
            np.save(f"{a.out}/normals/{v}.npy", n.astype(np.float32))
            print(f"  {v}: normal signs {sx:+.0f} {sy:+.0f} {sz:+.0f}")


if __name__ == "__main__":
    main()
