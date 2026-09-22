#!/usr/bin/env python3
"""Validate a rendered ground-truth view set and produce viewable previews of
the float passes.

Checks that the six views really are one consistent object seen from six
directions — the property the fusion stage depends on and the reason this
render exists at all:

  * silhouette extents must agree across views
        front/back/top/bottom  -> same width   (object X)
        right/left/top/bottom  -> same length  (object Y)
        front/back/right/left  -> same height  (object Z)
  * depth must lie inside the normalized bounding box
  * normals must be unit length inside the mask

Writes preview/<view>_depth.png and preview/<view>_normal.png, the latter in
camera space with the usual [0,1] encoding.

Run:
  assets/blender/blender -b -P tools/inspect_gt.py -- --dir refs/lucy_gt
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

# which object axis each view's silhouette width and height correspond to
EXTENT_AXES = {
    "01_front":  ("X", "Z"),
    "02_right":  ("Y", "Z"),
    "03_back":   ("X", "Z"),
    "04_left":   ("Y", "Z"),
    "05_top":    ("X", "Y"),
    "06_bottom": ("X", "Y"),
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    return p.parse_args(argv)


def load_pixels(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    # Blender hands back rows bottom-up; flip so row 0 is the top of the frame.
    return px.reshape(h, w, 4)[::-1]


def save_rgb(arr, path):
    """arr: (h, w, 3) float in [0,1]."""
    h, w, _ = arr.shape
    img = bpy.data.images.new(os.path.basename(path), width=w, height=h,
                              alpha=False, float_buffer=False)
    img.colorspace_settings.name = "Non-Color"
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(arr, 0.0, 1.0)
    img.pixels.foreach_set(rgba[::-1].ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def main():
    args = parse_args()
    root = os.path.abspath(args.dir)
    meta = json.load(open(os.path.join(root, "cameras.json")))
    os.makedirs(os.path.join(root, "preview"), exist_ok=True)

    ortho = meta["ortho_scale"]
    res = meta["resolution"][0]
    units_per_px = ortho / res

    extents = {}
    stats = {}
    ok_depth = ok_normal = True
    print(f"{'view':<11}{'mask px':>10}{'cover%':>8}"
          f"{'width':>9}{'height':>9}{'depth min':>11}{'depth max':>11}"
          f"{'|n| err':>10}{'bbox':>7}")

    for view, info in meta["views"].items():
        mask = load_pixels(os.path.join(root, "mask", f"{view}.png"))[..., 0]
        depth = load_pixels(os.path.join(root, "depth", f"{view}.exr"))[..., 0]
        normal = load_pixels(os.path.join(root, "normal", f"{view}.exr"))[..., :3]

        # Extents come from the anti-aliased mask, which carries sub-pixel
        # silhouette information. Float-pass statistics come from the geometry
        # pass's own coverage: anything still at the background sentinel was
        # never hit, and the pass is point-sampled so there is no blended edge
        # to exclude.
        m = mask > 0.5
        solid = depth < 1e9
        n_px = int(m.sum())
        ys, xs = np.nonzero(m)
        w_px = xs.max() - xs.min() + 1
        h_px = ys.max() - ys.min() + 1
        width = w_px * units_per_px
        height = h_px * units_per_px

        d_in = depth[solid]
        d_lo, d_hi = float(d_in.min()), float(d_in.max())

        nn = normal[solid]
        norms = np.linalg.norm(nn, axis=1)
        n_err = float(np.abs(norms - 1.0).max())

        extents[view] = (width, height)
        # the camera sits 2.0 from the origin, so depth must land inside
        # 2.0 +/- half the normalized bounding box along the view axis
        half = max(meta["normalization"]["normalized_size"]) / 2 + 1e-3
        in_box = (d_lo >= 2.0 - half) and (d_hi <= 2.0 + half)
        print(f"{view:<11}{n_px:>10,}{100.0*n_px/m.size:>7.2f}%"
              f"{width:>9.4f}{height:>9.4f}{d_lo:>11.4f}{d_hi:>11.4f}"
              f"{n_err:>10.5f}{'OK' if in_box else 'FAIL':>7}")
        ok_depth = ok_depth and in_box
        ok_normal = ok_normal and n_err < 1e-3
        stats[view] = {"n_err_mean": float(np.abs(norms - 1.0).mean()),
                       "n_err_max": n_err}

        # depth preview, normalized across the object's own depth range
        dv = np.zeros_like(depth)
        dv[m] = 1.0 - (np.clip(depth[m], d_lo, d_hi) - d_lo) / max(d_hi - d_lo, 1e-9)
        save_rgb(np.repeat(dv[..., None], 3, axis=2),
                 os.path.join(root, "preview", f"{view}_depth.png"))

        # world -> camera space normals, then the standard [0,1] encoding
        cam = Matrix(info["matrix_world"]).to_3x3()
        R = np.array(cam.inverted().transposed()).T
        nc = normal @ R.T
        enc = np.where(m[..., None], nc * 0.5 + 0.5, 0.5)
        save_rgb(enc, os.path.join(root, "preview", f"{view}_normal.png"))

    print("\ncross-view extent agreement (should match to ~1 pixel):")
    axis_vals = {}
    for view, (w, h) in extents.items():
        aw, ah = EXTENT_AXES[view]
        axis_vals.setdefault(aw, []).append((view, w))
        axis_vals.setdefault(ah, []).append((view, h))

    ok = True
    for axis in "XYZ":
        vals = axis_vals.get(axis, [])
        if len(vals) < 2:
            continue
        sizes = [v for _, v in vals]
        spread = max(sizes) - min(sizes)
        spread_px = spread / units_per_px
        flag = "OK " if spread_px <= 2.0 else "*** "
        ok &= spread_px <= 2.0
        print(f"  {flag}axis {axis}: {np.mean(sizes):.4f} "
              f"+/- {spread/2:.5f}  ({spread_px:.2f} px spread)  "
              + ", ".join(f"{v}={s:.4f}" for v, s in vals))

    box = meta["normalization"]["normalized_size"]
    print(f"\nnormalized bbox: {['%.4f' % b for b in box]}")
    print("alignment :", "PASS" if ok else "FAIL")
    print("depth     :", "PASS" if ok_depth else "FAIL")
    worst = max(stats.values(), key=lambda s: s["n_err_max"])
    print(f"normals   : {'PASS' if ok_normal else 'FAIL'}"
          f"   (worst |n|-1: max {worst['n_err_max']:.2e}, "
          f"mean {worst['n_err_mean']:.2e})")
    if not (ok and ok_depth and ok_normal):
        sys.exit(1)


if __name__ == "__main__":
    main()
