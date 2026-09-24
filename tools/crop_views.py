#!/usr/bin/env python3
"""Zoomed crops of the views themselves, as extra cameras for mesh_fit.

A normal estimator sees a 768 px view with a face 60 px across and returns a
blur there. Cropping the same region out of every view and enlarging it gives
the estimator 3x the pixels on it, while the images stay exactly the ones the
views agree on (a fresh generation of the close-up, tried first, imagined a
different pose: 62% overlap with the main views from the side). Each crop is
an orthographic camera of its own -- the parent camera shifted to the crop's
centre, with the crop's width as its ortho scale -- so the alignment is exact.

The region is a world-space height band (e.g. everything above the chest),
projected into each view; its crop is the band's bounding square there.

  python3 tools/crop_views.py --views data/cat6/views --out data/cat6_head/views \
      --zmin -0.07 --pad 0.04
  python3 tools/crop_views.py --views data/cat6/views --out data/cat6_tiles/views --tiles 3
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zmin", type=float, default=None, help="world height where the band starts")
    ap.add_argument("--tiles", type=int, default=0,
                    help="instead of a band: an n x n grid of overlapping tiles over each view's object")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--zmax", type=float, default=9.0)
    ap.add_argument("--pad", type=float, default=0.03, help="world units of margin")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--weight", type=float, default=1.0)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.views, "cameras.json")))
    O = float(meta["ortho_scale"])
    views = {}
    for d in ("rgb", "mask"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    def emit(v, W, m, rgb, R, box, name):
        side = box[2] - box[0]
        cc, rc = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        rgb.crop(box).resize((a.size, a.size), Image.LANCZOS).save(os.path.join(a.out, "rgb", f"{name}.png"))
        Image.fromarray((m * 255).astype(np.uint8)).crop(box).resize((a.size, a.size), Image.BILINEAR) \
            .save(os.path.join(a.out, "mask", f"{name}.png"))
        # camera: parent shifted to the crop centre, ortho = crop width in world units
        Wc = W.copy()
        Wc[:3, 3] = W[:3, 3] + (cc / R - 0.5) * O * W[:3, 0] + (0.5 - rc / R) * O * W[:3, 1]
        views[name] = {"matrix_world": Wc.tolist(), "ortho_scale": side / R * O,
                       "weight": a.weight, "parent": v, "box_px": [float(x) for x in box]}
        print(f"{name}: {side:.0f} px -> {a.size} (x{a.size / side:.2f})")

    for v, info in meta["views"].items():
        W = np.array(info["matrix_world"])
        m = np.asarray(Image.open(os.path.join(a.views, "mask", f"{v}.png"))) > 127
        rgb = Image.open(os.path.join(a.views, "rgb", f"{v}.png")).convert("RGB")
        R = m.shape[0]
        if a.tiles:
            ys, xs = np.nonzero(m)
            span = max(ys.max() - ys.min(), xs.max() - xs.min()) + 8
            side = span / (a.tiles - (a.tiles - 1) * a.overlap)
            step = side * (1 - a.overlap)
            y0 = (ys.min() + ys.max()) / 2 - span / 2
            x0 = (xs.min() + xs.max()) / 2 - span / 2
            for i in range(a.tiles):
                for j in range(a.tiles):
                    box = (x0 + j * step, y0 + i * step, x0 + j * step + side, y0 + i * step + side)
                    b = [int(max(0, np.floor(box[1]))), int(min(R, np.ceil(box[3]))),
                         int(max(0, np.floor(box[0]))), int(min(R, np.ceil(box[2])))]
                    if m[b[0]:b[1], b[2]:b[3]].mean() < 0.08:
                        continue
                    emit(v, W, m, rgb, R, box, f"tile_{v}_{i}{j}")
            continue
        # rows of the band: world height z maps to row (0.5 - (z - cz)/O) R for a level view
        up = W[:3, 1]
        cz = W[:3, 3] @ up
        row_of = lambda z: (0.5 - (z - cz) / O) * R  # noqa: E731
        r_top = max(0, int(np.floor(row_of(min(a.zmax, 1e3) + a.pad))))
        r_bot = min(R, int(np.ceil(row_of(a.zmin - a.pad))))
        rows = np.nonzero(m[r_top:r_bot].any(1))[0] + r_top
        cols = np.nonzero(m[r_top:r_bot].any(0))[0]
        if not len(rows):
            continue
        pad_px = a.pad / O * R
        r0, r1 = rows.min() - pad_px, min(rows.max() + pad_px, r_bot)
        c0, c1 = cols.min() - pad_px, cols.max() + pad_px
        side = max(r1 - r0, c1 - c0)
        rc, cc = (r0 + r1) / 2, (c0 + c1) / 2
        emit(v, W, m, rgb, R, (cc - side / 2, rc - side / 2, cc + side / 2, rc + side / 2), f"crop_{v}")
    json.dump({"ortho_scale": O, "resolution": [a.size, a.size], "views": views},
              open(os.path.join(a.out, "cameras.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
