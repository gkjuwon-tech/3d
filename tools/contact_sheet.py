#!/usr/bin/env python3
"""Tile a view directory into one contact sheet for a consistency eyeball pass,
and report per-view alignment stats.

Usage:
  python3 tools/contact_sheet.py refs/cinder_basilisk
"""
import os, sys
from PIL import Image

ORDER = ["01_front", "02_right", "03_back", "04_left", "05_top", "06_bottom"]
CELL = 512


def bbox_of_subject(im, bg_tol=18):
    """Bounding box of the non-background pixels, assuming a flat grey plate."""
    g = im.convert("L")
    bg = g.getpixel((2, 2))
    mask = g.point(lambda v: 255 if abs(v - bg) > bg_tol else 0)
    return mask.getbbox()


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "refs/cinder_basilisk"
    # rendered view sets keep the beauty pass in rgb/; generated ones are flat
    src = os.path.join(d, "rgb") if os.path.isdir(os.path.join(d, "rgb")) else d
    found = [(n, os.path.join(src, n + ".png")) for n in ORDER
             if os.path.exists(os.path.join(src, n + ".png"))]
    if not found:
        sys.exit(f"no views found in {d}")

    cols, rows = 3, (len(found) + 2) // 3
    sheet = Image.new("RGB", (cols * CELL, rows * CELL), (40, 40, 40))

    print(f"{'view':<12}{'size':>12}{'subject w x h':>16}{'fill %':>9}")
    for i, (name, path) in enumerate(found):
        im = Image.open(path).convert("RGB")
        bb = bbox_of_subject(im)
        if bb:
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            fill = 100.0 * (w * h) / (im.width * im.height)
            print(f"{name:<12}{f'{im.width}x{im.height}':>12}"
                  f"{f'{w}x{h}':>16}{fill:>8.1f}%")
        else:
            print(f"{name:<12}{f'{im.width}x{im.height}':>12}{'-':>16}{'-':>9}")
        im.thumbnail((CELL, CELL))
        sheet.paste(im, ((i % cols) * CELL + (CELL - im.width) // 2,
                         (i // cols) * CELL + (CELL - im.height) // 2))

    out = os.path.join(d, "_sheet.jpg")
    sheet.save(out, quality=88)
    print("\nwrote", out)
    print("\nAlignment check: front/back/top/bottom subject widths should match "
          "each other,\nand right/left lengths should match top/bottom lengths. "
          "Large gaps mean\nthe views were not rendered at the same scale.")


if __name__ == "__main__":
    main()
