#!/usr/bin/env python3
"""Hero front view -> orthographic turnaround sheets (opposite views paired).

  frame   : cut the hero out of its background and place it in a standard panel
            (figure height 80% of the panel, base centred, bottom at 92%)
  sheet1  : [front | back]   the front panel is the framed hero itself
  sheet2  : [right | left]   with sheet 1 shown as a reference

Tick marks at the panel edges mark the rows where the figure's top and bottom
must sit, so every panel shares one scale and one vertical placement.

    python3 tools/genlab/turnaround.py --hero data/cat/gen/hero_front_a.png --out data/cat/gen
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from gen_image import generate  # noqa: E402
import measure  # noqa: E402

P, GUT, BG = 1024, 64, (128, 128, 128)
TOP, BOT = 0.12, 0.92          # figure extent, as fractions of the panel height

SHEET1 = """A two-panel ORTHOGRAPHIC turnaround sheet of one sculpture (matte white clay,
mid-grey #808080 background, even soft studio light, no perspective).

The attached image is this sheet with its left panel already done: the FRONT view.
Copy the left panel exactly as it is, pixel for pixel.

Fill the right panel with the BACK view of the very same sculpture: the camera has
orbited exactly 180 degrees around the vertical axis. Everything is left-right
mirrored compared to the front view: what was on the image left is now on the image
right. Same sculpture, same pose, same proportions, same scale: the figure's top and
the bottom of its base sit exactly at the rows marked by the small black tick marks at
the panel edges, like in the left panel. The round base is centred in the panel. Show
the back of the head and ears, the back of the dress and its lacing, the tail, and the
flames from behind. Keep the white gutter and the tick marks. Keep the wide 2:1 sheet."""

SHEET2 = """A two-panel ORTHOGRAPHIC turnaround sheet of one sculpture (matte white clay,
mid-grey #808080 background, even soft studio light, no perspective).

The second attached image shows the FRONT view (left) and BACK view (right) of this
sculpture. The first attached image is the empty sheet to fill, with small black tick
marks at the panel edges.

Left panel: the view from the sculpture's LEFT-hand side as seen in the front view's
image-right... precisely: the camera orbits 90 degrees to the RIGHT from the front view
(it now stands where the front view's right image edge was, looking back toward the
centre). Right panel: the opposite side, the camera orbited 90 degrees to the LEFT from
the front view. Both are true side profiles.

Same sculpture, same pose, same proportions and exactly the same scale as the front and
back views: the figure's top and the bottom of its base sit at the rows marked by the
tick marks. The round base is centred in each panel and has the same width as in the
front view. The flames keep their direction in 3D, so in these side views they are seen
from the side and foreshortened accordingly. Keep the white gutter and the tick marks.
Keep the wide 2:1 sheet."""


def frame(hero_path):
    g = np.asarray(Image.open(hero_path).convert("RGB"), np.float64)
    m = measure.mask_from(g, np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]), 0), tol=14)
    lab, n = ndimage.label(m)
    m = lab == 1 + np.argmax(ndimage.sum(m, lab, range(1, n + 1)))
    ys, xs = np.nonzero(m)
    y0, y1 = ys.min(), ys.max()
    # the base's centre: the widest rows near the bottom
    rows = m[y1 - (y1 - y0) // 20:y1 + 1]
    bx = np.nonzero(rows.any(0))[0]; cx = (bx.min() + bx.max()) / 2
    s = (BOT - TOP) * P / (y1 - y0 + 1)
    W, H = g.shape[1], g.shape[0]
    im = Image.fromarray(np.where(m[..., None], g, BG).astype(np.uint8)).resize(
        (round(W * s), round(H * s)), Image.LANCZOS)
    mk = Image.fromarray((m * 255).astype(np.uint8)).resize(im.size, Image.LANCZOS)
    pan = Image.new("RGB", (P, P), BG)
    ox = round(P / 2 - cx * s); oy = round(BOT * P - (y1 + 1) * s)
    pan.paste(im, (ox, oy), mk)
    return pan


def ticks(d, x0):
    for fr in (TOP, BOT):
        y = round(fr * P)
        d.rectangle([x0, y - 2, x0 + 14, y + 1], fill=(0, 0, 0))
        d.rectangle([x0 + P - 15, y - 2, x0 + P - 1, y + 1], fill=(0, 0, 0))


def sheet(left=None, right=None):
    im = Image.new("RGB", (2 * P + GUT, P), (255, 255, 255))
    for i, pnl in enumerate((left, right)):
        im.paste(pnl if pnl is not None else Image.new("RGB", (P, P), BG), (i * (P + GUT), 0))
    d = ImageDraw.Draw(im)
    for i in range(2):
        ticks(d, i * (P + GUT))
    return im


def panels(path):
    g = Image.open(path).convert("RGB").resize((2 * P + GUT, P), Image.LANCZOS)
    return [g.crop((i * (P + GUT), 0, i * (P + GUT) + P, P)) for i in range(2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tries", type=int, default=2, help="candidates per sheet")
    ap.add_argument("--step", choices=["1", "2"], default="1")
    ap.add_argument("--sheet1", default=None, help="chosen sheet-1 image for step 2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    front = frame(a.hero); front.save(f"{a.out}/panel_front.png")
    if a.step == "1":
        ref = f"{a.out}/sheet1_ref.png"; sheet(front).save(ref)
        jobs = [(SHEET1, f"{a.out}/sheet1_{k}.png", [ref]) for k in "abcd"[:a.tries]]
    else:
        ref = f"{a.out}/sheet2_ref.png"; sheet().save(ref)
        jobs = [(SHEET2, f"{a.out}/sheet2_{k}.png", [ref, a.sheet1]) for k in "abcd"[:a.tries]]
    jobs = [j for j in jobs if not os.path.exists(j[1])]
    with ThreadPoolExecutor(len(jobs) or 1) as ex:
        list(ex.map(lambda j: print(f"wrote {j[1]} ({generate(*j):.0f}s)", flush=True), jobs))


if __name__ == "__main__":
    main()
