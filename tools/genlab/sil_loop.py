#!/usr/bin/env python3
"""Make the generated views agree on their outlines, before anything else.

Generated turnaround views disagree by 10-25% of their silhouettes (the cat:
skirt width, flame size, tail), and no fusion can make one object of views that
disagree about its outline. Each round: carve a tolerant hull from the current
silhouettes (a voxel goes only when two views call it empty), project it into
every view -- that projection is the outline every view is asked to fill --
and have each opposite-view sheet redrawn to fill its two outlines exactly.
Stops when every view's silhouette matches the hull's projection to --target.

    python3 tools/genlab/sil_loop.py --start data/catgen/views --out data/cat/sil \
        --style data/cat/gen/sheet1_a.png data/cat/gen/sheet2_d.png
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from gen_image import generate  # noqa: E402
from carve import grid, carve, silhouette  # noqa: E402
from sil_check import sil  # noqa: E402
from turnaround import P, GUT, BG, ticks, panels  # noqa: E402

PAIRS = [("01_front", "03_back"), ("02_right", "04_left"), ("d315", "d135"), ("d045", "d225")]

PROMPT = """Redraw a two-panel ORTHOGRAPHIC view sheet of one sculpture (matte white clay,
mid-grey #808080 background, even soft studio light, no perspective).

The second attached image is the current sheet: the sculpture seen by the same two
cameras. The first attached image is the same sheet layout with a light-grey TARGET
SHAPE in each panel. Redraw each panel's sculpture so that its outline matches that
target shape exactly: fill the whole shape and nothing outside it. Keep the sculpture's
pose, parts and details from the current sheet, only adjust proportions and positions
(skirt width, arms, tail, flames, head) so the outline fits the target shape. The target
shapes come from the sculpture's 3D volume, so fitting them makes every camera agree.

Same scale and placement as the target shapes. Keep the white gutter and the tick marks.
Keep the wide 2:1 sheet."""

STYLE = """

Any further attached images show the same sculpture from other cameras: keep its look."""


def outline(meta, v, pts, n, h, res):
    s = silhouette(meta, v, pts, res, 2 * h / n / meta["ortho_scale"] * res)
    s = ndimage.binary_closing(s, iterations=2)
    s = ndimage.binary_fill_holes(s) if False else s
    # the voxel footprint splat overstates the outline by about half a voxel
    return ndimage.binary_erosion(s, iterations=1)


def sheet(imgs):
    im = Image.new("RGB", (2 * P + GUT, P), (255, 255, 255))
    for j, x in enumerate(imgs):
        im.paste(x.convert("RGB").resize((P, P), Image.LANCZOS), (j * (P + GUT), 0))
    d = ImageDraw.Draw(im)
    for j in range(2):
        ticks(d, j * (P + GUT))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="view set: cameras.json, rgb/, mask/ (registered)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--target", type=float, default=0.97)
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--dilate", type=float, default=2.0)
    ap.add_argument("--votes", type=int, default=2)
    ap.add_argument("--style", nargs="*", default=[])
    ap.add_argument("--keep-above", type=float, default=0.975,
                    help="a sheet whose views are both covered this well is not redrawn")
    ap.add_argument("--pairs", default=None, help="a:b,c:d,... opposite-view sheets (default: the ring of 8)")
    a = ap.parse_args()
    global PAIRS
    if a.pairs:
        PAIRS = [tuple(p.split(":")) for p in a.pairs.split(",")]
    meta = json.load(open(f"{a.start}/cameras.json"))
    views = [v for p in PAIRS for v in p]
    r0 = f"{a.out}/r0"
    if not os.path.exists(r0):
        os.makedirs(f"{r0}/rgb"); os.makedirs(f"{r0}/mask")
        shutil.copy(f"{a.start}/cameras.json", r0)
        for v in views:
            shutil.copy(f"{a.start}/rgb/{v}.png", f"{r0}/rgb/{v}.png")
            shutil.copy(f"{a.start}/mask/{v}.png", f"{r0}/mask/{v}.png")
    h = meta["ortho_scale"] / 2
    _, X = grid(a.n, h)
    for r in range(a.rounds + 1):
        rd = f"{a.out}/r{r}"
        masks = {v: np.asarray(Image.open(f"{rd}/mask/{v}.png").convert("L")) > 127 for v in views}
        occ = carve(meta, masks, X, a.dilate, a.votes)
        pts = X[occ]
        tgt = {v: outline(meta, v, pts, a.n, h, P) for v in views}
        strict = X[carve(meta, masks, X, a.dilate, 1)]
        ious = {}
        covers = {}
        line = []
        for v in views:
            m = masks[v]
            ious[v] = (m & tgt[v]).sum() / (m | tgt[v]).sum()
            sv = silhouette(meta, v, strict, P, 2 * h / a.n / meta["ortho_scale"] * P)
            cover = (m & sv).sum() / m.sum()
            covers[v] = cover
            line.append(f"{v} {ious[v]:.3f}/{cover:.3f}")
        print(f"round {r}: outline IoU / covered by the strict hull:  " + "  ".join(line), flush=True)
        np.savez_compressed(f"{rd}/hull.npz", occ=occ.reshape(a.n, a.n, a.n), lo=-h, hi=h)
        if min(ious.values()) >= a.target or r == a.rounds:
            print(f"stop at round {r} (worst outline IoU {min(ious.values()):.3f})")
            break
        nd = f"{a.out}/r{r+1}"
        os.makedirs(f"{nd}/rgb", exist_ok=True); os.makedirs(f"{nd}/mask", exist_ok=True)
        shutil.copy(f"{rd}/cameras.json", nd)
        jobs = []
        keep = set()
        for k, pair in enumerate(PAIRS):
            if all(covers[v] >= a.keep_above for v in pair):
                keep.add(k)            # this sheet already agrees: carry it forward unchanged
                for v in pair:
                    shutil.copy(f"{rd}/rgb/{v}.png", f"{nd}/rgb/{v}.png")
                    shutil.copy(f"{rd}/mask/{v}.png", f"{nd}/mask/{v}.png")
                continue
            pans = []
            for v in pair:
                pan = np.full((P, P, 3), BG, np.uint8); pan[tgt[v]] = 200
                pans.append(Image.fromarray(pan))
            ref_t = f"{nd}/s{k}_target.png"; sheet(pans).save(ref_t)
            ref_c = f"{nd}/s{k}_current.png"; sheet([Image.open(f"{rd}/rgb/{v}.png") for v in pair]).save(ref_c)
            out = f"{nd}/s{k}_gen.png"
            if not os.path.exists(out):
                jobs.append((PROMPT + (STYLE if a.style else ""), out, [ref_t, ref_c] + list(a.style)))
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda j: print(f"  wrote {os.path.basename(j[1])} ({generate(*j):.0f}s)", flush=True), jobs))
        for k, pair in enumerate(PAIRS):
            if k in keep:
                continue
            for v, pnl in zip(pair, panels(f"{nd}/s{k}_gen.png")):
                pnl.save(f"{nd}/rgb/{v}.png")
                Image.fromarray((sil(pnl) * 255).astype(np.uint8)).save(f"{nd}/mask/{v}.png")


if __name__ == "__main__":
    main()
