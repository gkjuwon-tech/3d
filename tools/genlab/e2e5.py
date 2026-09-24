#!/usr/bin/env python3
"""E2: a two-panel sheet redrawn -- does each panel keep its registration?
E5: detailed normal map from an image + a coarse (proxy) normal map.

    python3 tools/genlab/e2e5.py make    # reference images
    python3 tools/genlab/e2e5.py gen     # both generations (parallel)
    python3 tools/genlab/e2e5.py score
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from gen_image import generate  # noqa: E402
import measure  # noqa: E402
import bands  # noqa: E402

OUT = "data/genlab"
GT = "refs/lucy_gt"
PAIR = ("01_front", "10_az315_up")
P = 1024          # panel size in the reference sheet
GUT = 64          # gutter between panels
MARK = 24         # registration square size, in the gutter corners

E2_PROMPT = """Redraw the attached reference sheet exactly.

It is a sheet of two panels: the same statue seen by two orthographic cameras, each
panel a square with a grey background, separated by a white gutter with small black
registration squares. Reproduce the whole sheet pixel-for-pixel in layout: the same
two panels in the same places and sizes, the same gutter and squares, the statue at
exactly the same position, size and pose inside each panel. Keep the same grey studio
look. Do not crop, zoom, re-centre, or change the aspect ratio: output a wide image
with the sheet's 2:1 proportions as closely as possible."""

E5_PROMPT = """Make a detailed surface-normal map of the statue in the first reference.

The second reference is a coarse normal map of the same statue from the same camera:
same position, same size, same outline. It uses the standard encoding
R = (x+1)/2, G = (y+1)/2, B = (z+1)/2 with x to the right, y up and z toward the
viewer, so surfaces facing the camera are light purple-blue (128,128,255), facing
right are pinkish, facing up are greenish. The background is black.

Your output: the same normal map, same encoding, same outline and same large-scale
colours as the coarse one, but with all the fine surface detail visible in the first
reference (folds, hair strands, feathers, face, fingers) added as correct normal
variation. It must be a normal map, not a picture: no lighting, no shadows, no
shading. Output a square image."""


def sheet():
    W = 2 * P + GUT
    im = Image.new("RGB", (W, P), (255, 255, 255))
    for i, v in enumerate(PAIR):
        im.paste(Image.open(f"{GT}/rgb/{v}.png").convert("RGB").resize((P, P), Image.LANCZOS),
                 (i * (P + GUT), 0))
    d = ImageDraw.Draw(im)
    x0 = P + (GUT - MARK) // 2
    for y in (8, P - 8 - MARK):
        d.rectangle([x0, y, x0 + MARK - 1, y + MARK - 1], fill=(0, 0, 0))
    return im


def coarse_normals(view, sig=12):
    nt, ok = measure.gt_cam_normals(view, GT)
    nb = bands.blur_n(nt, ok, sig)
    enc = ((nb + 1) / 2 * 255 + 0.5).clip(0, 255).astype(np.uint8)
    enc[~ok] = 0
    return Image.fromarray(enc)


def make():
    os.makedirs(f"{OUT}/e2", exist_ok=True); os.makedirs(f"{OUT}/e5", exist_ok=True)
    sheet().save(f"{OUT}/e2/ref_sheet.png")
    coarse_normals("02_right").save(f"{OUT}/e5/coarse_02_right.png")


def gen():
    jobs = [(E2_PROMPT, f"{OUT}/e2/gen_sheet.png", [f"{OUT}/e2/ref_sheet.png"]),
            (E5_PROMPT, f"{OUT}/e5/gen_normal_02_right.png",
             [f"{GT}/rgb/02_right.png", f"{OUT}/e5/coarse_02_right.png"])]

    def one(j):
        dt = generate(*j)
        print(f"wrote {j[1]} ({dt:.0f}s)", flush=True)
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(one, jobs))


def score():
    # E2: find the panels in the generated sheet by scaling the reference layout
    g = Image.open(f"{OUT}/e2/gen_sheet.png").convert("RGB")
    print(f"E2 generated sheet size {g.size} (reference {2*P+GUT}x{P})")
    ref = sheet()
    gs = np.asarray(g.resize(ref.size, Image.LANCZOS), np.float64)
    for i, v in enumerate(PAIR):
        x0 = i * (P + GUT)
        pg = gs[:, x0:x0 + P]
        gt = measure.load(f"{GT}/rgb/{v}.png", P)
        mt = measure.mask_from(gt, gt[0, 0]); mg = measure.mask_from(pg, pg[5, 5])
        raw = measure.iou(mg, mt)
        v_, s, ty, tx = measure.align(mg, mt)
        bd = measure.boundary_dist(measure.warp(mg, s, ty, tx) > 0.5, mt)
        bd0 = measure.boundary_dist(mg, mt)
        print(f"  panel {v}: raw IoU {raw:.4f} (boundary median {np.median(bd0):.1f}px, "
              f"p90 {np.percentile(bd0, 90):.1f}px at {P})  aligned IoU {v_:.4f} "
              f"scale {s:.4f} shift {ty:+.1f},{tx:+.1f}px")
    # E5
    S = 1024
    nt, ok = measure.gt_cam_normals("02_right", GT)
    rs = lambda x: np.stack([np.asarray(Image.fromarray(x[..., c].astype(np.float32)).resize(
        (S, S), Image.BILINEAR)) for c in range(3)], -1)
    nt = rs(nt); nt /= np.linalg.norm(nt, axis=-1, keepdims=True).clip(1e-9)
    gimg = measure.load(f"{GT}/rgb/02_right.png", S); m = measure.mask_from(gimg, gimg[0, 0])
    dec = lambda p: (lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9))(
        measure.load(p, S) / 255 * 2 - 1)
    bands.print_report("E5 coarse input (proxy, blur 12px@2048)", dec(f"{OUT}/e5/coarse_02_right.png"), nt, m)
    bands.print_report("E5 generated detailed normals", dec(f"{OUT}/e5/gen_normal_02_right.png"), nt, m)
    bands.print_report("E4 generated normals (no proxy)", dec(f"{OUT}/e4/normal_02_right.png"), nt, m)


if __name__ == "__main__":
    {"make": make, "gen": gen, "score": score}[sys.argv[1]]()
