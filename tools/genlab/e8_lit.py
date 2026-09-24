#!/usr/bin/env python3
"""E8: does a lit appearance image give truer detail normals than a flat one?

Same view, same coarse proxy normals (proxy2); the appearance reference is
(a) the flat studio render, (b) the ground-truth normals shaded by one raking
light (Lambertian, no shadows), (c) three such lights as three references.

    python3 tools/genlab/e8_lit.py gen|score
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from gen_image import generate  # noqa: E402
import measure  # noqa: E402
import bands  # noqa: E402

GT, PX, OUT, V = "refs/lucy_gt", "data/genlab/proxy2/views", "data/genlab/e8", "02_right"
LIGHTS = {"ul": (-0.55, 0.55, 0.63), "r": (0.8, 0.1, 0.6), "d": (0.1, -0.75, 0.65)}

PROMPT = """Make a detailed surface-normal map of the statue in the first reference image{s}.

The last reference is a coarse normal map of the same statue from the same camera:
same position, same size, same outline. It uses the standard encoding
R = (x+1)/2, G = (y+1)/2, B = (z+1)/2 with x to the right, y up and z toward the
viewer, so surfaces facing the camera are light purple-blue (128,128,255). The
background is black.

Your output: the same normal map, same encoding, same outline and same large-scale
colours as the coarse one, but with all the fine surface detail visible in the
photo reference{s} (folds, hair strands, feathers, face, fingers) added as correct
normal variation.{extra} It must be a normal map, not a picture: no lighting, no
shadows. Output a square image."""


def lit(L):
    nt, ok = measure.gt_cam_normals(V, GT)
    L = np.asarray(L) / np.linalg.norm(L)
    s = np.clip(nt @ L, 0, 1) * 0.9 + 0.05
    s = np.where(ok, s, 0.0)
    g = np.where(s <= 0.0031308, 12.92 * s, 1.055 * s ** (1 / 2.4) - 0.055)
    return Image.fromarray((g * 255).astype(np.uint8)).convert("RGB")


def coarse():
    n, ok = measure.gt_cam_normals(V, PX)
    e = ((n + 1) / 2 * 255 + 0.5).clip(0, 255).astype(np.uint8); e[~ok] = 0
    return Image.fromarray(e)


def gen():
    os.makedirs(OUT, exist_ok=True)
    coarse().save(f"{OUT}/coarse.png")
    for k, L in LIGHTS.items():
        lit(L).save(f"{OUT}/lit_{k}.png")
    one = PROMPT.format(s="", extra="")
    three = PROMPT.format(s="s (the same statue under three different lights: read the "
                          "surface shape from how the light falls)", extra="")
    jobs = [(one, f"{OUT}/flat.png", [f"{GT}/rgb/{V}.png", f"{OUT}/coarse.png"]),
            (PROMPT.format(s=" (lit by one raking light: read the surface shape from the shading)", extra=""),
             f"{OUT}/lit1.png", [f"{OUT}/lit_ul.png", f"{OUT}/coarse.png"]),
            (three, f"{OUT}/lit3.png", [f"{OUT}/lit_{k}.png" for k in LIGHTS] + [f"{OUT}/coarse.png"])]
    jobs = [j for j in jobs if not os.path.exists(j[1])]
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda j: print(f"wrote {j[1]} ({generate(*j):.0f}s)", flush=True), jobs))


def score():
    S = 1024
    nt, ok = measure.gt_cam_normals(V, GT)
    nt = np.stack([np.asarray(Image.fromarray(nt[..., c].astype(np.float32)).resize((S, S), Image.BILINEAR))
                   for c in range(3)], -1)
    nt /= np.linalg.norm(nt, axis=-1, keepdims=True).clip(1e-9)
    pc, pok = measure.gt_cam_normals(V, PX)
    m = pok & (np.linalg.norm(nt, axis=-1) > 0.5)
    from scipy import ndimage
    mm = ndimage.binary_erosion(m, iterations=3)
    det = lambda x: (bands.blur_n(x, m, 1) - bands.blur_n(x, m, 8))[mm][:, :2].ravel()

    def comb(n):
        o = bands.blur_n(pc, m, 8) + n - bands.blur_n(n, m, 8)
        return o / np.linalg.norm(o, axis=-1, keepdims=True).clip(1e-9)
    e = bands.ang(pc[mm], nt[mm])
    print(f"{'proxy2 alone':22} vs truth {np.median(e):5.1f}deg  detail r {np.corrcoef(det(pc), det(nt))[0,1]:.2f}")
    for k in ("flat", "lit1", "lit3"):
        p = f"{OUT}/{k}.png"
        if not os.path.exists(p):
            continue
        g = measure.load(p, S)
        gm = measure.mask_from(g, (0, 0, 0), tol=24)
        _, s, ty, tx = measure.align(gm, pok)
        g = measure.warp(g, s, ty, tx, order=1)
        n = g / 255 * 2 - 1; n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-9)
        c = comb(n)
        e = bands.ang(c[mm], nt[mm])
        print(f"{k:22} vs truth {np.median(e):5.1f}deg  detail r {np.corrcoef(det(n), det(nt))[0,1]:.2f}"
              f"  (proxy low + this detail)")


if __name__ == "__main__":
    {"gen": gen, "score": score}[sys.argv[1]]()
