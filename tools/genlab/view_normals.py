#!/usr/bin/env python3
"""Generated clay views -> normal maps (one call per view).

References: the view itself (the shape to follow, pixel for pixel) and a key
sphere in the target encoding. Placement is then checked against the view's
silhouette and fixed by a similarity alignment.

    python3 tools/genlab/view_normals.py --views data/cat/views --out data/cat/vnormals
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from gen_image import generate  # noqa: E402
import measure  # noqa: E402

PROMPT = """Convert the sculpture in the first attached image into its surface-normal map.

Output the normal map of exactly that sculpture from exactly that camera: same outline,
same position and size in the frame, every fold, lace edge, fur tuft, flame curl and
facial feature where it is in the image. Orthographic camera.

Encoding (the second attached image is a key: a sphere in this encoding):
R = (x+1)/2, G = (y+1)/2, B = (z+1)/2 with x to the image right, y up, z toward the
viewer. Surfaces facing the camera are light purple-blue (128,128,255); facing right
pinkish-red; facing left blue-cyan; facing up light green; facing down purple.
Background pure black. It must be a normal map, not a picture: no lighting, no
shadows, no ambient occlusion. Output a square image."""


def key_sphere(size=512):
    v, u = np.mgrid[0:size, 0:size]
    u = (u + 0.5) / size * 2 - 1; v = -((v + 0.5) / size * 2 - 1)
    r2 = (u / 0.9) ** 2 + (v / 0.9) ** 2
    n = np.stack([u / 0.9, v / 0.9, np.sqrt(np.clip(1 - r2, 0, 1))], -1)
    e = ((n + 1) / 2 * 255 + 0.5).astype(np.uint8); e[r2 > 1] = 0
    return Image.fromarray(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    key = f"{a.out}/key_sphere.png"; key_sphere().save(key)
    meta = json.load(open(f"{a.views}/cameras.json"))
    views = [v for v in meta["views"] if os.path.exists(f"{a.views}/rgb/{v}.png")]
    jobs = [(PROMPT, f"{a.out}/raw_{v}.png", [f"{a.views}/rgb/{v}.png", key]) for v in views
            if not os.path.exists(f"{a.out}/raw_{v}.png")]
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda j: print(f"wrote {j[1]} ({generate(*j):.0f}s)", flush=True), jobs))
    for v in views:
        m = np.asarray(Image.open(f"{a.views}/mask/{v}.png").convert("L")) > 127
        res = m.shape[0]
        g = measure.load(f"{a.out}/raw_{v}.png", res)
        gm = measure.mask_from(g, (0, 0, 0), tol=24)
        iou0 = measure.iou(gm, m)
        iou, s, ty, tx = measure.align(gm, m)
        g = measure.warp(g, s, ty, tx, order=1)
        n = g / 255 * 2 - 1
        n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-9)
        n[~m] = 0
        np.save(f"{a.out}/{v}.npy", n.astype(np.float32))
        print(f"  {v:10} outline IoU {iou0:.3f} -> {iou:.3f} (scale {s:.3f} shift {ty:+.1f},{tx:+.1f})")


if __name__ == "__main__":
    main()
