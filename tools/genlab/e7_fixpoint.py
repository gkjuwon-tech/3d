#!/usr/bin/env python3
"""E7: does a detailed normal map survive a pass through the model?

The consistency loop (generate -> fuse into one mesh -> render -> generate on
that render) only converges if the model keeps what the reference already has.
Reference = the ground-truth detailed normal map. (a) the detail prompt used for
coarse proxies (E5); (b) a plain "reproduce exactly" prompt.

    python3 tools/genlab/e7_fixpoint.py gen|score
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from gen_image import generate  # noqa: E402
import measure  # noqa: E402
import bands  # noqa: E402
from e2e5 import E5_PROMPT  # noqa: E402

GT, OUT, V = "refs/lucy_gt", "data/genlab/e7", "02_right"
COPY_PROMPT = """Reproduce the attached surface-normal map exactly: same encoding
(R,G,B = (x+1)/2,(y+1)/2,(z+1)/2), same outline, same position and size, every
fine detail kept where it is. Do not add, remove, sharpen, smooth or restyle
anything. Black background. Output a square image."""


def ref_path():
    p = f"{OUT}/ref_{V}.png"
    if not os.path.exists(p):
        os.makedirs(OUT, exist_ok=True)
        nt, ok = measure.gt_cam_normals(V, GT)
        enc = ((nt + 1) / 2 * 255 + 0.5).clip(0, 255).astype(np.uint8); enc[~ok] = 0
        Image.fromarray(enc).save(p)
    return p


def gen():
    r = ref_path()
    jobs = [(E5_PROMPT, f"{OUT}/detail_{V}.png", [f"{GT}/rgb/{V}.png", r]),
            (COPY_PROMPT, f"{OUT}/copy_{V}.png", [r])]
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda j: print(f"wrote {j[1]} ({generate(*j):.0f}s)", flush=True), jobs))


def score():
    S = 1024
    nt, ok = measure.gt_cam_normals(V, GT)
    rs = lambda x: np.stack([np.asarray(Image.fromarray(x[..., c].astype(np.float32)).resize(
        (S, S), Image.BILINEAR)) for c in range(3)], -1)
    nt = rs(nt); nt /= np.linalg.norm(nt, axis=-1, keepdims=True).clip(1e-9)
    g = measure.load(f"{GT}/rgb/{V}.png", S); m = measure.mask_from(g, g[0, 0])
    dec = lambda p: (lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9))(
        measure.load(p, S) / 255 * 2 - 1)
    bands.print_report("reference itself, 8-bit encoded", dec(ref_path()), nt, m)
    for k in ("detail", "copy"):
        p = f"{OUT}/{k}_{V}.png"
        if os.path.exists(p):
            bands.print_report(f"through the model ({k} prompt)", dec(p), nt, m)


if __name__ == "__main__":
    {"gen": gen, "score": score}[sys.argv[1]]()
