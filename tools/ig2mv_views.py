#!/usr/bin/env python3
"""kaggle_ig2mv.py output -> a view set for kaggle_normals.py and mesh_fit.py.

The cameras come from the kernel (already in the mesh's frame, each with its
own ortho scale); the masks are cut from the generated images themselves, so
that where the painter drew the outline differently from the mesh the fit
sees it. A close-up's frame edge cuts through the body, and the rendered mesh
is cut by the same frame, so nothing special is needed there.

  python3 tools/ig2mv_views.py --src data/owlig3/ig2mv --out data/owlig3/views
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mvgen_views as mv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight", type=float, default=1.0)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.src, "cameras.json")))
    for d in ("rgb", "mask"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    views = {}
    for n, info in meta["views"].items():
        im = Image.open(os.path.join(a.src, "rgb", f"{n}.png")).convert("RGB")
        g = np.asarray(Image.open(os.path.join(a.src, "geom_mask", f"{n}.png"))) > 127
        # background colour from well outside the geometry: a close-up's
        # border is mostly object, and the border's median made the books
        # the background colour
        far = ~ndimage.binary_dilation(g, iterations=12)
        bg = np.median(np.asarray(im, np.float64)[far], 0) if far.sum() > 2000 else None
        m = mv.mask_of(im, bg=bg)
        iou = (m & g).sum() / max((m | g).sum(), 1)
        im.save(os.path.join(a.out, "rgb", f"{n}.png"))
        Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(a.out, "mask", f"{n}.png"))
        views[n] = {"matrix_world": info["matrix_world"], "ortho_scale": info["ortho_scale"],
                    "weight": a.weight}
        print(f"{n}: drawn outline vs mesh outline IoU {iou:.3f}")
    json.dump({"ortho_scale": meta["ortho_scale"], "resolution": meta["resolution"], "views": views},
              open(os.path.join(a.out, "cameras.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
