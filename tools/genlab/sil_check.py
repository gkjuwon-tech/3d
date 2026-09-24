#!/usr/bin/env python3
"""Silhouette checks for an orthographic turnaround.

Orthographic views from opposite directions have mirror-image silhouettes, and
every view of one object shares its vertical extent (the top and the bottom of
the figure sit on the same rows). Both are checkable without knowing the shape.

    python3 tools/genlab/sil_check.py data/cat/gen/panel_front.png data/cat/gen/sheet1_a.png ...
"""
import sys
import os

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path[:0] = [os.path.dirname(__file__)]
import measure  # noqa: E402
from turnaround import panels, P  # noqa: E402


def sil(img):
    g = np.asarray(img.convert("RGB"), np.float64)
    m = measure.mask_from(g, (128, 128, 128), tol=16)
    m[:, :20] = False; m[:, -20:] = False           # the tick marks
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    return lab == 1 + np.argmax(ndimage.sum(m, lab, range(1, n + 1)))


def extent(m):
    ys, xs = np.nonzero(m)
    return ys.min(), ys.max(), xs.min(), xs.max()


def main():
    ref = sil(Image.open(sys.argv[1]))
    t0, b0, _, _ = extent(ref)
    print(f"reference front: rows {t0}-{b0}")
    for p in sys.argv[2:]:
        l, r = [sil(x) for x in panels(p)]
        tl, bl, _, _ = extent(l); tr, br, _, _ = extent(r)
        mir = r[:, ::-1]
        v, s, ty, tx = measure.align(mir, l)
        print(f"{os.path.basename(p)}: left vs reference IoU {measure.iou(l, ref):.3f} rows {tl}-{bl};"
              f" right rows {tr}-{br};  mirrored right vs left IoU raw {measure.iou(mir, l):.3f},"
              f" best-aligned {v:.3f} (scale {s:.3f} shift {ty:+.0f},{tx:+.0f})")


if __name__ == "__main__":
    main()
