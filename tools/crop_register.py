#!/usr/bin/env python3
"""Close-up views of one region, placed in the main views' 3D frame.

A region the main views draw too small (a face 60 px across) is cropped from
the front view, enlarged, and given to MV-Adapter again: six new views of just
that region, drawn jointly, at 1.8x or more the resolution. This finds where
those close-up cameras are in the main frame, so tools/mesh_fit.py can use
them as extra views:

  - scale and the offset in the front view's plane are exact: the crop box and
    MV-Adapter's own framing (frame.json: its crop, scale and offset) compose
    to one similarity from main front pixels to close-up front pixels
  - the offset along the front camera's axis is the one unknown; it is found
    by sliding the close-up side views along it until their silhouettes, above
    the cut, overlap the main side views best
  - the crop's lower edge is a cut through the body, not an outline, so every
    close-up view gets a valid mask that stops a margin above that plane

  python3 tools/crop_register.py --main data/cat6 --crop data/cathead --out data/cathead
     (reads data/cathead/{crop.json, mvgen/}, writes data/cathead/views/)
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mvgen_views as mv  # noqa: E402


def project(W, ortho, res, X):
    p = X - W[:3, 3]
    col = ((p @ W[:3, 0]) / ortho + 0.5) * res - 0.5
    row = (0.5 - (p @ W[:3, 1]) / ortho) * res - 0.5
    return row, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="main data dir (views/)")
    ap.add_argument("--crop", required=True, help="close-up dir (crop.json, mvgen/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=12.0, help="px kept clear above the cut")
    ap.add_argument("--weight", type=float, default=2.0,
                    help="weight of the close-up views in the fit, relative to the main ones")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.main, "views", "cameras.json")))
    O = float(meta["ortho_scale"])
    R = meta["resolution"][0]
    crop = json.load(open(os.path.join(a.crop, "crop.json")))
    fr = json.load(open(os.path.join(a.crop, "mvgen", "frame.json")))
    r0, r1, c0, c1 = crop["box"]
    z = crop["size"] / (r1 - r0)
    s = fr["scale"]
    by0, _, bx0, _ = fr["crop_box"]
    oy, ox = fr["offset"]
    k = z * s
    # main front pixel p -> close-up front pixel g = k p + b (row and col)
    b_r = (-r0 * z - by0) * s + oy
    b_c = (-c0 * z - bx0) * s + ox
    Wf = np.array(meta["views"]["01_front"]["matrix_world"])
    res = crop["size"]
    oc = O / k                                            # close-up ortho scale
    # the main-front pixel under the close-up's centre, and the world offset
    # of that point in the front camera's plane
    pc_r = ((res - 1) / 2 - b_r) / k
    pc_c = ((res - 1) / 2 - b_c) / k
    dx = ((pc_c + 0.5) / R - 0.5) * O                     # along front right
    dz = (0.5 - (pc_r + 0.5) / R) * O                     # along front up
    t0 = dx * Wf[:3, 0] + dz * Wf[:3, 1]
    axis = Wf[:3, 2]                                      # the unknown direction
    z_cut = (0.5 - (r1 - 0.5) / R) * O                    # world height of the chest cut
    print(f"close-up: zoom {k:.3f}, ortho {oc:.4f}, in-plane offset {t0.round(4)}, cut at z {z_cut:.4f}")

    lay = mv.layout(os.path.join(a.crop, "mvgen"))
    cm = {n: mv.mask_of(Image.open(os.path.join(a.crop, "mvgen", f"view_s0_{st}.png")).convert("RGB"))
          for n, st, _, _ in lay}
    mm = {n: np.asarray(Image.open(os.path.join(a.main, "views", "mask", f"{n}.png"))) > 127
          for n in meta["views"]}

    def cams(t):
        out = {}
        for n, _, az, el in lay:
            W = np.array(meta["views"][n]["matrix_world"])
            W = W.copy()
            W[:3, 3] = W[:3, 3] + t                       # same direction, centred on t
            out[n] = W
        return out

    def valid(W):
        # rows whose level ray passes above the cut plane (all views level)
        rows = np.arange(res)
        zrow = W[:3, 3][2] + (0.5 - (rows + 0.5) / res) * oc
        v = np.zeros((res, res), bool)
        v[zrow > z_cut + a.margin * oc / res] = True
        return v

    def score(t):
        tot = 0.0
        W = cams(t)
        for n in cm:
            if n not in ("02_right", "04_left", "d045", "d315"):
                continue
            # main mask resampled into this close-up camera
            rr, cc = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
            Wc = W[n]
            # pixel -> world point on the camera plane, then -> main pixel
            px = (cc + 0.5) / res - 0.5
            py = 0.5 - (rr + 0.5) / res
            Xw = Wc[:3, 3] + px[..., None] * oc * Wc[:3, 0] + py[..., None] * oc * Wc[:3, 1]
            Wm = np.array(meta["views"][n]["matrix_world"])
            mr, mc = project(Wm, O, R, Xw.reshape(-1, 3))
            mres = ndimage.map_coordinates(mm[n].astype(float), [mr, mc], order=1, cval=0).reshape(res, res) > 0.5
            vm = valid(Wc)
            A, B = cm[n] & vm, mres & vm
            tot += (A & B).sum() / max((A | B).sum(), 1)
        return tot / 4

    best = max(((score(t0 + d * axis), d) for d in np.linspace(-0.15, 0.15, 61)))
    d0 = best[1]
    fine = max(((score(t0 + d * axis), d) for d in np.linspace(d0 - 0.006, d0 + 0.006, 25)))
    t = t0 + fine[1] * axis
    print(f"offset along the front axis {fine[1]:+.4f}: side-view IoU above the cut {fine[0]:.3f}")

    W = cams(t)
    for d in ("views/rgb", "views/mask", "views/valid"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    views = {}
    for n, st, _, _ in lay:
        name = f"crop_{n}"
        views[name] = {"matrix_world": W[n].tolist(), "ortho_scale": oc, "weight": a.weight}
        Image.open(os.path.join(a.crop, "mvgen", f"view_s0_{st}.png")).convert("RGB").save(
            os.path.join(a.out, "views", "rgb", f"{name}.png"))
        Image.fromarray((cm[n] * 255).astype(np.uint8)).save(os.path.join(a.out, "views", "mask", f"{name}.png"))
        Image.fromarray((valid(W[n]) * 255).astype(np.uint8)).save(os.path.join(a.out, "views", "valid", f"{name}.png"))
    json.dump({"ortho_scale": oc, "resolution": [res, res], "views": views, "crop_of": a.main,
               "cut_z": z_cut}, open(os.path.join(a.out, "views", "cameras.json"), "w"), indent=1)
    print("wrote", os.path.join(a.out, "views"))


if __name__ == "__main__":
    main()
