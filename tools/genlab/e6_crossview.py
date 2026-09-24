#!/usr/bin/env python3
"""E6: is generated surface detail the same surface in two views?

Two overlapping views, detailed normal maps generated from (image + coarse proxy
normals), (a) each view on its own, (b) both in one two-panel sheet. Then every
pixel of view A that view B also sees (ground-truth depth, z-test) is carried
into B, and A's generated world normal is compared with B's at that point.

    python3 tools/genlab/e6_crossview.py make|gen|score
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from gen_image import generate  # noqa: E402
import measure  # noqa: E402
import bands  # noqa: E402
from e2e5 import E5_PROMPT, coarse_normals  # noqa: E402

GT = "refs/lucy_gt"
OUT = "data/genlab/e6"
A, B = "01_front", "10_az315_up"
P, GUT = 1024, 64
RES = 1024

SHEET_PROMPT = """Make a detailed surface-normal sheet of the statue.

The first reference is a sheet of two panels: the same statue seen by two
orthographic cameras. The second reference is the matching sheet of coarse normal
maps, panel for panel: same layout, same positions, same outlines. Each panel's
normals are in that panel's own camera frame, standard encoding
R = (x+1)/2, G = (y+1)/2, B = (z+1)/2 with x right, y up, z toward that panel's
viewer, so surfaces facing the camera are light purple-blue (128,128,255). Black
background, white gutter with black registration squares.

Your output: the same two-panel normal sheet, same layout, same encoding, same
outlines and same large-scale colours as the coarse sheet, with all the fine
surface detail from the first sheet added as correct normal variation. The two
panels show ONE statue, so every fold, strand and feather must be the same surface
in both panels, just seen from the two cameras. No lighting, no shading. Keep the
sheet's wide 2:1 proportions."""


def two_panel(imgs):
    im = Image.new("RGB", (2 * P + GUT, P), (255, 255, 255))
    for i, x in enumerate(imgs):
        im.paste(x.convert("RGB").resize((P, P), Image.LANCZOS), (i * (P + GUT), 0))
    from PIL import ImageDraw
    d = ImageDraw.Draw(im); x0 = P + (GUT - 24) // 2
    for y in (8, P - 32):
        d.rectangle([x0, y, x0 + 23, y + 23], fill=(0, 0, 0))
    return im


def make():
    os.makedirs(OUT, exist_ok=True)
    for v in (A, B):
        coarse_normals(v).save(f"{OUT}/coarse_{v}.png")
    two_panel([Image.open(f"{GT}/rgb/{v}.png") for v in (A, B)]).save(f"{OUT}/sheet_rgb.png")
    two_panel([Image.open(f"{OUT}/coarse_{v}.png") for v in (A, B)]).save(f"{OUT}/sheet_coarse.png")


def gen():
    jobs = [(E5_PROMPT, f"{OUT}/solo_{v}.png", [f"{GT}/rgb/{v}.png", f"{OUT}/coarse_{v}.png"])
            for v in (A, B)]
    jobs.append((SHEET_PROMPT, f"{OUT}/sheet_gen.png", [f"{OUT}/sheet_rgb.png", f"{OUT}/sheet_coarse.png"]))
    jobs = [j for j in jobs if not os.path.exists(j[1])]

    def one(j):
        dt = generate(*j); print(f"wrote {j[1]} ({dt:.0f}s)", flush=True)
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(one, jobs))


def dec(img):
    a = np.asarray(img.convert("RGB").resize((RES, RES), Image.LANCZOS), np.float64) / 255 * 2 - 1
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9)


def panels(path):
    g = Image.open(path).convert("RGB").resize((2 * P + GUT, P), Image.LANCZOS)
    return [g.crop((i * (P + GUT), 0, i * (P + GUT) + P, P)) for i in range(2)]


def rs(x, order=Image.NEAREST):
    return np.asarray(Image.fromarray(x.astype(np.float32)).resize((RES, RES), order))


class View:
    def __init__(self, meta, v):
        m = np.array(meta["views"][v]["matrix_world"])
        self.right, self.up, self.back, self.loc, self.R = m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3], m[:3, :3]
        self.o = meta["ortho_scale"]
        d = np.load(f"{GT}/depth_npy/{v}.npy")
        self.depth = rs(np.where(d < 1e3, d, np.inf))
        nt, ok = measure.gt_cam_normals(v, GT)
        nt = np.stack([rs(nt[..., c], Image.BILINEAR) for c in range(3)], -1)
        self.gt = nt / np.linalg.norm(nt, axis=-1, keepdims=True).clip(1e-9)
        self.mask = np.isfinite(self.depth) & (rs(ok.astype(np.float32)) > 0.5)

    def points(self):
        c = ((np.arange(RES) + 0.5) / RES - 0.5) * self.o
        P0 = self.loc + c[None, :, None] * self.right + (-c[:, None, None]) * self.up
        return P0 + np.where(self.mask, self.depth, 0)[..., None] * (-self.back)

    def project(self, X):
        d = X - self.loc
        u, v, z = d @ self.right, d @ self.up, d @ (-self.back)
        col = (u / self.o + 0.5) * RES - 0.5; row = (-v / self.o + 0.5) * RES - 0.5
        return row, col, z

    def to_world(self, n):  # camera -> world
        return n @ self.R.T


def compare(va, vb, na, nb, sig=0):
    """A's normals vs B's at the same surface points (world frame)."""
    X = va.points()
    row, col, z = vb.project(X)
    ri, ci = np.round(row).astype(int), np.round(col).astype(int)
    inb = (ri >= 0) & (ri < RES) & (ci >= 0) & (ci < RES)
    ri, ci = ri.clip(0, RES - 1), ci.clip(0, RES - 1)
    vis = va.mask & inb & vb.mask[ri, ci] & (np.abs(vb.depth[ri, ci] - z) < 3 * va.o / RES)
    vis = ndimage.binary_erosion(vis, iterations=3)
    na_ = va.to_world(bands.blur_n(na, va.mask, sig))
    nbw = vb.to_world(bands.blur_n(nb, vb.mask, sig))
    nb_ = np.stack([ndimage.map_coordinates(nbw[..., k], [row, col], order=1) for k in range(3)], -1)
    nb_ /= np.linalg.norm(nb_, axis=-1, keepdims=True).clip(1e-9)
    e = bands.ang(na_[vis], nb_[vis])
    return e, vis


def score():
    meta = json.load(open(f"{GT}/cameras.json"))
    va, vb = View(meta, A), View(meta, B)
    sets = {"ground truth": (va.gt, vb.gt),
            "coarse proxy": (dec(Image.open(f"{OUT}/coarse_{A}.png")), dec(Image.open(f"{OUT}/coarse_{B}.png")))}
    if os.path.exists(f"{OUT}/solo_{A}.png") and os.path.exists(f"{OUT}/solo_{B}.png"):
        sets["generated, separately"] = (dec(Image.open(f"{OUT}/solo_{A}.png")), dec(Image.open(f"{OUT}/solo_{B}.png")))
    if os.path.exists(f"{OUT}/sheet_gen.png"):
        pa, pb = panels(f"{OUT}/sheet_gen.png")
        sets["generated, one sheet"] = (dec(pa), dec(pb))
    # image-space blur is not the same smoothing in two views (different
    # projections), so everything is compared unblurred; ground truth shows
    # the floor (resampling), the proxy what the coarse layer alone gives
    print(f"{A} vs {B}: angle between the two views' normals at shared surface points")
    X = va.points(); row, col, z = vb.project(X)
    ca, cb = sets["coarse proxy"]
    for name, (na, nb) in sets.items():
        e, vis = compare(va, vb, na, nb, 0)
        ea = bands.ang(na[va.mask], va.gt[va.mask]); eb = bands.ang(nb[vb.mask], vb.gt[vb.mask])
        # detail = departure from the proxy, carried into world frame
        da = va.to_world(na - ca); db = vb.to_world(nb - cb)
        db_ = np.stack([ndimage.map_coordinates(db[..., k], [row, col], order=1) for k in range(3)], -1)
        dt = va.to_world(va.gt - ca)
        r_ab = np.corrcoef(da[vis].ravel(), db_[vis].ravel())[0, 1] if name != "coarse proxy" else float("nan")
        r_at = np.corrcoef(da[vis].ravel(), dt[vis].ravel())[0, 1] if name != "coarse proxy" else float("nan")
        print(f"  {name:22} A<->B median {np.median(e):5.1f}deg  <5 {100*(e<5).mean():4.1f}%  "
              f"<10 {100*(e<10).mean():4.1f}%  | detail r A<->B {r_ab:.2f}  A<->truth {r_at:.2f}"
              f" | vs truth: A {np.median(ea):5.1f}  B {np.median(eb):5.1f}  (shared px {vis.sum()})")


if __name__ == "__main__":
    {"make": make, "gen": gen, "score": score}[sys.argv[1]]()
