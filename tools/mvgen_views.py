#!/usr/bin/env python3
"""Multi-view generated images -> a stage2 view set.

MV-Adapter draws orthographic views (frustum +-0.55, i.e. ortho_scale 1.1 --
the scale stage2's renders use) at the azimuths and elevations the kernel
asked for (tools/kaggle_mvgen.py writes them to frame.json; the Hugging Face
Space draws six level ones at 0, 45, 90, 180, 270, 315). This writes data/<name>/views/{rgb,mask,
cameras.json}. MV-Adapter places azimuth a at (cos(a-90), sin(a-90), 0) with z up,
so a = 0 is the front camera (-y) and a = 90 the +x one: ours at 270 + a.
Silhouettes alone cannot tell (a mirrored object agrees just as well), so the
convention is taken from its code, and the agreement is only reported.

  views   --src data/catmv14/mvgen --out data/catmv14
  merge   --src data/cat8/mvgen --src2 data/cat8b/mvgen --out data/cat8m/mvgen
  normals --src data/catmv/normals_raw --out data/catmv   (sign-fixed from the silhouette)
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from PIL import Image
from scipy import ndimage

AZ = [0, 45, 90, 180, 270, 315]
NAME = {0: "01_front", 45: "d045", 90: "d090", 180: "03_back", 270: "d270", 315: "d315"}


def camera(az, el=0.0):
    """orthographic camera at azimuth az (ours: 270 = front, 0 = the +x side)"""
    d = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                  np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
    right = np.cross([0, 0, 1], d); right /= np.linalg.norm(right)
    up = np.cross(d, right)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, d, 2 * d
    return M


def mask_of(img, tol=7.0, grad=3.0, keep_frac=0.01):
    """background = flat, background-coloured and connected to the border. The
    clay's shadowed hems are background-grey too, but never flat, so a colour
    test alone floods into them; requiring smoothness stops it at the edge"""
    a = np.asarray(img.convert("RGB"), np.float64)
    bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), 0)
    g = a.mean(-1)
    sm = ndimage.gaussian_filter(g, 1.0)
    gm = np.hypot(ndimage.sobel(sm, 0), ndimage.sobel(sm, 1)) / 8
    near = (np.abs(a - bg).max(-1) < tol) & (ndimage.maximum_filter(gm, 3) < grad)
    lab, _ = ndimage.label(near)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    m = ndimage.binary_opening(ndimage.binary_fill_holes(~np.isin(lab, edge[edge > 0])), iterations=1)
    lab, n = ndimage.label(m)
    if not n:
        return m
    # every piece of real size, not only the largest: a flame or a raised
    # hand can be drawn clear of the body in one view and touching it in the
    # next, and dropping it from one view carves it out of all of them
    size = ndimage.sum(m, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.nonzero(size >= keep_frac * size.max())[0])


def agreement(masks, mats, o, n=192):
    """share of each silhouette kept by the strict visual hull of all of them"""
    h = o / 2
    c = (np.arange(n) + 0.5) / n * 2 * h - h
    X = np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)
    res = next(iter(masks.values())).shape[0]

    def proj(M, P):
        r = ((-(P @ M[:3, 1]) / o + 0.5) * res).astype(int).clip(0, res - 1)
        q = (((P @ M[:3, 0]) / o + 0.5) * res).astype(int).clip(0, res - 1)
        return r, q
    occ = np.ones(len(X), bool)
    for v, m in masks.items():
        md = ndimage.binary_dilation(m, iterations=2)
        r, q = proj(mats[v], X); occ &= md[r, q]
    out = {}
    for v, m in masks.items():
        s = np.zeros_like(m); r, q = proj(mats[v], X[occ]); s[r, q] = True
        s = ndimage.binary_dilation(s, iterations=int(np.ceil(o / n / o * res)))
        out[v] = (m & s).sum() / m.sum()
    return out


def layout(src):
    """[(name, file stem, MV-Adapter azimuth, elevation)]: from the kernel's
    frame.json, else the Hugging Face Space's six views, saved by index"""
    fj = os.path.join(src, "frame.json")
    if os.path.exists(fj):
        return [(v[0], v[0], v[1], v[2]) for v in json.load(open(fj))["views"]]
    return [(NAME[z], str(i), z, 0.0) for i, z in enumerate(AZ)]


def views(a):
    files = glob.glob(f"{a.src}/view_s*_*.png")
    seeds = sorted({int(re.search(r"view_s(\d+)_", f).group(1)) for f in files})
    s = seeds[0] if a.seed is None else a.seed
    lay = layout(a.src)
    imgs = {n: Image.open(f"{a.src}/view_s{s}_{stem}.png").convert("RGB") for n, stem, _, _ in lay}
    masks = {n: mask_of(im) for n, im in imgs.items()}
    # MV-Adapter's camera for (az, el) sits where ours for (270 + az, el) does,
    # with the same right and up vectors (both look-at with z up)
    mats = {n: camera((270 + az) % 360, el) for n, _, az, el in lay}
    ag = agreement(masks, mats, 1.1)
    print("silhouette agreement (share kept by the strict hull): "
          + " ".join(f"{v} {x:.3f}" for v, x in ag.items()))
    for d in ("views/rgb", "views/mask"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    res = next(iter(masks.values())).shape[0]
    json.dump({"ortho_scale": 1.1, "resolution": [res, res],
               "views": {v: {"matrix_world": M.tolist()} for v, M in mats.items()}},
              open(f"{a.out}/views/cameras.json", "w"), indent=1)
    for n, im in imgs.items():
        im.save(f"{a.out}/views/rgb/{n}.png")
        Image.fromarray((masks[n] * 255).astype(np.uint8)).save(f"{a.out}/views/mask/{n}.png")


def merge(a):
    """Two MV-Adapter passes -> one ring of eight level views.

    MV-Adapter i2mv draws the six azimuths it was trained on (0, 45, 90, 180,
    270, 315) consistently -- 98-99% silhouette agreement on the cat -- but
    135 and 225 are off its training set, and asking for them dropped every
    view to 82-95%. So the back diagonals come from a second pass whose
    reference image is the first pass's back view: its 45 and 315 are the
    ring's 225 and 135. The second pass frames its reference afresh (crops it
    and scales it to 90% of the canvas), so it is mapped back with one scale
    and one 3D shift, fitted on the four views both passes drew (the bounding
    boxes of their silhouettes), and the fit is reported as IoU.
    """
    la, lb = layout(a.src), layout(a.src2)
    trained = {0, 45, 90, 180, 270, 315}
    A = {az % 360: (n, st) for n, st, az, el in la if el == 0 and az % 360 in trained}
    B = {int(round(az + a.offset)) % 360: (n, st) for n, st, az, el in lb
         if el == 0 and az % 360 in trained}

    def load(src, st):
        im = Image.open(f"{src}/view_s0_{st}.png").convert("RGB")
        return im, mask_of(im)
    ma = {z: load(a.src, st) for z, (n, st) in A.items()}
    mb = {z: load(a.src2, st) for z, (n, st) in B.items()}
    both = sorted(set(A) & set(B))
    new = sorted(set(B) - set(A))
    res = next(iter(ma.values()))[1].shape[0]
    c = (res - 1) / 2

    def box(m):
        ys, xs = np.nonzero(m)
        return ys.min(), ys.max(), xs.min(), xs.max()
    # scale and vertical shift from the heights, shared by every view
    sc, vs = [], []
    for z in both:
        ta, ba, _, _ = box(ma[z][1]); tb, bb, _, _ = box(mb[z][1])
        s_ = (bb - tb) / max(ba - ta, 1)
        sc.append(s_); vs.append((tb + bb) / 2 - (s_ * ((ta + ba) / 2 - c) + c))
    s_ = float(np.median(sc)); v_ = float(np.median(vs))
    # horizontal shift per view is a 3D shift seen along that view's right axis
    rows, rhs = [], []
    for z in both:
        _, _, la_, ra = box(ma[z][1]); _, _, lb_, rb = box(mb[z][1])
        h = (lb_ + rb) / 2 - (s_ * ((la_ + ra) / 2 - c) + c)
        R = camera((270 + z) % 360)[:3, 0]
        rows.append(R[:2]); rhs.append(h)
    txy, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    print(f"second pass: scale {s_:.4f}, vertical {v_:+.1f} px, shift {txy.round(1)} px")

    def to_a(img, z, order):
        """resample a second-pass view into the first pass's frame"""
        h = float(camera((270 + z) % 360)[:3, 0][:2] @ txy)
        arr = np.asarray(img, np.float64)
        rr, cc = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
        src_r = s_ * (rr - c) + c + v_
        src_c = s_ * (cc - c) + c + h
        chans = arr[..., None] if arr.ndim == 2 else arr
        out = np.stack([ndimage.map_coordinates(chans[..., k], [src_r, src_c], order=order,
                                                mode="nearest") for k in range(chans.shape[2])], -1)
        return out[..., 0] if arr.ndim == 2 else out
    for z in both:
        w = to_a(mb[z][1].astype(np.float64), z, 1) > 0.5
        iou = (w & ma[z][1]).sum() / (w | ma[z][1]).sum()
        print(f"  az {z:3d}: IoU after mapping {iou:.3f}")
    os.makedirs(a.out, exist_ok=True)
    views = []
    for z in sorted(set(A) | set(B)):
        if z in A:
            n, _ = A[z]; im, _ = ma[z]
        else:
            n = f"d{z:03d}"
            im = Image.fromarray(to_a(mb[z][0], z, 1).clip(0, 255).astype(np.uint8))
        im.save(f"{a.out}/view_s0_{n}.png")
        views.append([n, z, 0])
    json.dump({"views": views, "merged_from": [a.src, a.src2]},
              open(f"{a.out}/frame.json", "w"), indent=1)
    print("merged", [v[0] for v in views], "->", a.out)


def calibrate(n, m):
    """axis signs that make the rim normals point outward and the body face the camera"""
    d_in = ndimage.distance_transform_edt(m)
    rim = m & (d_in <= 2)
    g = np.stack(np.gradient(ndimage.gaussian_filter(m.astype(float), 2)), -1)
    sx = np.sign(np.sum(n[..., 0][rim] * -g[..., 1][rim])) or 1
    sy = np.sign(np.sum(n[..., 1][rim] * g[..., 0][rim])) or 1
    sz = np.sign(np.mean(n[..., 2][m & (d_in > 10)])) or 1
    return sx, sy, sz


def normals(a):
    os.makedirs(f"{a.out}/normals", exist_ok=True)
    for p in sorted(glob.glob(f"{a.out}/views/mask/*.png")):
        v = os.path.basename(p)[:-4]
        m = np.asarray(Image.open(p).convert("L")) > 127
        n = np.load(f"{a.src}/{v}.npy").astype(np.float64)
        if n.shape[:2] != m.shape:
            n = np.stack([np.asarray(Image.fromarray(n[..., c].astype(np.float32)).resize(
                m.shape[::-1], Image.BILINEAR)) for c in range(3)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-9)
        sx, sy, sz = calibrate(n, m)
        n *= np.array([sx, sy, sz]); n[~m] = 0
        np.save(f"{a.out}/normals/{v}.npy", n.astype(np.float32))
        print(f"{v}: axis signs {sx:+.0f} {sy:+.0f} {sz:+.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["views", "normals", "merge"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--src2", help="merge: the second pass (reference = first pass's back view)")
    ap.add_argument("--offset", type=float, default=180.0,
                    help="merge: azimuth of the second pass's reference in the first pass")
    a = ap.parse_args()
    {"views": views, "normals": normals, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()
