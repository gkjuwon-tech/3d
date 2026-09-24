#!/usr/bin/env python3
"""E3: can the image model relight one view physically enough for photometric stereo?

For each of the 12 rig lights (camera space, photometric.light_set(12)) it asks
for the view re-rendered under that light alone, with two references: the view
itself (geometry) and a matte sphere lit by exactly that light (lighting guide).

    python3 tools/genlab/e3_relight.py gen   --view 02_right --out data/genlab/e3
    python3 tools/genlab/e3_relight.py solve --view 02_right --out data/genlab/e3
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from photometric import light_set  # noqa: E402
from gen_image import generate  # noqa: E402
import measure  # noqa: E402

PROMPT = """Re-render the sculpture in the first reference image under new lighting.

Keep EVERYTHING about the object and the camera identical to the first reference:
same statue, same pose, same pixel position and size in the frame, same outline,
same orthographic camera. Do not move, crop, zoom, rotate, or restyle anything.

Lighting: exactly ONE distant directional light, coming from the same direction as
the light on the sphere in the second reference image. Copy that sphere's lighting:
where its bright spot is, the object's surfaces facing that way are brightest;
surfaces facing away are pure black. No ambient light, no fill light, no rim light,
no bounce light, no sky. Cast shadows stay (the object shadows itself).

Material: perfectly matte white plaster (Lambertian), no specular highlights, no
texture, no colour. Background: pure black (#000000). Output a square image."""


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def sphere(L, size=512):
    v, u = np.mgrid[0:size, 0:size]
    u = (u + 0.5) / size * 2 - 1; v = (v + 0.5) / size * 2 - 1
    r2 = u * u + v * v
    # sphere radius 0.9 of the frame
    n = np.stack([u / 0.9, -v / 0.9, np.sqrt(np.clip(1 - r2 / 0.81, 0, 1))], -1)
    s = np.clip(n @ np.asarray(L), 0, 1) * (r2 < 0.81)
    return (srgb(s)[..., None] * 255 + 0.5).astype(np.uint8).repeat(3, -1)


def gen(a):
    lights = light_set(12)
    os.makedirs(a.out, exist_ok=True)
    json.dump(lights, open(f"{a.out}/lights.json", "w"), indent=1)
    ref = f"{a.views}/rgb/{a.view}.png"
    jobs = []
    for k, L in lights.items():
        sp = f"{a.out}/sphere_{k}.png"
        Image.fromarray(sphere(L)).save(sp)
        out = f"{a.out}/{a.view}_{k}.png"
        if not os.path.exists(out):
            jobs.append((k, sp, out))

    def one(j):
        k, sp, out = j
        try:
            dt = generate(PROMPT, out, [ref, sp])
            print(f"  {k}: {dt:.0f}s", flush=True)
        except Exception as e:  # keep the others going
            print(f"  {k}: FAILED {e}", flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(one, jobs))


def solve_ps(obs, L, near_deg=20.0, frac=0.02, floor=1e-3):
    """obs (N, K) linear intensities -> unit normals (N, 3), lit count"""
    litm = obs > np.maximum(frac * obs.max(1, keepdims=True), floor)
    near = L[:, 2] > np.cos(np.radians(near_deg))
    if near.any() and (~near).sum() >= 3:
        outer_ok = litm[:, ~near].sum(1) >= 3
        litm[np.ix_(outer_ok, near)] = False
    g = np.zeros((len(obs), 3))
    pats, inv = np.unique(litm, axis=0, return_inverse=True)
    inv = inv.ravel()
    for j, pat in enumerate(pats):
        if pat.sum() < 3:
            continue
        sel = inv == j
        g[sel] = np.linalg.lstsq(L[pat], obs[sel][:, pat].T, rcond=None)[0].T
    a = np.linalg.norm(g, axis=1)
    n = g / np.maximum(a, 1e-9)[:, None]
    n[a < 1e-6] = 0
    return n, litm.sum(1)


def fit_light(I, N, m):
    """oracle: the light (and albedo scale) that best explains image I on the
    true normals, from pixels that are clearly lit"""
    sel = m & (I > 0.05)
    Lg = np.linalg.lstsq(N[sel], I[sel], rcond=None)[0]
    for _ in range(5):  # refit without the attached-shadow side
        sel = m & (N @ Lg > 0.05) & (I > 0.02)
        Lg = np.linalg.lstsq(N[sel], I[sel], rcond=None)[0]
    return Lg


def ang(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def solve(a):
    lights = json.load(open(f"{a.out}/lights.json"))
    names = [k for k in lights if os.path.exists(f"{a.out}/{a.view}_{k}.png")]
    print(f"{len(names)} of {len(lights)} lights generated")
    S = a.size
    nt, okt = measure.gt_cam_normals(a.view, a.views)
    gt_img = measure.load(f"{a.views}/rgb/{a.view}.png", S)
    gt_m = measure.mask_from(gt_img, gt_img[0, 0])
    nt = np.stack([np.asarray(Image.fromarray(nt[..., c].astype(np.float32)).resize(
        (S, S), Image.NEAREST)) for c in range(3)], -1)
    obs, masks, rep = [], [], []
    for k in names:
        im = measure.load(f"{a.out}/{a.view}_{k}.png", S)
        g = lin(im.mean(-1) / 255.0)
        # registration: the lit image's own silhouette is unreliable (black on
        # black), so align on everything not near-black
        gm = ndimage_fill(g > 0.01)
        best = measure.align(gm, gt_m)
        v, s, ty, tx = best
        g = measure.warp(g, s, ty, tx, order=1)
        obs.append(g); masks.append(measure.warp(gm, s, ty, tx) > 0.5)
        rep.append((k, v, s, ty, tx))
    obs = np.stack(obs, -1)
    m = gt_m & (np.linalg.norm(nt, axis=-1) > 0.5)
    N = nt[m]; B = obs[m]
    Lnom = np.array([lights[k] for k in names]); Lnom /= np.linalg.norm(Lnom, axis=1, keepdims=True)
    print(f"{'light':>5} {'IoU':>6} {'scale':>6} {'shift':>12} {'fit vs nominal':>15} {'fit resid':>9} {'|L|':>6}")
    Lfit = []
    for i, (k, v, s, ty, tx) in enumerate(rep):
        Lg = fit_light(B[:, i], N, np.ones(len(N), bool))
        Lfit.append(Lg)
        pred = np.clip(N @ Lg, 0, None)
        lit = pred > 0.05
        res = np.median(np.abs(pred[lit] - B[lit, i])) / max(np.median(B[lit, i]), 1e-6)
        print(f"{k:>5} {v:6.3f} {s:6.3f} {ty:5.1f},{tx:5.1f}  "
              f"{ang(Lg / np.linalg.norm(Lg), Lnom[i]):11.1f} deg {res:8.2f}  {np.linalg.norm(Lg):.2f}")
    Lfit = np.array(Lfit)
    for label, L in [("nominal lights", Lnom), ("oracle-fit lights", Lfit)]:
        n, cnt = solve_ps(B.copy(), L)
        good = np.linalg.norm(n, axis=1) > 0.5
        e = ang(n[good], N[good])
        print(f"PS with {label:18}: median {np.median(e):5.1f} deg  mean {e.mean():5.1f}"
              f"  <5 {100*(e<5).mean():4.1f}%  <10 {100*(e<10).mean():4.1f}%  "
              f"<20 {100*(e<20).mean():4.1f}%  solved {100*good.mean():.0f}%")
        full = np.zeros((S, S, 3)); full[m] = n
        np.save(f"{a.out}/ps_{label.split()[0]}.npy", full.astype(np.float32))
    # look: generated lights montage + normals
    tiles = [np.asarray(Image.open(f"{a.out}/{a.view}_{k}.png").convert("RGB").resize((256, 256))) for k in names]
    sph = [np.asarray(Image.open(f"{a.out}/sphere_{k}.png").convert("RGB").resize((256, 256))) for k in names]
    rows = [np.concatenate(sph[i:i + 6], 1) for i in range(0, len(names), 6)]
    rows2 = [np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(names), 6)]
    if all(r.shape == rows[0].shape for r in rows + rows2):
        Image.fromarray(np.concatenate(sum(zip(rows, rows2), ()), 0)).save(f"{a.out}/look_lights.png")
    nv = lambda x: ((x * [1, 1, 1] + 1) / 2 * 255 * (np.linalg.norm(x, axis=-1, keepdims=True) > 0.5)).astype(np.uint8)
    ps = np.load(f"{a.out}/ps_oracle-fit.npy")
    Image.fromarray(np.concatenate([nv(nt * m[..., None]), nv(ps)], 1)).resize((1024, 512)).save(
        f"{a.out}/look_normals.png")


def ndimage_fill(m):
    from scipy import ndimage
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return ndimage.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "solve"])
    ap.add_argument("--view", default="02_right")
    ap.add_argument("--views", default="refs/lucy_gt")
    ap.add_argument("--out", default="data/genlab/e3")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--size", type=int, default=1024)
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else solve(a)


if __name__ == "__main__":
    main()
