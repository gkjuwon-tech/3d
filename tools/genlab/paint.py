#!/usr/bin/env python3
"""Surface-painting consistency engine (docs/GEN_PROBES.md).

The image model is asked for per-view detailed normal maps, but no two views
are ever allowed to disagree: every surface point of the proxy belongs to
exactly one view (the one that sees it most squarely), and only that view's
generated detail is kept there. Views are painted in two-panel sheets, the two
panels facing opposite ways so they share no surface; each later sheet is shown
what earlier sheets already painted, and fills in around it.

After the last sheet, every view's normal map is rendered from the one painted
surface, so all eighteen agree by construction. Those maps go to stage2 in
place of photometric-stereo normals.

    python3 tools/genlab/paint.py --views data/genlab/proxy/views \
        --appearance refs/lucy_gt --out data/genlab/paint
    python3 tools/genlab/paint.py ... --finish     # render final normals only

--appearance: a view set whose rgb/ shows the detail to paint (for Lucy, the
ground-truth renders; for a generated object, generated appearance views).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
import bands  # noqa: E402
import measure  # noqa: E402

# opposite views share a sheet: they see disjoint surface, so the two panels
# never paint the same point. Ordered so the widest views come first.
SHEETS = [("01_front", "03_back"), ("02_right", "04_left"),
          ("10_az315_up", "12_az135_dn"), ("07_az45_up", "13_az225_dn"),
          ("08_az135_up", "14_az315_dn"), ("09_az225_up", "11_az45_dn"),
          ("05_top", "06_bottom"), ("15_az285_up45", "17_az180_up30"),
          ("16_az90_up30", "18_az345_up30")]

P, GUT = 1024, 64

PROMPT = """Make a detailed surface-normal sheet of the statue.

The first reference is a sheet of two panels: the statue seen by two orthographic
cameras. The second reference is the matching sheet of the statue's current
surface-normal maps, panel for panel: same layout, same positions, same outlines.
Each panel is in its own camera frame, standard encoding R = (x+1)/2,
G = (y+1)/2, B = (z+1)/2 with x right, y up, z toward that panel's viewer, so
surfaces facing the camera are light purple-blue (128,128,255). Black background,
white gutter with black registration squares.

Some areas of the normal maps already show fine sculpted detail: keep those
exactly as they are. Other areas are still smooth: add the fine surface detail
seen in the first sheet there (folds, hair strands, feathers, face, fingers), as
correct normal variation that continues the neighbouring detailed areas without a
visible seam.

Output the same two-panel normal sheet: same layout, same encoding, same outlines,
same large-scale colours. It must be a normal map, not a picture: no lighting, no
shadows. Keep the sheet's wide 2:1 proportions."""


class View:
    def __init__(self, views, meta, v, res):
        m = np.array(meta["views"][v]["matrix_world"])
        self.name = v
        self.right, self.up, self.back, self.loc = m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3]
        self.R = m[:3, :3]
        self.o = meta["ortho_scale"]
        self.res = res
        d = np.load(f"{views}/depth_npy/{v}.npy")
        n = np.load(f"{views}/normal_npy/{v}.npy").astype(np.float64)
        assert d.shape[0] == res, "render the proxy at the working resolution"
        self.mask = (d < 1e3) & (np.linalg.norm(n, axis=-1) > 0.5)
        self.depth = np.where(self.mask, d, np.inf)
        self.nw = n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-9)  # world
        c = ((np.arange(res) + 0.5) / res - 0.5) * self.o
        P0 = self.loc + c[None, :, None] * self.right + (-c[:, None, None]) * self.up
        self.X = P0 + np.where(self.mask, d, 0)[..., None] * (-self.back)

    def project(self, X):
        d = X - self.loc
        u, v, z = d @ self.right, d @ self.up, d @ (-self.back)
        return (-v / self.o + 0.5) * self.res - 0.5, (u / self.o + 0.5) * self.res - 0.5, z

    def sees(self, X, tol_px=2.0):
        """(row, col, visible) for world points X"""
        r, c, z = self.project(X)
        ri = np.round(r).astype(int); ci = np.round(c).astype(int)
        inb = (ri >= 0) & (ri < self.res) & (ci >= 0) & (ci < self.res)
        ri, ci = ri.clip(0, self.res - 1), ci.clip(0, self.res - 1)
        vis = inb & self.mask[ri, ci] & (np.abs(self.depth[ri, ci] - z) < tol_px * self.o / self.res)
        return r, c, vis

    def cam(self, nw):
        return nw @ self.R

    def world(self, nc):
        return nc @ self.R.T


def ownership(V, painters=None):
    """owner[v][p] = index (in names) of the painting view that sees pixel p's
    surface point most squarely (largest cosine between proxy normal and view
    direction). Only painters can own; every view in V gets an owner map."""
    names = list(V)
    painters = painters or names
    own = {}
    for v in names:
        X = V[v].X
        best = np.full(X.shape[:2], -2.0); arg = np.full(X.shape[:2], -1)
        for i, u in enumerate(names):
            if u not in painters:
                continue
            _, _, vis = V[u].sees(X)
            cos = V[v].nw @ V[u].back
            better = V[v].mask & vis & (cos > best)
            best[better] = cos[better]; arg[better] = i
        own[v] = arg
    return names, own


def sample(img, r, c, valid):
    """bilinear, but only from valid pixels (renormalised weights)"""
    out = np.zeros(r.shape + img.shape[2:])
    wsum = np.zeros(r.shape)
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    H, W = valid.shape
    for dr in (0, 1):
        for dc in (0, 1):
            rr, cc = (r0 + dr).clip(0, H - 1), (c0 + dc).clip(0, W - 1)
            w = (1 - np.abs(r - (r0 + dr))) * (1 - np.abs(c - (c0 + dc)))
            w = w * valid[rr, cc]
            out += w[..., None] * img[rr, cc]
            wsum += w
    return out / np.maximum(wsum, 1e-9)[..., None], wsum > 1e-6


def render_state(V, names, own, painted, v):
    """view v's world normals from the painted surface; proxy where unpainted.
    Returns (normals world, painted mask)"""
    view = V[v]
    out = view.nw.copy()
    done = np.zeros(view.mask.shape, bool)
    for i, u in enumerate(names):
        if u not in painted:
            continue
        sel = view.mask & (own[v] == i)
        if not sel.any():
            continue
        r, c, _ = V[u].sees(view.X[sel])
        val, ok = sample(painted[u], r, c, (own[u] == i) & (np.abs(painted[u]).sum(-1) > 0))
        n = val / np.linalg.norm(val, axis=-1, keepdims=True).clip(1e-9)
        idx = np.nonzero(sel)
        out[idx[0][ok], idx[1][ok]] = n[ok]
        done[idx[0][ok], idx[1][ok]] = True
    return out, done


LOOSE = """Make a detailed surface-normal sheet of the sculpture.

The first reference is a sheet of two panels: the sculpture seen by two orthographic
cameras. It defines the sculpture: its exact outline, its pose and every shape in it.
The second reference is the matching sheet of a ROUGH draft of the normals, panel for
panel, same framing: a blocky approximation that is only roughly right. Where the draft
already shows fine sculpted detail that matches the first sheet, keep it; everywhere
else ignore the draft's shapes and draw the true ones from the first sheet.

Each panel is in its own camera frame, standard encoding R = (x+1)/2, G = (y+1)/2,
B = (z+1)/2 with x right, y up, z toward that panel's viewer: surfaces facing the camera
are light purple-blue (128,128,255). Black background, white gutter with black
registration squares.

Output the two-panel normal sheet of the sculpture in the first sheet: exactly its
outline and position in each panel, its true forms (head, face, ears, arms, hands,
dress, folds, lace, tail, flames), same encoding. It must be a normal map, not a
picture: no lighting, no shadows. Keep the sheet's wide 2:1 proportions."""


STYLE = """

Any further attached images show the same sculpture from other cameras (its turnaround):
follow their sculpted details and style wherever the first sheet does not show them
clearly."""


def appearance(app_dir, view, nw):
    """the view's look: a generated image where one exists, otherwise the current
    surface shaded as matte clay (smooth where nothing is painted yet)"""
    p = f"{app_dir}/rgb/{view.name}.png"
    if os.path.exists(p):
        return Image.open(p)
    nc = view.cam(nw)
    L = np.array([-0.35, 0.45, 0.82]); L /= np.linalg.norm(L)
    s = 0.18 + 0.8 * np.clip(nc @ L, 0, 1)
    g = np.where(view.mask, 255 * s ** (1 / 2.2), 128).astype(np.uint8)
    return Image.fromarray(np.repeat(g[..., None], 3, -1))


def enc(nc, mask):
    e = ((nc + 1) / 2 * 255 + 0.5).clip(0, 255).astype(np.uint8)
    e[~mask] = 0
    return Image.fromarray(e)


def two_panel(imgs):
    im = Image.new("RGB", (2 * P + GUT, P), (255, 255, 255))
    for i, x in enumerate(imgs):
        im.paste(x.convert("RGB").resize((P, P), Image.LANCZOS), (i * (P + GUT), 0))
    d = ImageDraw.Draw(im); x0 = P + (GUT - 24) // 2
    for y in (8, P - 32):
        d.rectangle([x0, y, x0 + 23, y + 23], fill=(0, 0, 0))
    return im


def panels(img):
    g = img.convert("RGB").resize((2 * P + GUT, P), Image.LANCZOS)
    return [g.crop((i * (P + GUT), 0, i * (P + GUT) + P, P)) for i in range(2)]


def take(view, panel, ref_nc, sig, align=True):
    """generated panel -> camera normals, and the panel's own silhouette:
    registered to the proxy outline (align) or taken where the sheet put it,
    large scale from the reference (proxy / painted state), detail from the panel"""
    res = view.res
    g = np.asarray(panel.resize((res, res), Image.LANCZOS), np.float64)
    gm = measure.mask_from(g, (0, 0, 0), tol=24)
    if align:
        iou, s, ty, tx = measure.align(gm, view.mask)
    else:
        iou, s, ty, tx = measure.iou(gm, view.mask), 1.0, 0.0, 0.0
    g = measure.warp(g, s, ty, tx, order=1)
    gm = measure.warp(gm, s, ty, tx) > 0.5
    n = g / 255 * 2 - 1
    n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-9)
    m = view.mask
    out = bands.blur_n(ref_nc, m, sig) + (n - bands.blur_n(n, m, sig))
    out /= np.linalg.norm(out, axis=-1, keepdims=True).clip(1e-9)
    return out, gm, (iou, s, ty, tx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True, help="proxy view set (depth_npy, normal_npy)")
    ap.add_argument("--appearance", required=True, help="view set whose rgb/ shows the detail")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=8.0,
                    help="px: below this scale the proxy/painted state rules, above it the model")
    ap.add_argument("--sheets", type=int, default=len(SHEETS))
    ap.add_argument("--finish", action="store_true", help="only render the final normals")
    ap.add_argument("--pairs", default=None,
                    help="sheets as a:b,c:d,... (opposite views); default: the Lucy set")
    ap.add_argument("--painters", default=None,
                    help="comma list of views allowed to own surface (default: all sheet views)")
    ap.add_argument("--render", default=None,
                    help="comma list of views to write final normals for (default: sheet views)")
    ap.add_argument("--loose", action="store_true",
                    help="the proxy is only a rough bound: shape and outline come from the "
                         "appearance images, panels are not re-registered to the proxy outline, "
                         "and only pixels inside a painted silhouette count as surface")
    ap.add_argument("--style", nargs="*", default=[],
                    help="extra reference images showing the object's look (turnaround sheets)")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/sheets", exist_ok=True); os.makedirs(f"{a.out}/painted", exist_ok=True)
    meta = json.load(open(f"{a.views}/cameras.json"))
    sheets = [tuple(p.split(":")) for p in a.pairs.split(",")] if a.pairs else SHEETS
    need = [v for p in sheets for v in p] + (a.render.split(",") if a.render else [])
    V = {v: View(a.views, meta, v, a.res) for v in dict.fromkeys(need)}
    painters = a.painters.split(",") if a.painters else [v for p in sheets for v in p]
    t0 = time.time()
    names, own = ownership(V, painters)
    share = {v: np.bincount(own[v][V[v].mask] + 1, minlength=len(names) + 1)[1:] for v in names}
    print(f"ownership {time.time()-t0:.0f}s; own share: " + ", ".join(
        f"{v[:2]} {100*share[v][i]/max(V[v].mask.sum(),1):.0f}%" for i, v in enumerate(names)), flush=True)

    from gen_image import generate
    painted = {}
    for v in names:
        p = f"{a.out}/painted/{v}.npy"
        if os.path.exists(p):
            painted[v] = np.load(p)
    for k, pair in enumerate(sheets[:a.sheets]):
        if a.finish or all(v in painted for v in pair):
            continue
        state = {v: render_state(V, names, own, painted, v) for v in pair}
        refs = [f"{a.out}/sheets/{k:02d}_rgb.png", f"{a.out}/sheets/{k:02d}_state.png"]
        two_panel([appearance(a.appearance, V[v], state[v][0]) for v in pair]).save(refs[0])
        two_panel([enc(V[v].cam(state[v][0]), V[v].mask) for v in pair]).save(refs[1])
        gen_p = f"{a.out}/sheets/{k:02d}_gen.png"
        if not os.path.exists(gen_p):
            prompt = (LOOSE if a.loose else PROMPT) + (STYLE if a.style else "")
            dt = generate(prompt, gen_p, refs + list(a.style))
            print(f"sheet {k} {pair}: generated in {dt:.0f}s", flush=True)
        for v, panel in zip(pair, panels(Image.open(gen_p))):
            ref_nc = V[v].cam(state[v][0])
            nc, gm, reg = take(V[v], panel, ref_nc, a.sigma, align=not a.loose)
            i = names.index(v)
            mine = V[v].mask & (own[v] == i)
            if a.loose:
                mine &= gm
            nw = V[v].world(nc); nw[~mine] = 0
            painted[v] = nw.astype(np.float32)
            np.save(f"{a.out}/painted/{v}.npy", painted[v])
            print(f"  {v}: panel IoU {reg[0]:.3f} scale {reg[1]:.4f} shift {reg[2]:+.1f},{reg[3]:+.1f}  "
                  f"owns {100*mine.sum()/V[v].mask.sum():.0f}% of its pixels, "
                  f"{100*state[v][1][mine].mean() if mine.any() else 0:.0f}% of those already painted",
                  flush=True)

    os.makedirs(f"{a.out}/normals", exist_ok=True)
    cover = []
    for v in (a.render.split(",") if a.render else names):
        nw, done = render_state(V, names, own, painted, v)
        nc = V[v].cam(nw); nc[~V[v].mask] = 0
        if a.loose:
            nc[~done] = 0      # outside every painted silhouette: not surface
        np.save(f"{a.out}/normals/{v}.npy", nc.astype(np.float32))
        enc(nc, V[v].mask).save(f"{a.out}/normals/{v}.png")
        cover.append(done[V[v].mask].mean())
    print(f"final normals: painted coverage {100*np.mean(cover):.1f}% "
          f"(min {100*np.min(cover):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
