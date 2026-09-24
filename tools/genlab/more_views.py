#!/usr/bin/env python3
"""More generated views, each held inside the current hull.

The visual hull contains the object, so the hull's silhouette from any new
camera is an outer bound on the true silhouette there. Each new two-panel sheet
(opposite cameras) is given those bounds as the panels to fill, plus the views
already made as the look to follow. The candidate that best fits its bounds is
kept, and its silhouette carves the hull further.

    python3 tools/genlab/more_views.py --views data/cat/views --hull data/cat/proxy/hull.npz \
        --gen data/cat/gen --refs data/cat/gen/sheet1_a.png data/cat/gen/sheet2_d.png
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from gen_image import generate  # noqa: E402
import measure  # noqa: E402
from carve import project, silhouette, grid  # noqa: E402
from sil_check import sil  # noqa: E402
from turnaround import P, GUT, BG, ticks, panels  # noqa: E402

# name -> (azimuth, elevation), azimuth 270 = the front camera, 0 = the right one
NEW = {"d315": (315, 0), "d135": (135, 0), "d045": (45, 0), "d225": (225, 0)}
SHEETS = [("d315", "d135"), ("d045", "d225")]
WORDS = {"d315": "orbited 45 degrees to the RIGHT of the front view (halfway between the front and the right-side view)",
         "d135": "orbited 135 degrees to the LEFT of the front view (halfway between the back and the left-side view)",
         "d045": "orbited 135 degrees to the RIGHT of the front view (halfway between the right-side and the back view)",
         "d225": "orbited 45 degrees to the LEFT of the front view (halfway between the front and the left-side view)"}

PROMPT = """A two-panel ORTHOGRAPHIC view sheet of the sculpture shown in the other
attached images (matte white clay, mid-grey #808080 background, even soft studio light,
no perspective). Those images show it from the front, back and both sides; it is one
rigid sculpture, so the new views must agree with all of them.

The first attached image is the sheet to fill. Each panel holds a light-grey shape: the
MAXIMUM outline the sculpture can have from that camera. Draw the sculpture so that it
fits inside that shape and touches its edges at the extremes (the top, the base, the
widest parts of the skirt, the flame, the tail), replacing the grey shape completely.

Left panel: the camera is {left}.
Right panel: the camera is {right}.

Same scale as the other views: the figure's top and the bottom of its base sit at the
rows marked by the tick marks. Keep the white gutter and the tick marks. Keep the wide
2:1 sheet."""


def camera(az, el):
    d = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                  np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
    right = np.cross([0, 0, 1], d); right /= np.linalg.norm(right)
    up = np.cross(d, right)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, d, 2 * d
    return M.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--hull", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--tries", type=int, default=2)
    a = ap.parse_args()
    meta = json.load(open(f"{a.views}/cameras.json"))
    for k, (az, el) in NEW.items():
        meta["views"].setdefault(k, {"matrix_world": camera(az, el)})
    json.dump(meta, open(f"{a.views}/cameras.json", "w"), indent=1)
    H = np.load(a.hull); occ = H["occ"]; h = float(H["hi"]); n = occ.shape[0]
    c, X = grid(n, h)
    pts = X[occ.ravel()]
    vox_px = 2 * h / n / meta["ortho_scale"] * P
    bound = {k: ndimage.binary_closing(silhouette(meta, k, pts, P, vox_px), iterations=2) for k in NEW}
    jobs = []
    for i, pair in enumerate(SHEETS):
        im = Image.new("RGB", (2 * P + GUT, P), (255, 255, 255))
        for j, k in enumerate(pair):
            pan = np.full((P, P, 3), BG, np.uint8); pan[bound[k]] = 200
            im.paste(Image.fromarray(pan), (j * (P + GUT), 0))
        d = ImageDraw.Draw(im)
        for j in range(2):
            ticks(d, j * (P + GUT))
        ref = f"{a.gen}/sheet{3+i}_ref.png"; im.save(ref)
        prompt = PROMPT.format(left=WORDS[pair[0]], right=WORDS[pair[1]])
        for t in "abcd"[:a.tries]:
            out = f"{a.gen}/sheet{3+i}_{t}.png"
            if not os.path.exists(out):
                jobs.append((prompt, out, [ref] + a.refs))
    with ThreadPoolExecutor(min(4, len(jobs)) or 1) as ex:
        list(ex.map(lambda j: print(f"wrote {j[1]} ({generate(*j):.0f}s)", flush=True), jobs))
    # keep, per sheet, the candidate whose panels best fit their bounds
    for i, pair in enumerate(SHEETS):
        best = None
        for t in "abcd"[:a.tries]:
            pans = panels(f"{a.gen}/sheet{3+i}_{t}.png")
            score, info = 0, []
            for k, p in zip(pair, pans):
                m = sil(p); b = bound[k]
                out_frac = (m & ~ndimage.binary_dilation(b, iterations=6)).sum() / m.sum()
                fill = (m & b).sum() / b.sum()
                score += fill - 3 * out_frac
                info.append(f"{k}: fills {100*fill:.0f}% of bound, {100*out_frac:.1f}% outside")
            print(f"sheet{3+i}_{t}: " + ";  ".join(info))
            if best is None or score > best[0]:
                best = (score, t, pans)
        print(f"  -> keep sheet{3+i}_{best[1]}")
        for k, p in zip(pair, best[2]):
            p.save(f"{a.views}/rgb/{k}.png")
            Image.fromarray((sil(p) * 255).astype(np.uint8)).save(f"{a.views}/mask/{k}.png")


if __name__ == "__main__":
    main()
