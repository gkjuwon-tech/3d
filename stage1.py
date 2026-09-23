#!/usr/bin/env python3
"""Stage 1 -- a mesh in, images out. The only stage that ever sees the mesh.

  data/<name>/views/   14 orthographic views (front, right, back, left, top,
                       bottom, and eight three-quarter views at +-45 degrees):
                       rgb + anti-aliased mask, plus the point-sampled depth and
                       normal passes kept for scoring only
  data/<name>/photo/   each view lit from twelve known directions (Lambertian,
                       with the shadows a real capture has); photo/lights.json
                       records them
  data/<name>/normals/ camera-space normals solved per pixel from the lights
                       that reach it; none where fewer than three do

Stage 2 reads views/{mask,rgb}, views/cameras.json and normals/ -- the
images and the cameras -- and nothing else.

Run:
  python3 stage1.py --mesh assets/bunny.ply --name bunny
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
BLENDER = os.path.join(ROOT, "assets", "blender", "blender")


def run(cmd, log):
    t0 = time.time()
    print(f"$ {' '.join(cmd)}", flush=True)
    with open(log, "a") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"failed ({r.returncode}); see {log}")
    print(f"  ok {time.time()-t0:.0f}s", flush=True)


def blender(script, args, log):
    run([BLENDER, "-b", "-P", os.path.join(ROOT, "tools", script), "--"] + args, log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=16,
                    help="beauty samples; only the anti-aliased mask edge "
                         "depends on it")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="degrees about +Z, to turn the model's front toward "
                         "the front camera; both renderers rotate first and "
                         "then re-centre on the bounding box, so any angle "
                         "keeps them aligned")
    ap.add_argument("--lights", type=int, default=12,
                    help="lights per view. Four left 17%% of a view with two "
                         "or fewer unshadowed next to Lucy's raised arm, and "
                         "the old solver returned garbage there; eight leave "
                         "under 1%% but still 30%% of the crevice under her "
                         "right ear; twelve (four more near the view axis, "
                         "see photometric.light_set) leave none")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    d = os.path.join(a.data, a.name)
    views, photo, normals = (os.path.join(d, x) for x in ("views", "photo", "normals"))
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, "stage1.log")
    mesh = os.path.abspath(a.mesh)
    t0 = time.time()

    if a.force or not os.path.exists(os.path.join(views, "cameras.json")):
        blender("render_orthoviews.py",
                ["--mesh", mesh, "--out", views, "--res", str(a.res),
                 "--samples", str(a.samples), "--aux", "diagonal8",
                 "--yaw", str(a.yaw)], log)
        blender("exr_to_npy.py", ["--dir", views], log)
    import json as _json
    lp = os.path.join(photo, "lights.json")
    have = len(_json.load(open(lp))) if os.path.exists(lp) else 0
    fresh_photo = False
    n_exr = len([f for f in os.listdir(photo) if f.endswith(".exr")]) \
        if os.path.isdir(photo) else 0
    if a.force or have != a.lights or n_exr < 14 * a.lights:
        fresh_photo = True
        blender("photometric.py",
                ["render", "--mesh", mesh, "--out", photo,
                 "--views-json", os.path.join(views, "cameras.json"),
                 "--res", str(a.res), "--yaw", str(a.yaw),
                 # a sun on a diffuse surface, direct light only, shades every
                 # sample alike: 4 undenoised samples score 0.82 deg mean error
                 # on the bunny's front against 0.74 for 24 denoised, in a
                 # seventh of the time
                 "--samples", "4", "--no-denoise", "--lights", str(a.lights)], log)
    if a.force or fresh_photo or not os.path.isdir(normals) or \
            len(os.listdir(normals)) < 14:
        blender("photometric.py",
                ["solve", "--dir", photo, "--out", normals,
                 "--views-json", os.path.join(views, "cameras.json"),
                 "--score-against", views], log)
        with open(log) as f:
            for line in f.read().splitlines():
                if "lights ->" in line:
                    print("   ", line.strip())
    print(f"stage 1 done: {d}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
