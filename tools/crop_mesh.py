#!/usr/bin/env python3
"""Cut a sphere out of a mesh, for close-up renders.

Loading the 28M-face scan into Blender to look at a face 50 voxels across is
most of a minute and several gigabytes; the triangles within a few radii of
the focus are a few hundred thousand. With --views the input is taken to be
the raw scan and moved into the render frame first, so ground truth and
reconstruction crop to the same place.

Run:
  python3 tools/crop_mesh.py --mesh assets/lucy_le.ply --views refs/lucy_gt \
      --center 0.02,-0.096,0.32 --radius 0.12 --out /tmp/gt_face.ply
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_hull import read_ply  # noqa: E402
from visual_hull import write_ply  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--views", default=None,
                    help="cameras.json dir; move a raw scan into its frame")
    ap.add_argument("--center", required=True)
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    v, f = read_ply(args.mesh)
    if args.views:
        meta = json.load(open(os.path.join(args.views, "cameras.json")))
        norm = meta["normalization"]
        yaw = np.radians(meta.get("yaw_deg", 0.0))
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        v = ((v.astype(np.float64) @ R.T) * norm["applied_scale"]
             + np.array(norm["applied_offset"])).astype(np.float32)
    ctr = np.array([float(t) for t in args.center.split(",")])
    inside = np.linalg.norm(v - ctr, axis=1) < args.radius
    keep = inside[f].all(axis=1)
    f = f[keep]
    used = np.unique(f)
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    write_ply(args.out, v[used].astype(np.float32),
              remap[f].astype(np.int32))
    print(f"{args.out}: {len(used):,} verts / {len(f):,} faces")


if __name__ == "__main__":
    main()
