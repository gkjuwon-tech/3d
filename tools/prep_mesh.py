#!/usr/bin/env python3
"""Bring any Stanford-style PLY into the form the renderer reads.

The scan repository ships every encoding the format allows -- ASCII with extra
per-vertex confidence and intensity, big-endian binary, faces with a leading
intensity byte -- and Blender 4.x reads none of the big-endian ones. This
parses with plyfile, keeps positions and triangles only, drops degenerate and
unreferenced geometry, turns the model so its up axis is +Z (the renderer's
convention), and writes little-endian binary.

Run:
  python3 tools/prep_mesh.py --src assets/stanford/dragon_recon/dragon_vrip.ply \
      --up y --out assets/dragon.ply
"""
import argparse
import os
import sys

import numpy as np
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visual_hull import write_ply  # noqa: E402

# rotations that carry the named source axis onto +Z
UP = {
    "z": np.eye(3),
    "y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    "-y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
    "x": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--up", default="z", choices=sorted(UP))
    args = ap.parse_args()

    ply = PlyData.read(args.src)
    vx = ply["vertex"]
    v = np.stack([vx["x"], vx["y"], vx["z"]], -1).astype(np.float64)
    fe = ply["face"].data
    key = "vertex_indices" if "vertex_indices" in fe.dtype.names else "vertex_index"
    polys = fe[key]
    tris = []
    for p in polys:                       # fan-triangulate anything larger
        p = np.asarray(p, dtype=np.int64)
        for i in range(1, len(p) - 1):
            tris.append((p[0], p[i], p[i + 1]))
    f = np.asarray(tris, dtype=np.int64)
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    used = np.unique(f)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    v = v[used] @ UP[args.up].T
    f = remap[f]
    write_ply(args.out, v.astype(np.float32), f.astype(np.int32))
    lo, hi = v.min(0), v.max(0)
    print(f"{args.out}: {len(v):,} verts / {len(f):,} faces  "
          f"size {np.round(hi - lo, 4).tolist()}  (z is up)")


if __name__ == "__main__":
    main()
