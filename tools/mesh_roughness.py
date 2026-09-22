#!/usr/bin/env python3
"""Measure how sharply a mesh's surface turns, edge by edge.

A visual hull carved from a binary voxel grid is covered in right-angle steps.
Those are not merely ugly: normal integration consumes surface gradient, and a
90 degree step is an infinite gradient, so a refinement driven by normals would
spend its budget chasing the grid instead of the object. The fraction of sharp
edges is therefore the number that decides whether smoothing did its job -- not
whether the render looks nice.

Run:
  python3 tools/mesh_roughness.py out/hull14_1024.ply out/final_mesh.ply
"""
import sys

import numpy as np


def read_ply(path):
    with open(path, "rb") as f:
        head = f.read(4096)
        end = head.find(b"end_header")
        off = head.find(b"\n", end) + 1
        nv = nf = 0
        for ln in head[:end].decode("ascii", "replace").splitlines():
            p = ln.split()
            if len(p) >= 3 and p[0] == "element":
                if p[1] == "vertex":
                    nv = int(p[2])
                elif p[1] == "face":
                    nf = int(p[2])
        f.seek(off)
        v = np.fromfile(f, dtype="<f4", count=nv * 3).reshape(nv, 3)
        fa = np.fromfile(f, dtype=np.dtype([("n", "u1"), ("v", "<i4", 3)]),
                         count=nf)["v"]
    return v, fa


def dihedrals(verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    # pair faces by shared edge: sort the edge list and take equal neighbours
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                faces[:, [2, 0]]]), axis=1)
    fi = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, fi = e[order], fi[order]
    same = (e[:-1] == e[1:]).all(1)
    i, j = fi[:-1][same], fi[1:][same]
    return np.degrees(np.arccos(np.clip((n[i] * n[j]).sum(1), -1, 1)))


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    print(f"{'mesh':<28}{'faces':>11}{'mean':>8}{'median':>9}"
          f"{'>30 deg':>10}{'>60 deg':>10}")
    for p in paths:
        v, f = read_ply(p)
        ang = dihedrals(v, f)
        print(f"{p.split('/')[-1]:<28}{len(f):>11,}{ang.mean():>7.2f}°"
              f"{np.median(ang):>8.2f}°{100*(ang>30).mean():>9.2f}%"
              f"{100*(ang>60).mean():>9.2f}%")


if __name__ == "__main__":
    main()
