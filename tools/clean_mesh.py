#!/usr/bin/env python3
"""Drop the pieces of a mesh that float free of it.

Fusion leaves slivers wherever a view's depth was wrong on the far side of a
thin feature: disconnected shells, most of them a few dozen faces. They are
counted by connected component over shared vertices, and every component
smaller than `--min-frac` of the largest is removed.

Run:
  python3 tools/clean_mesh.py --mesh out/fused.ply --out out/fused_clean.ply
"""
import argparse
import os
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_hull import read_ply  # noqa: E402
from visual_hull import write_ply  # noqa: E402


def clean(v, f, min_frac):
    n = len(v)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    G = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    k, lab = connected_components(G, directed=False)
    flab = lab[f[:, 0]]
    size = np.bincount(flab, minlength=k)
    keep_c = size >= min_frac * size.max()
    keep = keep_c[flab]
    f2 = f[keep]
    used = np.unique(f2)
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[f2], k, int(keep_c.sum()), int((~keep).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-frac", type=float, default=0.001)
    a = ap.parse_args()
    v, f = read_ply(a.mesh)
    v2, f2, k, kept, dropped = clean(v, f.astype(np.int64), a.min_frac)
    write_ply(a.out, v2.astype(np.float32), f2.astype(np.int32))
    print(f"{a.out}: {k:,} components, kept {kept}, dropped {dropped:,} of "
          f"{len(f):,} faces")


if __name__ == "__main__":
    main()
