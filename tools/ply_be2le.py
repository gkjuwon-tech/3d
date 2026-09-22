#!/usr/bin/env python3
"""Convert a big-endian binary PLY (e.g. the Stanford scans) to little-endian,
which is what Blender 4.x's PLY importer accepts. Reports the bounding box so
the render script knows which axis is up.

Usage:
  python3 tools/ply_be2le.py assets/lucy.ply assets/lucy_le.ply
"""
import sys
import numpy as np


def read_header(path):
    with open(path, "rb") as f:
        raw = f.read(4096)
    end = raw.find(b"end_header")
    if end < 0:
        sys.exit("end_header not found in first 4 KiB")
    offset = raw.find(b"\n", end) + 1
    lines = raw[:end].decode("ascii", "replace").splitlines()
    n_vert = n_face = 0
    fmt = None
    for ln in lines:
        p = ln.split()
        if not p:
            continue
        if p[0] == "format":
            fmt = p[1]
        elif p[0] == "element" and p[1] == "vertex":
            n_vert = int(p[2])
        elif p[0] == "element" and p[1] == "face":
            n_face = int(p[2])
    return fmt, n_vert, n_face, offset, lines


def main():
    src, dst = sys.argv[1], sys.argv[2]
    fmt, n_vert, n_face, offset, lines = read_header(src)
    print(f"source format : {fmt}")
    print(f"vertices      : {n_vert:,}")
    print(f"faces         : {n_face:,}")

    if "float" not in " ".join(lines) or n_vert == 0:
        sys.exit("unexpected header layout:\n" + "\n".join(lines))
    # Stanford scans: exactly x,y,z float per vertex, then uchar+3*int per face.
    n_props = sum(1 for ln in lines if ln.startswith("property float"))
    if n_props != 3:
        sys.exit(f"expected 3 float vertex properties, found {n_props}")

    be = fmt == "binary_big_endian"
    vdt = ">f4" if be else "<f4"
    fdt = np.dtype([("n", "u1"), ("v", (">i4" if be else "<i4"), 3)])

    with open(src, "rb") as f:
        f.seek(offset)
        verts = np.fromfile(f, dtype=vdt, count=n_vert * 3).reshape(n_vert, 3)
        faces = np.fromfile(f, dtype=fdt, count=n_face)

    if not np.all(faces["n"] == 3):
        sys.exit("mesh contains non-triangular faces")

    lo, hi = verts.min(0), verts.max(0)
    size = hi - lo
    print(f"bbox min      : {lo}")
    print(f"bbox max      : {hi}")
    print(f"bbox size     : {size}")
    print(f"tallest axis  : {'XYZ'[int(np.argmax(size))]}  "
          f"(aspect {size / size.max()})")

    v_out = verts.astype("<f4")
    f_out = np.empty(n_face, dtype=np.dtype([("n", "u1"), ("v", "<i4", 3)]))
    f_out["n"] = 3
    f_out["v"] = faces["v"].astype("<i4")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_vert}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {n_face}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    with open(dst, "wb") as f:
        f.write(header)
        v_out.tofile(f)
        f_out.tofile(f)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
