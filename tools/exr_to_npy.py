#!/usr/bin/env python3
"""Convert a view set's float EXR passes to .npy so the analysis tools can read
them without an EXR codec.

Blender is already the renderer, and it reads its own EXRs, so it does the
conversion rather than pulling in another image library.

Run:
  assets/blender/blender -b -P tools/exr_to_npy.py -- --dir refs/lucy_gt
"""
import argparse
import os
import sys

import bpy
import numpy as np


def load(path, channels):
    img = bpy.data.images.load(path)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)[::-1, :, :channels].squeeze()


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--passes", default="depth:1,normal:3")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.dir)
    for spec in args.passes.split(","):
        name, ch = spec.split(":")
        src = os.path.join(root, name)
        if not os.path.isdir(src):
            print(f"skip {name}: no such directory")
            continue
        dst = os.path.join(root, name + "_npy")
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(src)):
            if not f.endswith(".exr"):
                continue
            a = load(os.path.join(src, f), int(ch))
            out = os.path.join(dst, f[:-4] + ".npy")
            np.save(out, a.astype(np.float32))
            print(f"{name}/{f} -> {os.path.relpath(out, root)}  {a.shape}")


if __name__ == "__main__":
    main()
