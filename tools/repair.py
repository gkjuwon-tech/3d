#!/usr/bin/env python3
"""Close the holes a carve punches through a plate, and drop what it sheds.

The k=2 carve thins Lucy's wings correctly -- one of them comes out as the
blade it should be, where the hull had a slab. The other tears open, because a
blade only has to be over-carved by a voxel or two from both sides at once to
stop being a surface at all. The two failures are not symmetric in cost: a
wing that is a voxel too fat reads as a wing, a wing with a hole in it reads
as garbage.

A hole is small and a concavity is not, so a morphological closing separates
them. Dilating by r and eroding back fills any gap narrower than 2r, leaving
the armpits and the gap between the legs -- which are far wider -- carved.
The closing is then intersected with the hull, so the repair can never put
back a voxel the hull never had; containment is preserved by construction.

An opening afterwards removes the mirror-image artefact, the spikes and specks
left standing where a view carved around them, and a connected-component pass
drops whatever is still floating free of the statue.

All of the morphology here uses a box, which is separable: three 1-D passes
per operation instead of one pass over an (2r+1)^3 neighbourhood.
"""
import argparse
import sys

import numpy as np
from scipy.ndimage import label, maximum_filter1d, minimum_filter1d

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from carve_depth import finish


def box(vol, r, op):
    f = maximum_filter1d if op == "dilate" else minimum_filter1d
    out = vol
    for axis in range(3):
        out = f(out, 2 * r + 1, axis=axis, mode="constant", cval=0)
    return out


def closing(occ, r):
    return box(box(occ.view(np.uint8), r, "dilate"), r, "erode").astype(bool)


def opening(occ, r):
    return box(box(occ.view(np.uint8), r, "erode"), r, "dilate").astype(bool)


def largest_component(occ):
    lab, n = label(occ)
    if n <= 1:
        return occ, n, 0
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = int(counts.argmax())
    dropped = int(occ.sum() - counts[keep])
    return lab == keep, n, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carved", required=True, help="occ npz from carve_depth")
    ap.add_argument("--hull", required=True, help="occ npz the carve started from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--close", type=int, default=3,
                    help="fill gaps narrower than 2r voxels")
    ap.add_argument("--open", dest="open_r", type=int, default=1,
                    help="remove spurs thinner than 2r voxels")
    ap.add_argument("--no-component", action="store_true")
    args = ap.parse_args()

    def load(path):
        z = np.load(path)
        dims = z["dims"]
        occ = np.unpackbits(z["occ"])[:int(np.prod(dims))].reshape(dims)
        return occ.astype(bool), dims, z["lo"], float(z["h"])

    occ, dims, lo, h = load(args.carved)
    hull, hdims, _, _ = load(args.hull)
    if tuple(hdims) != tuple(dims):
        raise SystemExit(f"grid mismatch: carved {tuple(dims)} hull {tuple(hdims)}")
    start = int(occ.sum())
    print(f"grid {tuple(dims)}  carved {start:,}  hull {int(hull.sum()):,}")

    if args.close > 0:
        occ = closing(occ, args.close) & hull
        print(f"  close r={args.close}: +{occ.sum() - start:,} voxels refilled")
    if args.open_r > 0:
        before = int(occ.sum())
        occ = opening(occ, args.open_r)
        print(f"  open  r={args.open_r}: {occ.sum() - before:+,} voxels")
    if not args.no_component:
        before = int(occ.sum())
        occ, n, dropped = largest_component(occ)
        print(f"  components: {n}, dropped {before - int(occ.sum()):,} "
              f"voxels of floating debris")

    finish(occ, dims, lo, h, start, args.out)


if __name__ == "__main__":
    main()
