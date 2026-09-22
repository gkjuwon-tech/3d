#!/usr/bin/env python3
"""Hold the carve off structures the hull only barely resolves.

A quorum stops one bad view from deciding a voxel, but it does nothing about
where a mistake is expensive. Removing two voxels from the middle of the torso
is invisible; removing two from a wing membrane that is four voxels thick
punches a hole through it, and the Chamfer distance barely notices because a
membrane holds almost none of the surface samples. The picture notices.

So the carve is gated on local thickness. Erode the hull by T and dilate the
survivors back: whatever fails to come back has no core of its own -- it is a
plate or a filament, not a solid -- and is left exactly as the hull had it.
Everything with a core is carved on the quorum as before.

Erosion and dilation use a box, which is separable, so the cost is three
1-D passes per operation rather than a pass over a (2T+1)^3 neighbourhood.
"""
import argparse
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from carve_depth import finish


def box(vol, size, op):
    """Separable box erosion (min) or dilation (max), in place along axes."""
    f = minimum_filter1d if op == "erode" else maximum_filter1d
    out = vol
    for axis in range(3):
        out = f(out, size, axis=axis, mode="constant",
                cval=0 if op == "dilate" else 0)
    return out


def thin_mask(occ, t, slack):
    """True where the hull has no solid core within reach -- plates, filaments."""
    core = box(occ.view(np.uint8), 2 * t + 1, "erode")
    thick = box(core, 2 * (t + slack) + 1, "dilate").astype(bool)
    return occ & ~thick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ", required=True)
    ap.add_argument("--votes", required=True)
    ap.add_argument("--out")
    ap.add_argument("--quorum", type=int, default=2)
    ap.add_argument("--thickness", type=int, default=4,
                    help="half-thickness in voxels below which a region is "
                         "protected; a wing 2T voxels thick keeps its core")
    ap.add_argument("--slack", type=int, default=1,
                    help="extra dilation so a core covers its own surface")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    z = np.load(args.occ)
    dims = z["dims"]
    occ = np.unpackbits(z["occ"])[:int(np.prod(dims))].reshape(dims).astype(bool)
    lo, h = z["lo"], float(z["h"])
    start = int(occ.sum())

    votes = np.load(args.votes)["votes"]
    doomed = occ & (votes >= args.quorum)

    thin = thin_mask(occ, args.thickness, args.slack)
    spared = doomed & thin
    print(f"grid {tuple(dims)}  {start:,} voxels")
    print(f"  thin (T={args.thickness}, slack={args.slack}): {thin.sum():,} "
          f"({100*thin.sum()/start:.2f}% of the hull)")
    print(f"  quorum {args.quorum} would remove {doomed.sum():,}; "
          f"{spared.sum():,} of those are thin and are spared "
          f"({100*spared.sum()/max(doomed.sum(),1):.2f}%)")
    if args.report_only:
        return

    occ &= ~(doomed & ~thin)
    finish(occ, dims, lo, h, start, args.out)


if __name__ == "__main__":
    main()
