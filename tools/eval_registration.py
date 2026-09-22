#!/usr/bin/env python3
"""Score a multi-view model's camera solution against the true view directions.

Per-view depth can look reasonable while the views are stacked in one place, so
depth error alone does not say whether a multi-view model understood the view
configuration. This measures that directly.

The predicted cameras live in their own frame, so the comparison is made after
the single global rotation that best aligns the predicted view axes to the true
ones. What survives that alignment is real: a view the model placed on the
wrong side of the object cannot be rotated into agreement with the others.

Given several subsets of the same view set, it prints the error per view as a
series across subsets, which is what shows whether adding views helps.

Run:
  python3 tools/eval_registration.py --dir out/da3ctl --views refs/lucy_gt
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def best_rotation(pred, true):
    """Kabsch: the rotation carrying predicted axes onto true ones."""
    U, _, Vt = np.linalg.svd(pred.T @ true)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def forward_axes(extrinsics):
    """World-space view direction per camera, from world-to-camera [R|t]."""
    R = np.asarray(extrinsics, dtype=np.float64)[:, :3, :3]
    return np.transpose(R, (0, 2, 1))[..., :, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="directory holding da3_<subset>_extrinsics.npy")
    ap.add_argument("--views", required=True)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    man = json.load(open(args.manifest or os.path.join(args.dir,
                                                       "manifest.json")))
    used = man.get("notes", {}).get("da3_subsets", {})

    files = sorted(glob.glob(os.path.join(args.dir, "da3_*_extrinsics.npy")))
    if not files:
        raise SystemExit(f"no da3_*_extrinsics.npy under {args.dir}")

    results = {}
    for f in files:
        subset = re.search(r"da3_(.+)_extrinsics\.npy$", os.path.basename(f))[1]
        views = used.get(subset)
        if not views:
            print(f"skip {subset}: no view list in the manifest")
            continue
        E = np.load(f)
        if len(E) != len(views):
            print(f"skip {subset}: {len(E)} cameras for {len(views)} views")
            continue
        pred = forward_axes(E)
        true = np.array([-np.array(meta["views"][v]["location"])
                         / np.linalg.norm(meta["views"][v]["location"])
                         for v in views])
        aligned = pred @ best_rotation(pred, true).T
        err = np.degrees(np.arccos(
            np.clip(np.einsum("ij,ij->i", aligned, true), -1, 1)))
        results[subset] = dict(zip(views, err.tolist()))

    order = sorted(results, key=lambda s: len(results[s]))
    all_views = [v for v in meta["views"]
                 if any(v in r for r in results.values())]

    head = "".join(f"{s} ({len(results[s])})".rjust(14) for s in order)
    print(f"\nregistration error in degrees, after the best global rotation\n")
    print(f"{'view':<14}{head}")
    for v in all_views:
        row = "".join(
            (f"{results[s][v]:.1f}" if v in results[s] else "-").rjust(14)
            for s in order)
        flag = "   <-- pole" if v in ("05_top", "06_bottom") else ""
        print(f"{v:<14}{row}{flag}")

    print(f"\n{'mean, canonical six':<14}"
          + "".join(
              f"{np.mean([e for k, e in results[s].items() if not (k.endswith('_up') or k.endswith('_dn'))]):.1f}".rjust(14)
              for s in order))

    for s in order:
        r = results[s]
        if "05_top" in r and "06_bottom" in r:
            print(f"{'top vs bottom':<14}"
                  if s == order[0] else "", end="")
    print()
    print("a view the model put on the wrong side of the object cannot be "
          "rotated\ninto agreement, so a large error here is a real "
          "registration failure")

    out = os.path.join(args.dir, "registration_report.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
