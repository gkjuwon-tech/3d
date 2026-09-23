#!/usr/bin/env python3
"""Score the surface of a reconstruction, not just where it is.

eval_hull answers "is the surface in the right place" -- Chamfer distance,
containment, volume. It is blind to the two defects that dominate how the S2
mesh looks: terracing, one-voxel steps on a surface that is in the right
place on average, and crust, filaments standing a few voxels proud. Both are
nearly free in Chamfer and both wreck the surface normal. So this scores the
normal as well:

  accuracy / completeness   mean distance recon->GT and GT->recon
  F-score @ tau             harmonic mean of precision (recon points within
                            tau of GT) and recall (GT points within tau of
                            recon), at one and two voxels
  normal error              angle between each recon sample's face normal and
                            the normal of the nearest GT sample; median, mean,
                            and the fraction worse than 30 degrees

Every score is also reported per region -- the face, the chest, the skirt and
the plinth -- because a whole-model average lets a ruined face hide behind a
large, easy torso.

The ground truth is transformed into the render frame and sampled once, then
cached; loading the 28M-face scan is most of the cost otherwise.

Run:
  python3 tools/eval_surface.py --gt-mesh assets/lucy_le.ply \
      --views refs/lucy_gt --recon out/s2_mesh.ply
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_hull import read_ply, tri_areas  # noqa: E402

VOXEL = 1.0 / 1024

# name -> (centre, radius), in the normalised render frame. Located off the
# ground-truth front render and its depth map.
REGIONS = {
    "face":  ((0.020, -0.096, 0.320), 0.040),
    "chest": ((0.000, -0.122, 0.170), 0.060),
    "skirt": ((0.000, -0.100, -0.050), 0.080),
    "plinth": ((0.000, -0.094, -0.400), 0.080),
    # the thin, protruding places the holes are in (docs/KNOWN_ISSUES.md)
    "hand": ((-0.185, -0.086, 0.050), 0.035),
    "torch_hand": ((0.148, -0.157, 0.400), 0.035),
    "feet": ((0.010, -0.064, -0.370), 0.040),
    # hair: the head's sides and crown, around the face
    "hair_l": ((-0.028, -0.068, 0.330), 0.028),
    "hair_r": ((0.046, -0.068, 0.320), 0.028),
    "hair_top": ((0.010, -0.058, 0.360), 0.025),
}


def sample_with_normals(verts, faces, n, rng):
    areas = tri_areas(verts, faces)
    cdf = np.cumsum(areas)
    idx = np.clip(np.searchsorted(cdf, rng.random(n) * cdf[-1]),
                  0, len(faces) - 1)
    f = faces[idx]
    a = verts[f[:, 0]].astype(np.float64)
    b = verts[f[:, 1]].astype(np.float64)
    c = verts[f[:, 2]].astype(np.float64)
    u, v = rng.random((n, 1)), rng.random((n, 1))
    flip = (u + v) > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    p = a + u * (b - a) + v * (c - a)
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-20)
    return p, nrm


def gt_samples(gt_mesh, views, n, cache):
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if len(z["p"]) >= n:
            return z["p"][:n], z["n"][:n]
    meta = json.load(open(os.path.join(views, "cameras.json")))
    norm = meta["normalization"]
    gv, gf = read_ply(gt_mesh)
    yaw = np.radians(meta.get("yaw_deg", 0.0))
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    gv = ((gv.astype(np.float64) @ R.T) * norm["applied_scale"]
          + np.array(norm["applied_offset"])).astype(np.float32)
    p, nrm = sample_with_normals(gv, gf, n, np.random.default_rng(1))
    if cache:
        np.savez(cache, p=p.astype(np.float32), n=nrm.astype(np.float32))
    return p, nrm


def score(gp, gn, rp, rn, taus):
    d_rg, i_rg = cKDTree(gp).query(rp, workers=-1)
    d_gr, _ = cKDTree(rp).query(gp, workers=-1)
    cosang = np.abs(np.einsum("ij,ij->i", rn, gn[i_rg])).clip(0, 1)
    ang = np.degrees(np.arccos(cosang))
    out = {"accuracy": float(d_rg.mean()), "completeness": float(d_gr.mean()),
           "chamfer": float(0.5 * (d_rg.mean() + d_gr.mean())),
           "normal_median": float(np.median(ang)),
           "normal_mean": float(ang.mean()),
           "normal_bad30": float(100 * (ang > 30).mean())}
    for t in taus:
        prec = (d_rg < t * VOXEL).mean()
        rec = (d_gr < t * VOXEL).mean()
        out[f"F@{t:g}"] = float(100 * 2 * prec * rec / max(prec + rec, 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-mesh", default="assets/lucy_le.ply")
    ap.add_argument("--views", default="refs/lucy_gt")
    ap.add_argument("--recon", required=True, nargs="+")
    ap.add_argument("--samples", type=int, default=2_000_000)
    ap.add_argument("--cache", default="out/gt_samples.npz")
    ap.add_argument("--json", default=None)
    ap.add_argument("--regions", default="none", choices=["none", "lucy"],
                    help="the named regions are located on Lucy; other models "
                         "are scored whole")
    args = ap.parse_args()

    gp, gn = gt_samples(args.gt_mesh, args.views, args.samples, args.cache)
    taus = (1, 2)
    rows = {}
    for path in args.recon:
        rv, rf = read_ply(path)
        rp, rn = sample_with_normals(rv, rf, args.samples,
                                     np.random.default_rng(0))
        res = {"all": score(gp, gn, rp, rn, taus)}
        for name, (c, r) in (REGIONS.items() if args.regions == "lucy" else []):
            c = np.array(c)
            gm = np.linalg.norm(gp - c, axis=1) < r
            rm = np.linalg.norm(rp - c, axis=1) < r
            if gm.sum() > 100 and rm.sum() > 100:
                res[name] = score(gp[gm], gn[gm], rp[rm], rn[rm], taus)
        rows[path] = res

    keys = ["chamfer", "F@1", "F@2", "normal_median", "normal_mean",
            "normal_bad30"]
    for region in ["all"] + (list(REGIONS) if args.regions == "lucy" else []):
        print(f"\n[{region}]")
        print(f"  {'mesh':<28}" + "".join(f"{k:>14}" for k in keys))
        for path, res in rows.items():
            if region not in res:
                continue
            r = res[region]
            name = os.path.basename(path)[:28]
            print(f"  {name:<28}" + "".join(
                f"{r[k]:>14.6f}" if k == "chamfer" else f"{r[k]:>14.2f}"
                for k in keys))
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
