#!/usr/bin/env python3
"""Score a reconstruction against the mesh the views were rendered from.

Four numbers, each answering a different question:

  silhouette IoU   does the result explain the observations it was built from?
  containment      does it actually contain the true surface? A visual hull
                   that loses any of it has a bug, not an approximation error.
  Chamfer          how far is its surface from the true surface?
  volume ratio     how much of it is fat the silhouettes could never remove?

Run:
  python3 tools/eval_hull.py --gt-mesh assets/lucy_le.ply \
      --views refs/lucy_gt --recon out/lucy_hull.ply \
      --recon-views out/hull_views --occ out/lucy_hull_occ.npz
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

CHUNK = 4_000_000


def read_ply(path):
    with open(path, "rb") as f:
        head = f.read(4096)
        end = head.find(b"end_header")
        offset = head.find(b"\n", end) + 1
        lines = head[:end].decode("ascii", "replace").splitlines()
        n_vert = n_face = 0
        for ln in lines:
            p = ln.split()
            if len(p) >= 3 and p[0] == "element":
                if p[1] == "vertex":
                    n_vert = int(p[2])
                elif p[1] == "face":
                    n_face = int(p[2])
        f.seek(offset)
        verts = np.fromfile(f, dtype="<f4", count=n_vert * 3).reshape(n_vert, 3)
        fdt = np.dtype([("n", "u1"), ("v", "<i4", 3)])
        faces = np.fromfile(f, dtype=fdt, count=n_face)["v"]
    return verts, faces


def tri_areas(verts, faces):
    """Per-triangle area, chunked so a 28M-face mesh does not need 1 GB of
    gathered vertices at once."""
    out = np.empty(len(faces), dtype=np.float64)
    for i in range(0, len(faces), CHUNK):
        f = faces[i:i + CHUNK]
        a, b, c = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
        out[i:i + CHUNK] = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return out


def mesh_volume(verts, faces):
    """Signed volume by the divergence theorem; correct for a closed mesh."""
    total = 0.0
    for i in range(0, len(faces), CHUNK):
        f = faces[i:i + CHUNK]
        a, b, c = (verts[f[:, 0]].astype(np.float64),
                   verts[f[:, 1]].astype(np.float64),
                   verts[f[:, 2]].astype(np.float64))
        total += np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    return abs(total)


def sample_surface(verts, faces, n, rng):
    areas = tri_areas(verts, faces)
    cdf = np.cumsum(areas)
    idx = np.searchsorted(cdf, rng.random(n) * cdf[-1])
    idx = np.clip(idx, 0, len(faces) - 1)
    f = faces[idx]
    a, b, c = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    flip = (u + v) > 1
    u[flip] = 1 - u[flip]
    v[flip] = 1 - v[flip]
    return (a + u * (b - a) + v * (c - a)).astype(np.float64), areas.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-mesh", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--recon-views", default=None)
    ap.add_argument("--occ", default=None)
    ap.add_argument("--samples", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    norm = meta["normalization"]

    print("loading ground-truth mesh ...", flush=True)
    gv, gf = read_ply(args.gt_mesh)
    print(f"  {len(gv):,} verts / {len(gf):,} faces")

    # Reproduce the render's object transform: yaw about Z, then the uniform
    # scale and offset recorded when the view set was rendered.
    yaw = np.radians(meta.get("yaw_deg", 0.0))
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    gv = (gv.astype(np.float64) @ R.T) * norm["applied_scale"] \
        + np.array(norm["applied_offset"])
    gv = gv.astype(np.float32)

    print("loading reconstruction ...", flush=True)
    rv, rf = read_ply(args.recon)
    print(f"  {len(rv):,} verts / {len(rf):,} faces")

    print(f"sampling {args.samples:,} surface points on each ...", flush=True)
    gp, g_area = sample_surface(gv, gf, args.samples, rng)
    rp, r_area = sample_surface(rv, rf, args.samples, rng)

    print("building trees ...", flush=True)
    d_gr, _ = cKDTree(rp).query(gp, workers=-1)   # gt -> recon
    d_rg, _ = cKDTree(gp).query(rp, workers=-1)   # recon -> gt

    results = {}

    # --- silhouette IoU -----------------------------------------------------
    if args.recon_views:
        ious = {}
        skipped = []
        for view in meta["views"]:
            pa = os.path.join(args.views, "mask", f"{view}.png")
            pb = os.path.join(args.recon_views, "mask", f"{view}.png")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                # a reconstruction may have been rendered from fewer views than
                # the reference set holds; score the overlap rather than fail
                skipped.append(view)
                continue
            a = np.asarray(Image.open(pa).convert("L"))
            b = np.asarray(Image.open(pb).convert("L"))
            A, B = a > 127, b > 127
            inter = np.logical_and(A, B).sum()
            union = np.logical_or(A, B).sum()
            ious[view] = float(inter) / float(union)
        results["silhouette_iou"] = ious
        if skipped:
            results["silhouette_iou_skipped"] = skipped
            print(f"note: no reconstruction render for {len(skipped)} view(s): "
                  + ", ".join(skipped))

    # --- containment --------------------------------------------------------
    if args.occ:
        z = np.load(args.occ)
        dims = z["dims"]
        occ = np.unpackbits(z["occ"])[:int(np.prod(dims))].reshape(dims)
        lo, h = z["lo"], float(z["h"])
        idx = np.floor((gp - lo) / h).astype(np.int64)
        inside_grid = np.all((idx >= 0) & (idx < dims), axis=1)
        inside = np.zeros(len(gp), dtype=bool)
        ii = idx[inside_grid]
        inside[inside_grid] = occ[ii[:, 0], ii[:, 1], ii[:, 2]].astype(bool)
        results["containment"] = float(inside.mean())
        results["voxel_size"] = h

    # --- Chamfer ------------------------------------------------------------
    results["chamfer_mean"] = float((d_gr.mean() + d_rg.mean()) / 2)
    results["gt_to_recon_mean"] = float(d_gr.mean())
    results["gt_to_recon_p95"] = float(np.percentile(d_gr, 95))
    results["gt_to_recon_max"] = float(d_gr.max())
    results["recon_to_gt_mean"] = float(d_rg.mean())
    results["recon_to_gt_p95"] = float(np.percentile(d_rg, 95))
    results["recon_to_gt_max"] = float(d_rg.max())

    # --- volume -------------------------------------------------------------
    vg = mesh_volume(gv, gf)
    vr = mesh_volume(rv, rf)
    results["gt_volume"] = vg
    results["recon_volume"] = vr
    results["volume_ratio"] = vr / vg
    results["gt_area"] = float(g_area)
    results["recon_area"] = float(r_area)

    h = results.get("voxel_size")
    print("\n" + "=" * 62)
    if "silhouette_iou" in results:
        print("silhouette IoU (does it explain the views it was built from)")
        for v, i in results["silhouette_iou"].items():
            print(f"    {v:<12} {i:.5f}")
        print(f"    {'mean':<12} "
              f"{np.mean(list(results['silhouette_iou'].values())):.5f}")
    if "containment" in results:
        print(f"\ncontainment  : {results['containment']*100:.4f}% of the true "
              f"surface lies inside the hull")
    print(f"\nChamfer (normalized units, object height == 1.0)")
    print(f"    symmetric mean   {results['chamfer_mean']:.6f}"
          + (f"   ({results['chamfer_mean']/h:.2f} voxels)" if h else ""))
    print(f"    gt -> recon      mean {results['gt_to_recon_mean']:.6f}  "
          f"p95 {results['gt_to_recon_p95']:.6f}  "
          f"max {results['gt_to_recon_max']:.6f}")
    print(f"    recon -> gt      mean {results['recon_to_gt_mean']:.6f}  "
          f"p95 {results['recon_to_gt_p95']:.6f}  "
          f"max {results['recon_to_gt_max']:.6f}")
    print(f"\nvolume       : gt {vg:.6f}   recon {vr:.6f}   "
          f"ratio {results['volume_ratio']:.3f}x")
    print("=" * 62)

    out = os.path.splitext(args.recon)[0] + "_eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
