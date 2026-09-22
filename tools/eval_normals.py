#!/usr/bin/env python3
"""Score estimated surface normals against the true ones, in degrees.

Normals matter more than depth in this pipeline: they carry no scale or shift
ambiguity, so nothing has to be solved before they can be used, and they hold
high-frequency surface detail that a depth map smooths away. This measures
whether that theoretical advantage survives contact with an actual estimator on
orthographic renders, which are out of distribution for models trained on
photographs.

Run:
  python3 tools/eval_normals.py --est out/est --views refs/lucy_gt --out out/m2
"""
import argparse
import json
import os

import numpy as np

BG = 1e9


def to_camera_space(n_world, matrix_world):
    """World normals -> camera space, matching the convention the estimators
    predict in: +x right, +y up, +z toward the viewer."""
    m = np.array(matrix_world, dtype=np.float64)
    R = m[:3, :3]
    return n_world @ R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flip", default="auto",
                    help="auto, none, or a sign triple like +,-,+ to match the "
                         "estimator's axis convention")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    manifest = json.load(open(os.path.join(args.est, "manifest.json")))
    os.makedirs(args.out, exist_ok=True)

    models = [k for k, v in manifest["models"].items() if v["kind"] == "normals"]
    print("normal models:", models)

    report = {}
    for model in models:
        rows = {}
        signs_used = None
        for view, info in meta["views"].items():
            gt_w = np.load(os.path.join(args.views, "normal_npy", f"{view}.npy"))
            depth = np.load(os.path.join(args.views, "depth_npy", f"{view}.npy"))
            est = np.load(os.path.join(args.est, model, f"{view}.npy")
                          ).astype(np.float64)
            hit = depth < BG

            gt = to_camera_space(gt_w.reshape(-1, 3), info["matrix_world"])
            gt = gt.reshape(gt_w.shape)
            gt /= np.linalg.norm(gt, axis=2, keepdims=True).clip(1e-9)
            est /= np.linalg.norm(est, axis=2, keepdims=True).clip(1e-9)

            if args.flip == "auto" and signs_used is None:
                # Conventions differ between estimators by per-axis sign; pick
                # the one that fits, once, on the first view, and keep it.
                best, signs_used = None, (1, 1, 1)
                for sx in (1, -1):
                    for sy in (1, -1):
                        for sz in (1, -1):
                            e = est * np.array([sx, sy, sz])
                            c = np.clip((e[hit] * gt[hit]).sum(1), -1, 1)
                            mu = np.degrees(np.arccos(c)).mean()
                            if best is None or mu < best:
                                best, signs_used = mu, (sx, sy, sz)
                print(f"  {model}: axis signs {signs_used} "
                      f"(mean {best:.2f} deg on {view})")
            elif args.flip not in ("auto", "none"):
                signs_used = tuple(1 if s.strip() == "+" else -1
                                   for s in args.flip.split(","))
            if signs_used:
                est = est * np.array(signs_used)

            cos = np.clip((est[hit] * gt[hit]).sum(1), -1, 1)
            ang = np.degrees(np.arccos(cos))
            # Baseline: assume every visible surface faces the camera. An
            # estimate that does not beat this is not carrying information
            # about the surface, whatever its output looks like.
            flat = np.zeros_like(gt[hit])
            flat[:, 2] = 1.0
            ang_flat = np.degrees(np.arccos(
                np.clip((flat * gt[hit]).sum(1), -1, 1)))
            rows[view] = {
                "mean_deg": float(ang.mean()),
                "median_deg": float(np.median(ang)),
                "p95_deg": float(np.percentile(ang, 95)),
                "within_11_25": float((ang < 11.25).mean()),
                "within_22_5": float((ang < 22.5).mean()),
                "within_30": float((ang < 30.0).mean()),
                "flat_baseline_deg": float(ang_flat.mean()),
                "beats_baseline": bool(ang.mean() < ang_flat.mean()),
            }
        report[model] = {"per_view": rows, "axis_signs": list(signs_used)}

        print(f"\n=== {model}")
        print(f"{'view':<11}{'mean':>8}{'median':>9}{'<22.5':>8}"
              f"{'FLAT BASE':>11}   verdict")
        for v, r in rows.items():
            print(f"{v:<11}{r['mean_deg']:>8.2f}{r['median_deg']:>9.2f}"
                  f"{100*r['within_22_5']:>7.1f}%{r['flat_baseline_deg']:>11.2f}"
                  f"   {'estimator' if r['beats_baseline'] else 'FLAT GUESS WINS'}")
        mean = np.mean([r["mean_deg"] for r in rows.values()])
        med = np.mean([r["median_deg"] for r in rows.values()])
        print(f"{'mean':<11}{mean:>8.2f}{med:>9.2f}")

    with open(os.path.join(args.out, "normal_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}/normal_report.json")


if __name__ == "__main__":
    main()
