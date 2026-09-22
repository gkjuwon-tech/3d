#!/usr/bin/env python3
"""Find the edges a normal integration must not cross.

Between two adjacent pixels the normals describe a slope. Across an occlusion
boundary -- a fold in front of the body, a wing over a shoulder -- they describe
nothing at all: the two pixels lie on different surfaces, and the depth step
between them is unrelated to any local orientation. Measured on ground truth,
about 2% of edges are such steps, and excluding them moves the agreement
between normal-implied gradient and true gradient from 0.15 to 0.98.

Every detector here scores an *edge*, not a pixel, and looks at both of its
endpoints. A boundary sits between a grazing pixel and a frontal one, so a test
applied to the left pixel alone misses half of them by construction.

Run:
  python3 tools/discontinuity.py --views refs/lucy_gt --view 01_front
"""
import argparse
import json

import numpy as np
from PIL import Image
from scipy import ndimage

BG = 1e9


def edge_pairs(field, axis):
    """(a, b) views of a field along an axis, as the two ends of each edge."""
    if axis == 0:
        return field[:-1], field[1:]
    return field[:, :-1], field[:, 1:]


def pad_to(a, shape, axis):
    out = np.zeros(shape, dtype=a.dtype)
    if axis == 0:
        out[:-1] = a
    else:
        out[:, :-1] = a
    return out


def detectors(normals, rgb, hull, hit, axis):
    """Per-edge scores. Larger means more likely a discontinuity."""
    nz = np.abs(normals[..., 2])
    a, b = edge_pairs(nz, axis)
    graze = 1.0 / np.maximum(np.minimum(a, b), 1e-3)     # either end grazing

    na, nb = edge_pairs(normals, axis)
    turn = np.linalg.norm(na - nb, axis=-1)              # orientation snaps

    ga, gb = edge_pairs(rgb, axis)
    image = np.abs(ga - gb)                              # shading edge

    ha, hb = edge_pairs(np.where(np.isfinite(hull), hull, np.nan), axis)
    hull_jump = np.abs(ha - hb)

    ma, mb = edge_pairs(hit, axis)
    valid = ma & mb
    return {"graze": graze, "turn": turn, "image": image,
            "hull": np.nan_to_num(hull_jump)}, valid


def rank_normalise(score, valid):
    """Map a score onto [0, 1] by rank so detectors combine on equal terms."""
    out = np.zeros_like(score, dtype=np.float64)
    v = score[valid]
    order = np.argsort(v)
    r = np.empty(len(v))
    r[order] = np.arange(len(v)) / max(len(v) - 1, 1)
    out[valid] = r
    return out


def evaluate(views_dir, view, budgets, k_disc=8.0):
    meta = json.load(open(f"{views_dir}/cameras.json"))
    gt = np.load(f"{views_dir}/depth_npy/{view}.npy")
    hull = np.load(f"out/final_views/depth_npy/{view}.npy")
    n = np.load(f"{views_dir}/normal_npy/{view}.npy")
    R = np.array(meta["views"][view]["matrix_world"])[:3, :3]
    n = (n.reshape(-1, 3) @ R).reshape(n.shape)
    n /= np.linalg.norm(n, axis=2, keepdims=True).clip(1e-9)
    rgb = np.asarray(Image.open(f"{views_dir}/rgb/{view}.png").convert("L"),
                     dtype=np.float64) / 255.0
    hit = gt < BG
    core = ndimage.binary_erosion(hit, iterations=10)
    hull = np.where(hull < BG, hull, np.nan)

    results = {}
    for axis in (0, 1):
        g = np.where(hit, gt, np.nan)
        ga, gb = edge_pairs(g, axis)
        step = np.abs(ga - gb)
        det, valid = detectors(n, rgb, hull, hit, axis)
        ca, cb = edge_pairs(core, axis)
        valid = valid & ca & cb & np.isfinite(step)
        med = np.median(step[valid])
        truth = valid & (step > k_disc * med)
        results[axis] = (det, valid, truth)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--view", default="01_front")
    ap.add_argument("--budgets", default="1,3,5,8")
    args = ap.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    res = evaluate(args.views, args.view, budgets)

    names = ["graze", "turn", "image", "hull"]
    combos = [("graze",), ("turn",), ("image",), ("hull",),
              ("graze", "turn"), ("graze", "image"), ("turn", "image"),
              ("graze", "turn", "image"), ("graze", "turn", "image", "hull")]

    print(f"view {args.view}   discontinuity edges: ", end="")
    tot_t = sum(int(res[a][2].sum()) for a in (0, 1))
    tot_v = sum(int(res[a][1].sum()) for a in (0, 1))
    print(f"{tot_t:,} of {tot_v:,} ({100*tot_t/tot_v:.2f}%)\n")

    header = "".join(f"{int(b)}%".rjust(9) for b in budgets)
    print(f"{'detector':<28}{header}   <- recall at that edge budget")
    for combo in combos:
        recalls = []
        for b in budgets:
            hit_t = 0
            for axis in (0, 1):
                det, valid, truth = res[axis]
                s = np.zeros_like(det["graze"], dtype=np.float64)
                for k in combo:
                    s = np.maximum(s, rank_normalise(det[k], valid))
                thr = np.percentile(s[valid], 100 - b)
                cut = valid & (s >= thr)
                hit_t += int((cut & truth).sum())
            recalls.append(100 * hit_t / tot_t)
        print(f"{'+'.join(combo):<28}"
              + "".join(f"{r:8.1f}%" for r in recalls))


if __name__ == "__main__":
    main()
