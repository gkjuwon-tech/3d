"""Measure a generated image against Lucy's ground truth for one view.

mask_from(img, bg): object mask (background = flat colour connected to the border)
align(gen_mask, gt_mask): best scale + translation (similarity, no rotation)
"""
import json
import numpy as np
from PIL import Image
from scipy import ndimage


def load(p, size=None):
    im = Image.open(p).convert("RGB")
    if size and im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, np.float64)


def mask_from(img, bg, tol=18.0):
    d = np.abs(img - np.asarray(bg, np.float64)).max(-1)
    near = d < tol
    lab, _ = ndimage.label(near)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bgm = np.isin(lab, border[border > 0])
    return ndimage.binary_fill_holes(~bgm)


def warp(a, s, ty, tx, order=0):
    """a sampled so that output(p) = a((p - c - t) / s + c)"""
    H, W = a.shape[:2]
    c = np.array([(H - 1) / 2, (W - 1) / 2])
    mat = np.eye(2) / s
    off = c - (c + np.array([ty, tx])) / s
    if a.ndim == 2:
        return ndimage.affine_transform(a.astype(np.float64), mat, off, order=order)
    return np.stack([ndimage.affine_transform(a[..., k].astype(np.float64), mat, off,
                                              order=order) for k in range(a.shape[2])], -1)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def align(gen, gt):
    """scale + shift of gen onto gt maximising IoU (moments, then local search)"""
    def mom(m):
        ys, xs = np.nonzero(m); return ys.mean(), xs.mean(), np.sqrt(m.sum())
    gy, gx, gs = mom(gt); qy, qx, qs = mom(gen)
    s = gs / qs
    H, W = gt.shape
    c = np.array([(H - 1) / 2, (W - 1) / 2])
    ty = gy - (c[0] + (qy - c[0]) * s); tx = gx - (c[1] + (qx - c[1]) * s)
    best = (iou(warp(gen, s, ty, tx) > 0.5, gt), s, ty, tx)
    for step_s, step_t in [(0.02, 4), (0.005, 1), (0.001, 0.25)]:
        improved = True
        while improved:
            improved = False
            _, s0, y0, x0 = best
            for ds in (-step_s, 0, step_s):
                for dy in (-step_t, 0, step_t):
                    for dx in (-step_t, 0, step_t):
                        v = iou(warp(gen, s0 + ds, y0 + dy, x0 + dx) > 0.5, gt)
                        if v > best[0] + 1e-6:
                            best = (v, s0 + ds, y0 + dy, x0 + dx); improved = True
    return best


def boundary_dist(a, b):
    """distances (px) from a's boundary to b's boundary"""
    ea = a & ~ndimage.binary_erosion(a); eb = b & ~ndimage.binary_erosion(b)
    d = ndimage.distance_transform_edt(~eb)
    return d[ea]


def gt_cam_normals(view, views_dir="refs/lucy_gt"):
    meta = json.load(open(f"{views_dir}/cameras.json"))
    R = np.array(meta["views"][view]["matrix_world"])[:3, :3]
    n = np.load(f"{views_dir}/normal_npy/{view}.npy").astype(np.float64)
    ok = np.linalg.norm(n, axis=-1) > 0.5
    nc = (n.reshape(-1, 3) @ R).reshape(n.shape)
    nc /= np.linalg.norm(nc, axis=-1, keepdims=True).clip(1e-9)
    return nc, ok
