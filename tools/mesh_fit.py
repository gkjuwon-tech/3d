#!/usr/bin/env python3
"""One mesh, fitted to every view at once: silhouettes and normals.

stage2 builds a depth map per view and fuses them. That needs normals that
agree across views to under a degree (photometric stereo does); a learned
estimator is ~20 degrees off, differently in every view, and the per-view
depths then disagree by tens of voxels -- the fusion tears. Here there is only
ever one closed surface. It starts as the visual hull and is moved so that,
rendered from every camera, its outline matches the mask and its normals match
the normal map. Where two views disagree the surface settles between them
instead of tearing, and continuous remeshing (Palfinger 2022, as in
Unique3D's ISOMER) keeps the triangles even while it moves: coarse first, then
down to the detail the normals carry.

Losses per view, on the rendered image (nvdiffrast, orthographic):
  alpha   (rendered coverage - mask)^2 over the whole image
  normal  |rendered normal - target normal|^2 where both are the object
Every view's normals are rotated into world space first, so all views pull on
the same surface in the same frame.

Run (on a CUDA machine with nvdiffrast):
  python3 tools/mesh_fit.py --views data/cat6/views --normals data/cat6_nirne/normals \
      --out data/cat6_nirne/fit
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_views(views, normals, res):
    meta = json.load(open(os.path.join(views, "cameras.json")))
    names = list(meta["views"])
    M, masks, nrm, face = [], [], [], []
    for v in names:
        m = np.asarray(Image.open(os.path.join(views, "mask", f"{v}.png")).convert("L")
                       .resize((res, res), Image.BILINEAR), np.float32) / 255
        n = np.load(os.path.join(normals, f"{v}.npy")).astype(np.float32)
        if n.shape[0] != res:
            n = np.stack([np.asarray(Image.fromarray(n[..., c]).resize((res, res), Image.BILINEAR))
                          for c in range(3)], -1)
        W = np.array(meta["views"][v]["matrix_world"], np.float64)
        nw = n @ W[:3, :3].T.astype(np.float32)            # camera frame -> world
        nw /= np.linalg.norm(nw, axis=-1, keepdims=True).clip(1e-6)
        nw[m < 0.5] = 0
        fz = np.clip(n[..., 2], 0, 1)                     # how squarely it faces the camera
        fz[m < 0.5] = 0
        M.append(W); masks.append(m); nrm.append(nw); face.append(fz)
    return meta, names, np.stack(M), np.stack(masks), np.stack(nrm), np.stack(face)


def clip_matrices(M, ortho, near=0.1, far=4.0):
    """world -> clip for each orthographic camera (OpenGL conventions)"""
    h = ortho / 2
    P = np.array([[1 / h, 0, 0, 0], [0, 1 / h, 0, 0],
                  [0, 0, -2 / (far - near), -(far + near) / (far - near)], [0, 0, 0, 1]])
    return np.stack([P @ np.linalg.inv(m) for m in M]).astype(np.float32)


def carve_hull(M, masks, ortho, n):
    """visual hull as a mesh: carve an n^3 grid, marching cubes"""
    from skimage import measure
    from scipy import ndimage
    h = ortho / 2
    c = (np.arange(n) + 0.5) / n * 2 * h - h
    X = np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)
    occ = np.ones(len(X), bool)
    res = masks.shape[1]
    for W, m in zip(M, masks):
        md = ndimage.binary_dilation(m > 0.5, iterations=1)
        p = X - W[:3, 3]
        r = ((-(p @ W[:3, 1]) / ortho + 0.5) * res).astype(int).clip(0, res - 1)
        q = (((p @ W[:3, 0]) / ortho + 0.5) * res).astype(int).clip(0, res - 1)
        occ &= md[r, q]
    occ = occ.reshape(n, n, n).astype(np.float32)
    occ = ndimage.gaussian_filter(occ, 0.8)
    v, f, _, _ = measure.marching_cubes(np.pad(occ, 1), 0.5, spacing=(2 * h / n,) * 3)
    v = v - h - 2 * h / n + h / n
    return v.astype(np.float32), f[:, ::-1].copy().astype(np.int64)


def write_ply(path, v, f):
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n")
                 .encode())
        fh.write(np.asarray(v, "<f4").tobytes())
        rec = np.empty(len(f), dtype=[("n", "u1"), ("i", "<i4", (3,))])
        rec["n"] = 3
        rec["i"] = f
        fh.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--normals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=768, help="render and target resolution")
    ap.add_argument("--hull-res", type=int, default=128)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--edge-start", type=float, default=0.06)
    ap.add_argument("--edge-end", type=float, default=0.004)
    ap.add_argument("--edge-steps", type=float, default=0.6,
                    help="fraction of the steps over which edges shrink to --edge-end")
    ap.add_argument("--decay", type=float, default=0.995)
    ap.add_argument("--w-normal", type=float, default=1.0)
    ap.add_argument("--w-alpha", type=float, default=1.0)
    ap.add_argument("--w-expand", type=float, default=0.1)
    ap.add_argument("--laplacian", type=float, default=0.02)
    ap.add_argument("--select", type=float, default=8.0,
                    help="per triangle, weight view k by (cos_k / best cos)^s; 0 = all views equal")
    ap.add_argument("--blur", default="8,4,2,1,0",
                    help="gaussian sigmas (px) of the target normals, in equal stages of the run")
    ap.add_argument("--facing-pow", type=float, default=1.0,
                    help="weight each target normal by (its facing the camera)^p; 0 = uniform")
    ap.add_argument("--loss", default="l1", choices=["l1", "l2"],
                    help="normal loss per pixel: l1 (robust: a wrong normal pulls less) or l2")
    ap.add_argument("--snap-every", type=int, default=50, help="steps between progress renders")
    a = ap.parse_args()

    import torch
    import nvdiffrast.torch as dr
    from meshfit.opt import MeshOptimizer
    from meshfit.remesh import calc_vertex_normals

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    meta, names, M, masks, nrm, face = load_views(a.views, a.normals, a.res)
    ortho = float(meta["ortho_scale"])
    v0, f0 = carve_hull(M, masks, ortho, a.hull_res)
    write_ply(os.path.join(a.out, "init.ply"), v0, f0)
    print(f"{len(names)} views at {a.res}; hull {len(v0):,} vertices, {len(f0):,} faces "
          f"[{time.time()-t0:.0f}s]", flush=True)

    dev = "cuda"
    glctx = dr.RasterizeCudaContext(device=dev)
    mvp = torch.tensor(clip_matrices(M, ortho), device=dev)
    # nvdiffrast's first image row is the bottom one
    tgt_a = torch.tensor(masks[:, ::-1].copy(), device=dev)
    # coarse to fine in the targets too: the normals blurred at first, so the
    # masses settle before the surface chases pixel-level disagreement between
    # views (which it otherwise fits as chips and flakes)
    from scipy import ndimage
    def blurred(sig):
        if sig <= 0:
            return nrm
        out = np.empty_like(nrm)
        for k in range(len(nrm)):
            m = (masks[k] > 0.5).astype(np.float32)
            wsum = ndimage.gaussian_filter(m, sig)
            for c in range(3):
                out[k, ..., c] = ndimage.gaussian_filter(nrm[k, ..., c] * m, sig) / np.maximum(wsum, 1e-6)
            out[k] /= np.linalg.norm(out[k], axis=-1, keepdims=True).clip(1e-6)
            out[k][m < 0.5] = 0
        return out
    sigmas = [float(x) for x in a.blur.split(",")]
    levels = [torch.tensor(blurred(sg)[:, ::-1].copy(), device=dev) for sg in sigmas]
    tgt_n = levels[0]
    obj = tgt_a > 0.5
    # a learned normal is least reliable where the surface turns away from
    # the camera, and there every other view sees it better
    wgt = torch.tensor(face[:, ::-1].copy(), device=dev) ** a.facing_pow
    view_dir = torch.tensor(M[:, :3, 2], device=dev, dtype=torch.float32)   # toward each camera

    def render(v, n, f):
        vh = torch.cat([v, torch.ones_like(v[:, :1])], -1)
        clip = vh @ mvp.transpose(-2, -1)
        fi = f.int()
        rast, _ = dr.rasterize(glctx, clip, fi, resolution=[a.res, a.res])
        col, _ = dr.interpolate(n, rast, fi)
        alpha = (rast[..., 3:] > 0).float()
        out = dr.antialias(torch.cat([col, alpha], -1), rast, clip, fi)
        return out[..., :3], out[..., 3], rast[..., 3].long() - 1

    v = torch.tensor(v0, device=dev)
    f = torch.tensor(f0, device=dev)
    opt = MeshOptimizer(v, f, edge_len_lims=(a.edge_end, a.edge_start), local_edgelen=False,
                        laplacian_weight=a.laplacian, gain=0.1)
    v = opt.vertices
    log = []
    for i in range(a.steps):
        tgt_n = levels[min(len(levels) - 1, int(i / a.steps * len(levels)))]
        opt.zero_grad()
        opt._lr *= a.decay
        n = calc_vertex_normals(v, f)
        rn, ra, tri = render(v, n, f)
        both = obj & (ra > 0.5)
        if a.select > 0:
            # each triangle listens mostly to the view that sees it most
            # squarely: neighbouring views draw the same pleat a few pixels
            # apart, and averaging them carved a zigzag between the two
            with torch.no_grad():
                fn = torch.nn.functional.normalize(torch.linalg.cross(
                    v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1), dim=-1)
                cosv = (fn @ view_dir.T).clamp(min=0)                 # F, C
                sel = (cosv / cosv.max(1, keepdim=True).values.clamp(min=1e-6)) ** a.select
                pix = sel.T[torch.arange(len(names), device=dev)[:, None, None], tri.clamp(min=0)]
                pix = torch.where(tri >= 0, pix, torch.zeros_like(pix))
        # on colours (n + 1) / 2 and averaged over channels, as Unique3D does:
        # summed over [-1, 1] components it outweighed the silhouette 12 to 1
        # and the mesh swelled past its outline
        r = ((rn - tgt_n) / 2)[both]
        w = wgt[both] * (pix[both] if a.select > 0 else 1.0)
        per = r.pow(2).sum(-1) if a.loss == "l2" else (r.pow(2).sum(-1) + 1e-6).sqrt()
        l_n = (per * w).sum() / w.sum().clamp(min=1e-6) / 3
        l_a = (ra - tgt_a).pow(2).mean()
        l_e = 0.5 * ((v + n).detach() - v).pow(2).mean()
        loss = a.w_normal * l_n + a.w_alpha * l_a + a.w_expand * l_e
        loss = loss + (v.abs() > ortho / 2).float().mean() * 10
        loss.backward()
        opt.step()
        # target edge length on a fixed schedule, coarse to fine over the first
        # --edge-steps fraction: the optimiser's own controller lengthens edges
        # whenever the surface moves slowly, and left the cat 2,400 faces
        t = min(1.0, i / max(1, a.edge_steps * a.steps))
        opt._ref_len.fill_(a.edge_start * (a.edge_end / a.edge_start) ** t)
        v, f = opt.remesh()
        with torch.no_grad():
            cos = (torch.nn.functional.normalize(rn, dim=-1) * tgt_n).sum(-1)[both].clamp(-1, 1)
            ang = torch.rad2deg(torch.acos(cos)).median().item()
            iou = ((ra > 0.5) & obj).sum().item() / max(((ra > 0.5) | obj).sum().item(), 1)
        log.append({"step": i, "loss_normal": l_n.item(), "loss_alpha": l_a.item(),
                    "normal_median_deg": ang, "silhouette_iou": iou, "faces": len(f)})
        if i % 10 == 0 or i == a.steps - 1:
            print(f"step {i:4d}  normal {ang:5.1f} deg  IoU {iou:.4f}  faces {len(f):,}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        if a.snap_every and (i % a.snap_every == 0 or i == a.steps - 1):
            with torch.no_grad():
                img = ((rn.flip(1) + 1) / 2 * ra.flip(1)[..., None]).clamp(0, 1)
                tiles = [np.concatenate([img[k].cpu().numpy(),
                                         ((tgt_n[k].flip(0) + 1) / 2 * tgt_a[k].flip(0)[..., None]).cpu().numpy()], 0)
                         for k in range(len(names))]
                Image.fromarray((np.concatenate(tiles, 1) * 255).astype(np.uint8)).resize(
                    (len(names) * 256, 512)).save(os.path.join(a.out, f"snap_{i:04d}.png"))
    vf, ff = v.detach().cpu().numpy(), f.cpu().numpy()
    write_ply(os.path.join(a.out, "mesh.ply"), vf, ff)
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"))
    print(f"wrote {a.out}/mesh.ply  {len(ff):,} faces  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
