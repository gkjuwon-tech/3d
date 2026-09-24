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
    """masks, world-space normals, facing weights, and per view: camera,
    ortho scale, weight and valid region (views/valid/<view>.png, for
    close-ups whose crop edge is a cut through the body, not an outline)"""
    meta = json.load(open(os.path.join(views, "cameras.json")))
    names = list(meta["views"])
    out = {k: [] for k in ("M", "mask", "nrm", "face", "ortho", "weight", "valid")}
    for v in names:
        info = meta["views"][v]
        m = np.asarray(Image.open(os.path.join(views, "mask", f"{v}.png")).convert("L")
                       .resize((res, res), Image.BILINEAR), np.float32) / 255
        vp = os.path.join(views, "valid", f"{v}.png")
        val = (np.asarray(Image.open(vp).convert("L").resize((res, res), Image.NEAREST)) > 127
               if os.path.exists(vp) else np.ones((res, res), bool)).astype(np.float32)
        n = np.load(os.path.join(normals, f"{v}.npy")).astype(np.float32)
        if n.shape[0] != res:
            n = np.stack([np.asarray(Image.fromarray(n[..., c]).resize((res, res), Image.BILINEAR))
                          for c in range(3)], -1)
        W = np.array(info["matrix_world"], np.float64)
        nw = n @ W[:3, :3].T.astype(np.float32)            # camera frame -> world
        nw /= np.linalg.norm(nw, axis=-1, keepdims=True).clip(1e-6)
        nw[m < 0.5] = 0
        fz = np.clip(n[..., 2], 0, 1)                     # how squarely it faces the camera
        fz[m < 0.5] = 0
        for k, x in (("M", W), ("mask", m), ("nrm", nw), ("face", fz), ("valid", val),
                     ("ortho", float(info.get("ortho_scale", meta["ortho_scale"]))),
                     ("weight", float(info.get("weight", 1.0)))):
            out[k].append(x)
    return meta, names, out


def clip_matrices(M, orthos, near=0.1, far=4.0):
    """world -> clip for each orthographic camera (OpenGL conventions)"""
    out = []
    for m, o in zip(M, orthos):
        h = o / 2
        P = np.array([[1 / h, 0, 0, 0], [0, 1 / h, 0, 0],
                      [0, 0, -2 / (far - near), -(far + near) / (far - near)], [0, 0, 0, 1]])
        out.append(P @ np.linalg.inv(m))
    return np.stack(out).astype(np.float32)


def read_ply(path):
    """the binary PLY write_ply writes (float xyz, uchar-count int faces);
    no trimesh, which the Kaggle image does not have"""
    with open(path, "rb") as fh:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += fh.readline()
        lines = head.decode().splitlines()
        nv = next(int(l.split()[2]) for l in lines if l.startswith("element vertex"))
        nf = next(int(l.split()[2]) for l in lines if l.startswith("element face"))
        props = [l.split()[2] for l in lines if l.startswith("property float")]
        v = np.frombuffer(fh.read(nv * 4 * len(props)), "<f4").reshape(nv, len(props))[:, :3]
        rec = np.frombuffer(fh.read(nf * 13), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    return v.astype(np.float32), rec["i"].astype(np.int64)


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


def subdivide(v, f):
    """split every triangle into four at its edge midpoints (torch)"""
    import torch
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e_sorted = torch.sort(e, dim=1).values
    uniq, inv = torch.unique(e_sorted, dim=0, return_inverse=True)
    mid = (v[uniq[:, 0]] + v[uniq[:, 1]]) / 2
    nv = len(v)
    m = inv.view(3, -1).T + nv                           # F, 3: midpoints of edges 01, 12, 20
    a_, b_, c_ = f[:, 0], f[:, 1], f[:, 2]
    m01, m12, m20 = m[:, 0], m[:, 1], m[:, 2]
    nf = torch.cat([torch.stack([a_, m01, m20], 1), torch.stack([m01, b_, m12], 1),
                    torch.stack([m20, m12, c_], 1), torch.stack([m01, m12, m20], 1)], 0)
    return torch.cat([v, mid], 0), nf


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
    ap.add_argument("--extra-views", nargs="*", default=[],
                    help="more view sets (e.g. close-ups from crop_register.py), each with its "
                         "own cameras, ortho scales, weights and valid regions")
    ap.add_argument("--extra-normals", nargs="*", default=[])
    ap.add_argument("--init", default=None, help="start from this mesh instead of the hull")
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
    ap.add_argument("--cam-lr", type=float, default=2e-3, help="Adam step for the camera refinement")
    ap.add_argument("--cam-steps", type=float, default=0.6,
                    help="fraction of the run during which cameras are refined (then frozen)")
    ap.add_argument("--cam-reg", type=float, default=1e-3)
    ap.add_argument("--shading", default="flat", choices=["flat", "smooth"],
                    help="render each triangle's own normal (flat) or interpolated vertex normals")
    ap.add_argument("--detail-steps", type=int, default=0,
                    help="after the shape: steps of normal-direction displacement only")
    ap.add_argument("--max-disp", type=float, default=0.01, help="detail: largest displacement (world units)")
    ap.add_argument("--subdivide", type=int, default=2,
                    help="detail: split every triangle into 4 this many times first")
    ap.add_argument("--detail-lr", type=float, default=0.05)
    ap.add_argument("--detail-smooth", type=float, default=0.02)
    ap.add_argument("--box-margin", type=float, default=0.005,
                    help="world units the surface may stray outside the hull's bounding box")
    ap.add_argument("--rim", type=int, default=3,
                    help="px of each outline excluded from the normal loss")
    ap.add_argument("--dir-smooth", type=int, default=10,
                    help="detail: neighbour-averaging passes on the displacement direction")
    ap.add_argument("--views-per-step", type=int, default=0,
                    help="render a random subset of this many views per step (0 = all); "
                         "every 10th step (20th in detail) still renders all, for logging")
    ap.add_argument("--holdout", default="", help="views (comma list) left out of the fit, scored only")
    ap.add_argument("--select", type=float, default=8.0,
                    help="per triangle, weight view k by (cos_k / best cos)^s; 0 = all views equal")
    ap.add_argument("--blur", default="8,4,2,1,0",
                    help="gaussian sigmas (px) of the target normals, in equal stages of the run")
    ap.add_argument("--facing-pow", type=float, default=1.0,
                    help="weight each target normal by (its facing the camera)^p; 0 = uniform")
    ap.add_argument("--gm-deg", type=float, default=15.0,
                    help="gm loss: angle (deg) beyond which a target normal stops pulling")
    ap.add_argument("--loss", default="l1", choices=["l1", "l2", "gm"],
                    help="normal loss per pixel: l1 (robust: a wrong normal pulls less) or l2")
    ap.add_argument("--snap-every", type=int, default=50, help="steps between progress renders")
    a = ap.parse_args()

    import torch
    import nvdiffrast.torch as dr
    from meshfit.opt import MeshOptimizer
    from meshfit.remesh import calc_vertex_normals

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    meta, names, D = load_views(a.views, a.normals, a.res)
    ortho = float(meta["ortho_scale"])
    n_main = len(names)
    for ev, en in zip(a.extra_views, a.extra_normals):
        _, nm, E = load_views(ev, en, a.res)
        names += nm
        for k in D:
            D[k] += E[k]
    M, masks, nrm, face = (np.stack(D[k]) for k in ("M", "mask", "nrm", "face"))
    valid = np.stack(D["valid"])
    if a.init:
        v0, f0 = read_ply(a.init)
    else:
        v0, f0 = carve_hull(M[:n_main], masks[:n_main], ortho, a.hull_res)
    write_ply(os.path.join(a.out, "init.ply"), v0, f0)
    # nothing may leave the hull's box: a surface outside every camera's
    # frame meets no loss at all, and the inflation term alone grew the owl a
    # bowl 0.3 units below its plinth. (The check inherited from Unique3D,
    # (|v| > h).float().mean(), counts and has no gradient.)
    box_lo = v0.min(0) - a.box_margin
    box_hi = v0.max(0) + a.box_margin
    print(f"{len(names)} views at {a.res}; hull {len(v0):,} vertices, {len(f0):,} faces "
          f"[{time.time()-t0:.0f}s]", flush=True)

    dev = "cuda"
    glctx = dr.RasterizeCudaContext(device=dev)
    val = torch.tensor(valid[:, ::-1].copy(), device=dev)
    vw = torch.tensor(D["weight"], device=dev, dtype=torch.float32)[:, None, None]
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
    blo = torch.tensor(box_lo, device=dev, dtype=torch.float32)
    bhi = torch.tensor(box_hi, device=dev, dtype=torch.float32)
    # normals are compared only a few pixels inside each outline: at the rim a
    # learned normal is least reliable and the views' outlines disagree by a
    # pixel or two, and there the fit frayed every sharp edge into crumbs
    from scipy import ndimage as _ndi
    core = np.stack([_ndi.binary_erosion(m > 0.5, iterations=a.rim) if a.rim > 0 else m > 0.5
                     for m in masks])
    objn = torch.tensor(core[:, ::-1].copy(), device=dev)
    # a learned normal is least reliable where the surface turns away from
    # the camera, and there every other view sees it better
    wgt = torch.tensor(face[:, ::-1].copy(), device=dev) ** a.facing_pow * vw

    # --- camera refinement (bundle adjustment) ---------------------------------
    # The generator's cameras are nominal: a view "at 45 degrees" may have been
    # drawn at 44, a few pixels off centre, a percent larger. Then every
    # outline and fold points at a different 3D place in each view, and one
    # surface can only satisfy them all by zigzagging between them. So each
    # view's camera is refined with the mesh: a turn about the vertical axis,
    # a tilt, a shift in the image plane and a scale. The first view is held
    # fixed as the reference.
    C = len(names)
    Mt = torch.tensor(M, device=dev, dtype=torch.float32)
    Rt0 = Mt[:, :3, :3].clone()
    hs = torch.tensor([o / 2 for o in D["ortho"]], device=dev)
    cam = {k: torch.zeros(C, device=dev, requires_grad=True) for k in ("az", "tilt", "tx", "ty", "ls")}
    cam_opt = torch.optim.Adam(list(cam.values()), lr=a.cam_lr)
    free = torch.ones(C, device=dev)
    free[0] = 0                                           # reference view
    train = torch.ones(C, device=dev)
    if a.holdout:
        # left out of every loss and not refined: how well the mesh draws a
        # view it never saw is the honest test of whether it is one shape or
        # six separate illusions, each right from its own camera only
        for h_ in a.holdout.split(","):
            train[names.index(h_)] = 0
            free[names.index(h_)] = 0
    tr = train[:, None, None] > 0.5
    near, far = 0.1, 4.0

    def rot(axis, ang):
        """rotation matrices (C,3,3) about unit axes (C,3) by angles (C,)"""
        K = torch.zeros(C, 3, 3, device=dev)
        K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
        K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
        s_, c_ = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
        return torch.eye(3, device=dev)[None] + s_ * K + (1 - c_) * (K @ K)

    def cameras():
        """refined camera-to-world rotations (C,3,3) and world-to-clip (C,4,4)"""
        zax = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(C, 3)
        Rz = rot(zax, cam["az"] * free)
        Rt = rot(Rt0[:, :, 0], cam["tilt"] * free)
        Rm = Rz @ Rt
        R = Rm @ Rt0
        pos = (Rm @ Mt[:, :3, 3:])[..., 0]
        Vw = torch.zeros(C, 4, 4, device=dev)
        Vw[:, :3, :3] = R.transpose(1, 2)
        Vw[:, :3, 3] = -(R.transpose(1, 2) @ pos[..., None])[..., 0]
        Vw[:, 3, 3] = 1
        sc = hs * torch.exp(cam["ls"] * free)
        P = torch.zeros(C, 4, 4, device=dev)
        P[:, 0, 0], P[:, 1, 1] = 1 / sc, 1 / sc
        P[:, 0, 3], P[:, 1, 3] = cam["tx"] * free, cam["ty"] * free
        P[:, 2, 2], P[:, 2, 3] = -2 / (far - near), -(far + near) / (far - near)
        P[:, 3, 3] = 1
        return R, P @ Vw

    with torch.no_grad():
        err = (cameras()[1] - torch.tensor(clip_matrices(M, D["ortho"]), device=dev)).abs().max().item()
    print(f"camera model check (refinement at zero vs nominal): max diff {err:.2e}", flush=True)
    assert err < 1e-4, "refined camera model does not reduce to the nominal one"
    # targets are kept in each camera's own frame and turned into world space
    # with the refined rotation every step
    levels_cam = [torch.einsum("chwk,ckj->chwj", lv, Rt0) for lv in levels]

    def render(v, n, f, mvp):
        vh = torch.cat([v, torch.ones_like(v[:, :1])], -1)
        clip = vh @ mvp.transpose(-2, -1)
        fi = f.int()
        rast, _ = dr.rasterize(glctx, clip, fi, resolution=[a.res, a.res])
        tri_id = rast[..., 3].long() - 1
        if a.shading == "flat":
            # each pixel shows its own triangle's orientation. With vertex
            # normals interpolated across triangles, a sawtooth surface renders
            # as the smooth average of its teeth, so the normal loss cannot see
            # the zigzag it is satisfying; with the faces' own normals it can
            fn = torch.nn.functional.normalize(torch.linalg.cross(
                v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1), dim=-1)
            col = torch.where((tri_id >= 0)[..., None], fn[tri_id.clamp(min=0)],
                              torch.zeros(1, device=v.device))
        else:
            col, _ = dr.interpolate(n, rast, fi)
        alpha = (rast[..., 3:] > 0).float()
        out = dr.antialias(torch.cat([col, alpha], -1), rast, clip, fi)
        return out[..., :3], out[..., 3], tri_id

    def image_losses(v, f, n, mvp, tgt_n, R, idx=None):
        """losses over the views idx (all when None): rendering every view at
        every step made an 18-view fit three times slower than a 6-view one,
        and a random subset per step converges to the same place"""
        if idx is None:
            idx = torch.arange(len(names), device=dev)
        O, ON, VA, TA, WG, VW, TR = (x[idx] for x in (obj, objn, val, tgt_a, wgt, vw, tr))
        mvp, tgt_n, R = mvp[idx], tgt_n[idx], R[idx]
        view_dir = R[:, :, 2].detach()
        rn, ra, tri = render(v, n, f, mvp)
        seen = O & (ra > 0.5) & (VA > 0.5)
        both = seen & TR & ON
        pix = None
        if a.select > 0:
            # each triangle listens mostly to the view that sees it most
            # squarely: neighbouring views draw the same pleat a few pixels
            # apart, and averaging them carved a zigzag between the two
            with torch.no_grad():
                fn = torch.nn.functional.normalize(torch.linalg.cross(
                    v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1), dim=-1)
                cosv = (fn @ view_dir.T).clamp(min=0)                 # F, C
                sel = (cosv / cosv.max(1, keepdim=True).values.clamp(min=1e-6)) ** a.select
                pix = sel.T[torch.arange(len(idx), device=dev)[:, None, None], tri.clamp(min=0)]
                pix = torch.where(tri >= 0, pix, torch.zeros_like(pix))
        # on colours (n + 1) / 2 and averaged over channels, as Unique3D does:
        # summed over [-1, 1] components it outweighed the silhouette 12 to 1
        # and the mesh swelled past its outline
        r = ((rn - tgt_n) / 2)[both]
        w = WG[both] * (pix[both] if pix is not None else 1.0)
        r2 = r.pow(2).sum(-1)
        if a.loss == "l2":
            per = r2
        elif a.loss == "gm":
            # Geman-McClure: a normal far from what the surface and the other
            # views say stops pulling at all, instead of pulling half as hard
            # (L1). Where many zoomed views overlap, that is a vote: the
            # consensus wins and the outliers are ignored
            c2 = (np.sin(np.radians(a.gm_deg) / 2)) ** 2          # colour-space residual at that angle
            per = r2 / (r2 + c2)
        else:
            per = (r2 + 1e-6).sqrt()
        l_n = (per * w).sum() / w.sum().clamp(min=1e-6) / 3
        l_a = ((ra - TA).pow(2) * VA * VW * TR).sum() / (VA * VW * TR).sum().clamp(min=1e-6)
        return l_n, l_a, rn, ra, seen, both, tgt_n, O, TR

    def metrics(rn, ra, seen, both, tgt_n, O, TR):
        with torch.no_grad():
            angs = torch.rad2deg(torch.acos((torch.nn.functional.normalize(rn, dim=-1) * tgt_n)
                                            .sum(-1).clamp(-1, 1)))
            ang = angs[both].median().item()
            iou = ((ra > 0.5) & O & TR).sum().item() / max((((ra > 0.5) | O) & TR).sum().item(), 1)
            ho = seen & ~TR
            h_ang = angs[ho].median().item() if ho.any() else float("nan")
            h_iou = (((ra > 0.5) & O & ~TR).sum().item() /
                     max((((ra > 0.5) | O) & ~TR).sum().item(), 1)) if a.holdout else float("nan")
        return ang, iou, h_ang, h_iou

    v = torch.tensor(v0, device=dev)
    f = torch.tensor(f0, device=dev)
    opt = MeshOptimizer(v, f, edge_len_lims=(a.edge_end, a.edge_start), local_edgelen=False,
                        laplacian_weight=a.laplacian, gain=0.1)
    v = opt.vertices
    log = []
    for i in range(a.steps):
        refine = i < a.cam_steps * a.steps
        opt.zero_grad()
        cam_opt.zero_grad()
        opt._lr *= a.decay
        R, mvp = cameras()
        if not refine:
            R, mvp = R.detach(), mvp.detach()
        lv = levels_cam[min(len(levels) - 1, int(i / a.steps * len(levels)))]
        tgt_n = torch.einsum("chwj,ckj->chwk", lv, R.detach())
        tgt_n = torch.where(obj[..., None], tgt_n, torch.zeros_like(tgt_n))
        n = calc_vertex_normals(v, f)
        full = a.views_per_step <= 0 or i % 10 == 0 or i == a.steps - 1
        idx = None if full else torch.randperm(len(names), device=dev)[:a.views_per_step]
        l_n, l_a, rn, ra, seen, both, tgt_s, O_s, TR_s = image_losses(v, f, n, mvp, tgt_n, R, idx)
        l_e = 0.5 * ((v + n).detach() - v).pow(2).mean()
        loss = a.w_normal * l_n + a.w_alpha * l_a + a.w_expand * l_e
        if refine:
            loss = loss + a.cam_reg * sum((x * free).pow(2).mean() for x in cam.values())
        loss.backward()
        opt.step()
        with torch.no_grad():
            v.data.copy_(torch.maximum(torch.minimum(v.data, bhi), blo))
        if refine:
            cam_opt.step()
        # target edge length on a fixed schedule, coarse to fine over the first
        # --edge-steps fraction: the optimiser's own controller lengthens edges
        # whenever the surface moves slowly, and left the cat 2,400 faces
        t = min(1.0, i / max(1, a.edge_steps * a.steps))
        opt._ref_len.fill_(a.edge_start * (a.edge_end / a.edge_start) ** t)
        v, f = opt.remesh()
        ang, iou, h_ang, h_iou = metrics(rn, ra, seen, both, tgt_s, O_s, TR_s)
        log.append({"step": i, "loss_normal": l_n.item(), "loss_alpha": l_a.item(),
                    "normal_median_deg": ang, "silhouette_iou": iou, "faces": len(f),
                    "holdout_normal_median_deg": h_ang, "holdout_iou": h_iou})
        if i % 10 == 0 or i == a.steps - 1:
            print(f"step {i:4d}  normal {ang:5.1f} deg  IoU {iou:.4f}  faces {len(f):,}  "
                  + (f"| held out: normal {h_ang:5.1f} deg IoU {h_iou:.4f}  " if a.holdout else "")
                  + f"[{time.time()-t0:.0f}s]", flush=True)
        if i % 50 == 0 or i == a.steps - 1:
            with torch.no_grad():
                px = a.res / 2
                print("   cameras: " + "  ".join(
                    f"{names[k][:8]} az {np.degrees(cam['az'][k].item()):+.2f} tilt "
                    f"{np.degrees(cam['tilt'][k].item()):+.2f} shift "
                    f"({cam['tx'][k].item() * px:+.1f},{cam['ty'][k].item() * px:+.1f})px "
                    f"scale {np.exp(cam['ls'][k].item()):.4f}" for k in range(1, C)), flush=True)
        if a.snap_every and (i % a.snap_every == 0 or i == a.steps - 1):
            with torch.no_grad():
                img = ((rn.flip(1) + 1) / 2 * ra.flip(1)[..., None]).clamp(0, 1)
                tiles = [np.concatenate([img[k].cpu().numpy(),
                                         ((tgt_n[k].flip(0) + 1) / 2 * tgt_a[k].flip(0)[..., None]).cpu().numpy()], 0)
                         for k in range(len(names))]
                Image.fromarray((np.concatenate(tiles, 1) * 255).astype(np.uint8)).resize(
                    (len(names) * 256, 512)).save(os.path.join(a.out, f"snap_{i:04d}.png"))
    if a.detail_steps:
        # detail as a height field over the settled shape: every vertex may
        # only move along its own normal, and only a little. Free vertices
        # with six views to satisfy built a separate fin for each camera (right
        # from that camera, sheets from anywhere between); a displacement
        # along the normal can carve folds but cannot raise a new wall
        from meshfit.remesh import calc_edges
        with torch.no_grad():
            R, mvp = (x.detach() for x in cameras())
            tgt_n = torch.einsum("chwj,ckj->chwk", levels_cam[-1], R)
            tgt_n = torch.where(obj[..., None], tgt_n, torch.zeros_like(tgt_n))
        # finer triangles first: the shape stage keeps edges long so the mesh
        # cannot grow a fin per camera, and at that size an owl's eye was four
        # triangles wide. Each split turns every triangle into four (edge
        # midpoints, no smoothing), so the displacement has the pixels' worth
        # of vertices to carve with
        for _ in range(a.subdivide):
            v, f = subdivide(v.detach(), f)
        print(f"detail stage on {len(f):,} faces", flush=True)
        base = v.detach().clone()
        edges, _ = calc_edges(f)
        # displacement direction: the vertex normal, smoothed over the
        # neighbourhood. At a crease (a book's edge) the raw normal points
        # diagonally out of the corner, and moving along it pushed the edge
        # sideways into a frayed fringe
        nb = calc_vertex_normals(base, f).detach()
        for _ in range(a.dir_smooth):
            acc = torch.zeros_like(nb)
            acc.index_add_(0, edges[:, 0], nb[edges[:, 1]])
            acc.index_add_(0, edges[:, 1], nb[edges[:, 0]])
            nb = torch.nn.functional.normalize(nb + acc, dim=-1)
        el = (base[edges[:, 0]] - base[edges[:, 1]]).norm(dim=-1).mean()
        h = torch.zeros(len(base), device=dev, requires_grad=True)
        hopt = torch.optim.Adam([h], lr=a.detail_lr)
        for j in range(a.detail_steps):
            hopt.zero_grad()
            d = a.max_disp * torch.tanh(h)
            vv = base + d[:, None] * nb
            nn_ = calc_vertex_normals(vv, f)
            full = a.views_per_step <= 0 or j % 20 == 0 or j == a.detail_steps - 1
            idx = None if full else torch.randperm(len(names), device=dev)[:a.views_per_step]
            l_n, l_a, rn, ra, seen, both, tgt_s, O_s, TR_s = image_losses(vv, f, nn_, mvp, tgt_n, R, idx)
            l_s = ((d[edges[:, 0]] - d[edges[:, 1]]) / el).pow(2).mean()
            loss = a.w_normal * l_n + a.w_alpha * l_a + a.detail_smooth * l_s
            loss.backward()
            hopt.step()
            if j % 20 == 0 or j == a.detail_steps - 1:
                ang, iou, h_ang, h_iou = metrics(rn, ra, seen, both, tgt_s, O_s, TR_s)
                print(f"detail {j:4d}  normal {ang:5.1f} deg  IoU {iou:.4f}  "
                      + (f"| held out: normal {h_ang:5.1f} deg IoU {h_iou:.4f}  " if a.holdout else "")
                      + f"disp |d| {d.abs().mean().item() / el.item():.3f} edges", flush=True)
                log.append({"step": a.steps + j, "normal_median_deg": ang, "silhouette_iou": iou,
                            "holdout_normal_median_deg": h_ang, "holdout_iou": h_iou,
                            "faces": len(f), "stage": "detail"})
        v = (base + a.max_disp * torch.tanh(h)[:, None] * nb).detach()
    with torch.no_grad():
        json.dump({names[k]: {"az_deg": float(np.degrees(cam["az"][k].item())),
                              "tilt_deg": float(np.degrees(cam["tilt"][k].item())),
                              "shift_px": [cam["tx"][k].item() * a.res / 2, cam["ty"][k].item() * a.res / 2],
                              "scale": float(np.exp(cam["ls"][k].item()))} for k in range(C)},
                  open(os.path.join(a.out, "cameras_refined.json"), "w"), indent=1)
    vf, ff = v.detach().cpu().numpy(), f.cpu().numpy()
    write_ply(os.path.join(a.out, "mesh.ply"), vf, ff)
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"))
    print(f"wrote {a.out}/mesh.ply  {len(ff):,} faces  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
