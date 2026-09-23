#!/usr/bin/env python3
"""All fourteen depth maps as one linear system.

Each view on its own knows its surface's slopes (photometric normals, 0.9 deg
median error on Lucy) but not its height: normal integration leaves one
unknown constant per piece, and a piece is cut off wherever the view sees a
depth jump. Where the sweep found no anchor, the height was a guess, and
around toes, fingers, a torch knob -- five pixels of surface, then a jump --
every view guessed wrong on its own: 42% of the points still holed after two
consensus passes had no view with the right depth.

But the jumps are where *that* view's line of sight grazes an occluder, and
they are somewhere else in every other view. A toe cut off from the foot in
the front view runs smoothly into it seen from the side. So the views are
solved together, as one surface seen fourteen times:

  integrability   per view, depth differences follow the normals (robustly:
                  an edge that disagrees is a jump and loses its weight)
  anchors         sweep anchors, and silhouette contacts (below)
  coupling        a pixel of view A at depth z is a 3D point X; a view B that
                  faces X must have its own depth at X there. Linear in the two
                  depths (orthographic cameras, B's pixel fixed per round).
                  Only occlusion is excluded: if B sees something clearly in
                  front of X, X is hidden from B. If B's surface is *behind*
                  X, B has looked straight through a surface and is pulled
                  forward -- that is the case that punched the holes.

Contacts (the silhouette's free answers): the visual hull touches the true
surface along every view's rim, and exactly there the hull's normal equals
the surface's. A hull point whose normal matches this view's photometric
normal, and whose normal the other views facing it also report at its
projection, is taken as an anchor at the hull's depth. On Lucy the check
passes 80-89% of such points far from any sweep anchor, where nothing else
knows the depth.

Solved at half resolution (relief scale 2) with IRLS over all three robust
terms, the couplings re-linearised each round; the result goes back to
depth_mv.py --consensus as dense anchors for the full-resolution solve.

Measured on Lucy (pass-1 depth in, GT out), and not in the default pipeline:
  - it fixes 5-8% of the pixels the input had wrong by more than 3 voxels
    and breaks 3.5-8% of those it had right. A coupling is a local
    correction inside its visibility gate; the pieces that punch holes are
    5-100 voxels off, far outside it, and read as occlusion.
  - the one-sided occlusion rule (keep a coupling when B's surface is
    behind X) biases everything forward, median -0.8 voxel; --sym 3 halves
    that and fixes less.
  - contacts change nothing measurable (the same numbers with --no-contacts).
What did fix the anchors was upstream of all this: the sweep's matching
cost (normal_stereo.sweep_fast, trunc).

Run:
  python3 tools/joint_depth.py --views refs/lucy_gt --hull-views out/hull/views \
      --normals-dir out/ps_normals --depth-dir out/depth --out out/joint
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import ndimage, sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_contact as dcx  # noqa: E402
import normal_stereo as ns  # noqa: E402
from linsolve import spd_solve  # noqa: E402

BG = 1e9
warnings.filterwarnings("ignore", category=RuntimeWarning)


def blocks(a, k):
    H, W = a.shape[:2]
    h2, w2 = H // k, W // k
    return a[:h2 * k, :w2 * k].reshape((h2, k, w2, k) + a.shape[2:])


def cam(meta, v):
    m = np.array(meta["views"][v]["matrix_world"], dtype=np.float64)
    return m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3]


def depth_normals(D, px):
    """Camera-frame normals of a depth map (same convention as the
    photometric normals: n ~ (dD/dcol, -dD/drow, px) normalised)."""
    Dr = np.gradient(D, axis=0)
    Dc = np.gradient(D, axis=1)
    n = np.stack([Dc / px, -Dr / px, np.ones_like(D)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


class View:
    pass


def load_view(args, meta, v, k, res):
    """Half-resolution everything for one view."""
    V = View()
    V.name = v
    hull, n, rgb = dcx.load(args.views, args.hull_views, args.normals_dir, v, meta)
    V.px = meta["ortho_scale"] / res * k
    hb = blocks(hull, k)
    V.hit = np.isfinite(hb).all((1, 3))              # whole block on the object
    V.hull = np.where(V.hit, np.nanmean(hb, (1, 3)), np.nan)
    nb = blocks(n, k).mean((1, 3))
    V.n = nb / np.linalg.norm(nb, axis=-1, keepdims=True).clip(1e-9)
    rgbb = blocks(rgb, k).mean((1, 3))
    d = np.load(os.path.join(args.depth_dir, f"{v}.npy")).astype(np.float64)
    V.z0 = np.where(V.hit, np.nanmean(blocks(d, k), (1, 3)), np.nan)
    V.z0 = np.where(np.isfinite(V.z0), V.z0, V.hull)
    ad = np.load(os.path.join(args.depth_dir, f"{v}_anchordist.npy")).astype(np.float64)
    live = np.nan_to_num(ad, nan=1e9) == 0
    cnt = blocks(live, k).sum((1, 3))
    za = np.nansum(blocks(np.where(live, d, 0.0), k), (1, 3)) / np.maximum(cnt, 1)
    V.anc = V.hit & (cnt >= max(1, k * k // 2))
    V.anc_z = za
    keep = dcx.cut_edges(V.n, rgbb, V.hull, V.hit, args.budget)
    V.a, V.b, V.grad, V.idx = dcx.build_edges(V.n, V.hit, keep, V.px, args.nz_floor)
    V.N = int(V.hit.sum())
    V.rows, V.cols = np.nonzero(V.hit)
    right, up, back, loc = cam(meta, v)
    V.right, V.up, V.back, V.loc = right, up, back, loc
    R = np.array(meta["views"][v]["matrix_world"], dtype=np.float64)[:3, :3]
    V.nw = (V.n.reshape(-1, 3) @ R.T).reshape(V.n.shape)   # world normals
    resh = res // k
    V.resh = resh
    o = meta["ortho_scale"]
    x = ((V.cols + 0.5) / resh - 0.5) * o
    y = (0.5 - (V.rows + 0.5) / resh) * o
    V.P0 = loc + x[:, None] * right + y[:, None] * up        # depth 0 on the ray
    return V


def project(V, X, o):
    rel = X - V.loc
    col = (rel @ V.right / o + 0.5) * V.resh - 0.5
    row = (0.5 - rel @ V.up / o) * V.resh - 0.5
    w = -(rel @ V.back)
    return row, col, w


def contacts(Vs, o, h, ang_self, ang_other, min_agree):
    """Hull points that touch the true surface: see the module docstring."""
    out = {}
    for V in Vs:
        nh = depth_normals(np.where(V.hit, V.hull, np.nan), V.px)
        cosang = np.clip((nh * V.n).sum(-1), -1, 1)
        cand = V.hit & (cosang > np.cos(np.radians(ang_self)))
        r, c = np.nonzero(cand)
        if len(r) == 0:
            out[V.name] = (r, c)
            continue
        i = V.idx[r, c]
        X = V.P0[i] - V.hull[r, c][:, None] * V.back
        nw = V.nw[r, c]
        agree = np.zeros(len(r), int)
        for B in Vs:
            if B is V:
                continue
            facing = nw @ B.back > 0.3
            rr, cc, w = project(B, X, o)
            ri = np.clip(np.rint(rr).astype(int), 0, B.resh - 1)
            ci = np.clip(np.rint(cc).astype(int), 0, B.resh - 1)
            hb = B.hull[ri, ci]
            vis = facing & np.isfinite(hb) & (hb <= w + h)
            dot = (B.nw[ri, ci] * nw).sum(-1)
            agree += vis & (dot > np.cos(np.radians(ang_other)))
        ok = agree >= min_agree
        out[V.name] = (r[ok], c[ok])
    return out


def couplings(Vs, off, z, o, h, tau_occ, facing_min, stride, top, sym=0.0):
    """Rows  sum_i beta_i z_B[q_i] - gamma z_A[p] = c0  for visible pairs."""
    Ra, Ca, Va, rhs, pairs = [], [], [], [], []
    row0 = 0
    for A in Vs:
        sel = ((A.rows % stride) == 0) & ((A.cols % stride) == 0)
        ia = np.nonzero(sel)[0]
        zA = z[off[A.name] + ia]
        X = A.P0[ia] - zA[:, None] * A.back
        nw = A.nw[A.rows[ia], A.cols[ia]]
        # the `top` most frontal other views per pixel
        facing = np.stack([nw @ B.back for B in Vs])            # [V, n]
        facing[[j for j, B in enumerate(Vs) if B is A]] = -1
        rank = np.argsort(-facing, axis=0)[:top]
        for j, B in enumerate(Vs):
            if B is A:
                continue
            use = (rank == j).any(0) & (facing[j] > facing_min)
            if not use.any():
                continue
            p = ia[use]
            Xu = X[use]
            rr, cc, w = project(B, Xu, o)
            r0 = np.floor(rr).astype(int)
            c0_ = np.floor(cc).astype(int)
            fr, fc = rr - r0, cc - c0_
            inside = (r0 >= 0) & (c0_ >= 0) & (r0 < B.resh - 1) & (c0_ < B.resh - 1)
            r0, c0_, fr, fc, w, p = (q[inside] for q in (r0, c0_, fr, fc, w, p))
            idx4 = [B.idx[r0, c0_], B.idx[r0, c0_ + 1], B.idx[r0 + 1, c0_],
                    B.idx[r0 + 1, c0_ + 1]]
            ok = np.all([q >= 0 for q in idx4], axis=0)
            beta = [(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc]
            zB = sum(bt * z[off[B.name] + np.maximum(q, 0)] for bt, q in zip(beta, idx4))
            # hidden from B only if B sees a surface clearly in front of X;
            # with sym, also dropped when B's surface is clearly behind X
            ok &= (w - zB) < tau_occ * h
            if sym > 0:
                ok &= (zB - w) < sym * h
            if not ok.any():
                continue
            n = int(ok.sum())
            rows = row0 + np.arange(n)
            gamma = float(A.back @ B.back)
            c0v = w[ok] - gamma * z[off[A.name] + p[ok]]
            for bt, q in zip(beta, idx4):
                Ra.append(rows)
                Ca.append(off[B.name] + q[ok])
                Va.append(bt[ok])
            Ra.append(rows)
            Ca.append(off[A.name] + p[ok])
            Va.append(np.full(n, -gamma))
            rhs.append(c0v)
            pairs.append(n)
            row0 += n
    if row0 == 0:
        return None
    M = sparse.csr_matrix((np.concatenate(Va), (np.concatenate(Ra), np.concatenate(Ca))),
                          shape=(row0, len(z)))
    return M, np.concatenate(rhs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--hull-views", required=True)
    ap.add_argument("--normals-dir", required=True)
    ap.add_argument("--depth-dir", required=True,
                    help="current depth maps (with _anchordist.npy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--nz-floor", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=2e-4,
                    help="Cauchy scale of integrability residuals, per full-res px")
    ap.add_argument("--anchor-len", type=float, default=8.0)
    ap.add_argument("--couple-len", type=float, default=4.0,
                    help="coupling strength, as an anchor of this screening length")
    ap.add_argument("--contact-len", type=float, default=8.0)
    ap.add_argument("--sigma-vox", type=float, default=1.5,
                    help="Cauchy scale of anchor and coupling residuals")
    ap.add_argument("--tau-occ", type=float, default=3.0)
    ap.add_argument("--facing-min", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=2,
                    help="couple every stride-th pixel each way")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--no-contacts", action="store_true")
    ap.add_argument("--sym", type=float, default=0.0,
                    help="if > 0, also drop a coupling when B's surface is "
                         "this many voxels behind X")
    ap.add_argument("--no-couple", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    meta = json.load(open(os.path.join(args.views, "cameras.json")))
    res = meta["resolution"][0]
    o = meta["ortho_scale"]
    h = 1.0 / 1024
    k = args.scale
    Vs = [load_view(args, meta, v, k, res) for v in meta["views"]]
    off, tot = {}, 0
    for V in Vs:
        off[V.name] = tot
        tot += V.N
    z = np.concatenate([V.z0[V.hit] for V in Vs])
    print(f"loaded {len(Vs)} views, {tot:,} unknowns: {time.time()-t0:.0f}s", flush=True)

    # anchors: sweep anchors, and silhouette contacts
    AI, AZ, AW = [], [], []
    wa0 = 1.0 / ((Vs[0].px ** 2) * args.anchor_len ** 2)
    for V in Vs:
        m = V.anc
        AI.append(off[V.name] + V.idx[m])
        AZ.append(V.anc_z[m])
        AW.append(np.full(int(m.sum()), wa0))
    n_contact = 0
    if not args.no_contacts:
        con = contacts(Vs, o, h, 3.0, 3.0, 2)
        wc0 = 1.0 / ((Vs[0].px ** 2) * args.contact_len ** 2)
        for V in Vs:
            r, c = con[V.name]
            AI.append(off[V.name] + V.idx[r, c])
            AZ.append(V.hull[r, c])
            AW.append(np.full(len(r), wc0))
            n_contact += len(r)
    AI, AZ, AW = np.concatenate(AI), np.concatenate(AZ), np.concatenate(AW)
    print(f"anchors {len(AI) - n_contact:,} + contacts {n_contact:,}: "
          f"{time.time()-t0:.0f}s", flush=True)

    # integrability, all views block-diagonal
    Ea = np.concatenate([off[V.name] + V.a for V in Vs])
    Eb = np.concatenate([off[V.name] + V.b for V in Vs])
    Eg = np.concatenate([V.grad for V in Vs])
    Epx = np.concatenate([np.full(len(V.a), V.px) for V in Vs])
    sig_e = args.sigma * k
    sig_a = args.sigma_vox * h
    # robust weights start from the input depth, not from one: with every
    # edge at full weight the first solve integrates straight across the
    # jumps each view has already found, and smears them
    we = 1.0 / (1.0 + (((z[Eb] - z[Ea]) - Eg) / sig_e) ** 2)
    wa = AW / (1.0 + ((z[AI] - AZ) / sig_a) ** 2)
    wcpl = 1.0 / ((Vs[0].px ** 2) * args.couple_len ** 2)
    m_e = len(Ea)
    re = np.arange(m_e)
    for rnd in range(args.rounds + 1):
        sw = np.sqrt(we) / Epx
        E = sparse.csr_matrix((np.concatenate([sw, -sw]),
                               (np.concatenate([re, re]), np.concatenate([Eb, Ea]))),
                              shape=(m_e, tot))
        AtA = (E.T @ E).tocsr()
        Atb = E.T @ (Eg * sw)
        cpl = None if args.no_couple else couplings(
            Vs, off, z, o, h, args.tau_occ, args.facing_min, args.stride, args.top,
            args.sym)
        if cpl is not None:
            M, c = cpl
            res_c = M @ z - c
            wcv = wcpl / (1.0 + (res_c / (sig_a if rnd else 4 * sig_a)) ** 2)
            Mw = sparse.diags(np.sqrt(wcv)) @ M
            AtA = AtA + (Mw.T @ Mw).tocsr()
            Atb = Atb + Mw.T @ (c * np.sqrt(wcv))
        D = np.zeros(tot)
        np.add.at(D, AI, wa)
        rhs = np.zeros(tot)
        np.add.at(rhs, AI, wa * AZ)
        lam = 1e-7 * AtA.diagonal().mean()
        AtA = AtA + sparse.diags(D + lam)
        Atb = Atb + rhs + lam * z
        z = spd_solve(AtA.tocsr(), Atb, x0=z, tol=1e-6)
        res_e = (z[Eb] - z[Ea]) - Eg
        we = 1.0 / (1.0 + (res_e / sig_e) ** 2)
        wa = AW / (1.0 + ((z[AI] - AZ) / sig_a) ** 2)
        note = ""
        if cpl is not None:
            rc = M @ z - c
            note = (f"couplings {M.shape[0]:,} (|res|<1.5 vox "
                    f"{100*np.mean(np.abs(rc) < 1.5*h):.1f}%)")
        print(f"  round {rnd}: {note}  {time.time()-t0:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    for V in Vs:
        zh = np.full(V.hit.shape, np.nan)
        zh[V.hit] = z[off[V.name]:off[V.name] + V.N]
        # back to full resolution: bilinear inside, NaN off the object
        H = res
        filled = np.where(V.hit, zh, np.nanmean(zh))
        rr = (np.arange(H) + 0.5) / k - 0.5
        full = ndimage.map_coordinates(filled, np.meshgrid(rr, rr, indexing="ij"),
                                       order=1, mode="nearest")
        hull_full = np.load(os.path.join(args.hull_views, "depth_npy", f"{V.name}.npy"))
        onb = np.repeat(np.repeat(V.hit, k, 0), k, 1)
        onb = np.pad(onb, ((0, H - onb.shape[0]), (0, H - onb.shape[1])))
        full = np.where((hull_full < BG) & onb, np.maximum(full, hull_full), np.nan)
        np.save(os.path.join(args.out, f"{V.name}.npy"), full.astype(np.float32))
        note = ""
        gt_p = os.path.join(args.views, "depth_npy", f"{V.name}.npy")
        if os.path.exists(gt_p):
            gt = np.load(gt_p)
            m = np.isfinite(full) & (gt < BG)
            e = (full - gt)[m] / h
            d_in = np.load(os.path.join(args.depth_dir, f"{V.name}.npy"))
            m2 = m & np.isfinite(d_in)
            e2 = (d_in - gt)[m2] / h
            note = (f"|e|<1.5: joint {100*np.mean(np.abs(e) < 1.5):5.1f}%  "
                    f"input {100*np.mean(np.abs(e2) < 1.5):5.1f}%   "
                    f"deep>1.5: {100*np.mean(e > 1.5):4.1f}% / {100*np.mean(e2 > 1.5):4.1f}%  "
                    f"shallow<-1.5: {100*np.mean(e < -1.5):4.1f}% / {100*np.mean(e2 < -1.5):4.1f}%  "
                    f"median {np.median(e):+.2f}")
        print(f"  {V.name:<13} {note}", flush=True)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
