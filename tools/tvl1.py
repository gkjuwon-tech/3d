"""TV-L1 fusion of truncated signed distances (Zach, Pock & Bischof 2007;
histogram form after Zach 2008), on the xp backend.

Minimises over a field u in [-1, 1] (voxel units / trunc, >0 outside)

    E(u) = sum |grad u|  +  lam * sum_x sum_b h_b(x) |u(x) - c_b|,
    subject to u >= hull

where h_b(x) is the confidence-weighted count of views whose truncated
signed distance at x falls in bin b (centre c_b). The L1 data term makes the
fused value a (regularised) median of the views rather than their mean, so a
minority of views with a wrong depth cannot carve through what the others
agree on; the total variation term charges for surface area, so a surface
cannot shatter into fragments to satisfy disagreeing views. Solved with the
first-order primal-dual method of Chambolle & Pock (2011).

Measured, and not used: on the torch hand (a 124^3 crop of Lucy) the total
variation term is exactly the wrong prior for this problem. It charges for
surface area, and a thin feature is almost all surface, so the regulariser
removes it first -- at lam 0.5 the torch, the hand and the fingers vanished
and only the forearm stump survived; at lam 2 they came back as blocks,
because the minimiser snaps to the histogram's bin centres. The robust part
of the idea (a median over views instead of a mean) is what worked, without
the TV term: see fuse_field.py --fusion robust.
"""
from xp import xp


def grad(u):
    g = xp.zeros((3,) + u.shape, dtype=u.dtype)
    g[0, :-1] = u[1:] - u[:-1]
    g[1, :, :-1] = u[:, 1:] - u[:, :-1]
    g[2, :, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
    return g


def div(p):
    d = xp.zeros(p.shape[1:], dtype=p.dtype)
    d[:-1] += p[0, :-1]
    d[1:] -= p[0, :-1]
    d[:, :-1] += p[1, :, :-1]
    d[:, 1:] -= p[1, :, :-1]
    d[:, :, :-1] += p[2, :, :, :-1]
    d[:, :, 1:] -= p[2, :, :, :-1]
    return d


def prox_hist(u0, hist, centres, t):
    """argmin_u (u-u0)^2/(2t) + sum_b hist_b |u - c_b|, per voxel.

    Between consecutive centres the objective is quadratic with stationary
    point s_k = u0 - t * (weight below - weight above); the s_k fall as k
    rises while the centres climb, and the minimiser is where the two
    sequences cross (Li & Osher's generalised median). That crossing is
    max_k min(s_k, c_{k+1}), with c_{B+1} = +inf: B+1 elementwise steps.
    """
    B = len(centres)
    W = hist.sum(0)
    below = 0 * u0
    best = None
    for k in range(B + 1):
        s = u0 - t * (2 * below - W)
        if k < B:
            s = xp.minimum(s, float(centres[k]))
            below = below + hist[k]
        best = s if best is None else xp.maximum(best, s)
    return best


def tvl1(hist, centres, lower, lam=1.0, iters=300, u0=None, active=None,
         verbose=False):
    """Primal-dual TV-L1. hist: [B, *shape] weights; lower: hull bound on u
    (same shape); active: bool mask of voxels free to change (others keep u0).
    """
    L2 = 12.0                                         # ||grad||^2 <= 4 * 3
    tau = sigma = 1.0 / (L2 ** 0.5)
    u = lower.copy() if u0 is None else u0.copy()
    u = xp.maximum(u, lower)
    ubar = u.copy()
    p = xp.zeros((3,) + u.shape, dtype=u.dtype)
    for it in range(iters):
        p = p + sigma * grad(ubar)
        norm = xp.maximum(1.0, xp.sqrt((p * p).sum(0)))
        p = p / norm
        u_old = u
        u = prox_hist(u + tau * div(p), lam * hist, centres, tau)
        u = xp.clip(xp.maximum(u, lower), -1.0, 1.0)
        if active is not None:
            u = xp.where(active, u, u_old)
        ubar = 2 * u - u_old
        if verbose and it % 50 == 0:
            print(f"    tvl1 iter {it}: |du| {float(xp.abs(u - u_old).mean()):.2e}",
                  flush=True)
    return u
