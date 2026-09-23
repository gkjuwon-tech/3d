"""Sub-pixel signed distance to a silhouette, at the image's own resolution.

The hull needs the distance from every pixel to the silhouette outline, and
the outline has to be located to a fraction of a pixel: a whole-pixel
staircase, extruded by the cone, becomes ridges across the body. Finding it
on a 4x upsampled mask worked but put a 67-megapixel distance transform in
every view.

The anti-aliased Euclidean distance transform (Gustavson & Strand 2011, Pattern
Recognition Letters 32(2)) gets the same precision from coverage directly.
An edge pixel's coverage alpha and the local gradient direction fix where a
straight edge crosses it; Gustavson & Strand give the distance from the pixel
centre to that edge in closed form (0.02 px mean error on their tests). A
single distance transform at native resolution then finds each pixel's
nearest edge pixel, and the pixel's own distance is measured to that edge
pixel's sub-pixel edge line (near the outline) or edge point (further off).

Works on NumPy or CuPy arrays through tools/xp.py.
"""
from xp import ndi, xp


def edge_offset(alpha, gx, gy):
    """Signed distance (px) from an edge pixel's centre to the edge, >0 when
    the centre is outside, for an edge with unit normal (gx, gy)."""
    ax, ay = xp.abs(gx), xp.abs(gy)
    g1 = xp.maximum(ax, ay)
    g2 = xp.minimum(ax, ay)
    g1 = xp.maximum(g1, 1e-6)
    a1 = 0.5 * g2 / g1
    lo = 0.5 * (g1 + g2) - xp.sqrt(xp.maximum(2.0 * g1 * g2 * alpha, 0.0))
    mid = (0.5 - alpha) * g1
    hi = -0.5 * (g1 + g2) + xp.sqrt(xp.maximum(2.0 * g1 * g2 * (1.0 - alpha), 0.0))
    return xp.where(alpha < a1, lo, xp.where(alpha < 1.0 - a1, mid, hi))


def aa_sdf(alpha, blend=(1.5, 3.5), mode="level"):
    """Signed distance in pixels, >0 outside, from coverage in [0, 1]."""
    a = xp.clip(xp.asarray(alpha, dtype=xp.float32), 0.0, 1.0)
    inside = a >= 0.5
    # edge pixels: partial coverage, plus both sides of any hard step
    part = (a > 0.02) & (a < 0.98)
    hard = inside ^ ndi.binary_erosion(inside, iterations=1, border_value=0)
    hard |= (~inside) ^ ndi.binary_erosion(~inside, iterations=1, border_value=0)
    edge = part | hard
    # outward normal: coverage grows inward
    gy = -ndi.sobel(a, axis=0)
    gx = -ndi.sobel(a, axis=1)
    gn = xp.sqrt(gx * gx + gy * gy)
    ok = gn > 1e-6
    nx = xp.where(ok, gx / xp.where(ok, gn, 1), 0.0)
    ny = xp.where(ok, gy / xp.where(ok, gn, 1), 0.0)
    if mode == "box":
        # Gustavson & Strand's closed form, exact for area (box) sampling
        df = xp.where(ok, edge_offset(a, nx, ny), 0.5 - a)
    else:
        # filter-agnostic: first-order position of the 0.5 crossing along the
        # gradient. Blender's 1.5 px Blackman-Harris filter spreads an edge
        # over more than a pixel, which the box formula misreads by ~0.3 px;
        # the 0.5 level of any symmetric filter is still the true edge.
        slope = gn / 8.0                       # sobel of a unit ramp is 8
        df = xp.where(ok & (slope > 0.05),
                      xp.clip((0.5 - a) / xp.maximum(slope, 0.05), -1.5, 1.5),
                      0.5 - a)
    # nearest edge pixel of every pixel
    _, ind = ndi.distance_transform_edt(~edge, return_distances=True,
                                        return_indices=True)
    er, ec = ind[0], ind[1]
    H, W = a.shape
    rr = xp.arange(H, dtype=xp.float32)[:, None]
    cc = xp.arange(W, dtype=xp.float32)[None, :]
    dr = rr - er
    dc = cc - ec
    ny_e, nx_e, df_e = ny[er, ec], nx[er, ec], df[er, ec]
    # near: distance to the edge pixel's edge line
    near = df_e + dr * ny_e + dc * nx_e
    # far: distance to its edge point (centre moved onto the line)
    pr = dr + df_e * ny_e
    pc = dc + df_e * nx_e
    far = xp.sqrt(pr * pr + pc * pc)
    d_e = xp.sqrt(dr * dr + dc * dc)
    w = xp.clip((d_e - blend[0]) / (blend[1] - blend[0]), 0.0, 1.0)
    sign = xp.where(inside, -1.0, 1.0)
    mag = (1 - w) * xp.abs(near) + w * far
    # edge pixels and their immediate neighbours keep the line's own sign
    sd = xp.where(w == 0, near, sign * mag)
    return sd.astype(xp.float32)


def contour_sdf(alpha, band=6.0, spacing=0.1):
    """Signed distance in pixels, >0 outside, to the 0.5 coverage contour.

    The contour is traced with marching squares on the bilinear interpolant of
    the coverage -- the same level set the 4x-upsampled threshold samples, but
    continuous rather than quantised to a quarter pixel -- and resampled every
    `spacing` px. Pixels within `band` of it get their exact distance to the
    nearest contour point from a k-d tree; the rest, whose distance only has
    to be roughly right, take a native-resolution distance transform. This is
    filter-agnostic: the 0.5 level of any symmetric pixel filter is the edge,
    which the box-coverage formula of Gustavson & Strand is not (0.22 px mean
    error on a blurred edge, against 0.06 here).

    CPU only (marching squares and the tree); a few seconds per 2048^2 view.
    """
    import numpy as np
    from scipy import ndimage as sndi
    from scipy.spatial import cKDTree
    from skimage import measure
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    inside = a >= 0.5
    coarse = (sndi.distance_transform_edt(~inside)
              - sndi.distance_transform_edt(inside)).astype(np.float32)
    pts = []
    for cnt in measure.find_contours(np.pad(a, 1), 0.5):
        cnt = cnt - 1.0
        seg = np.diff(cnt, axis=0)
        n = np.maximum(np.ceil(np.linalg.norm(seg, axis=1) / spacing), 1).astype(int)
        for (p, d, k) in zip(cnt[:-1], seg, n):
            t = (np.arange(k) / k)[:, None]
            pts.append(p + t * d)
        pts.append(cnt[-1:])
    if not pts:
        return coarse
    pts = np.concatenate(pts)
    rr, cc = np.nonzero(np.abs(coarse) < band)
    dist, _ = cKDTree(pts).query(np.stack([rr, cc], 1), workers=-1)
    out = coarse.copy()
    out[rr, cc] = np.where(inside[rr, cc], -dist, dist)
    return out
