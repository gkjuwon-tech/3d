"""Scale-by-scale normal error: which frequencies of a normal map are right.

band_errors(n, nt, m): angle error of n vs truth nt after both are blurred at
several scales (low frequencies), and the error of the detail layer
(n - blur(n) vs nt - blur(nt)) as a correlation per band.
"""
import numpy as np
from scipy import ndimage


def blur_n(n, m, sig):
    if sig == 0:
        return n
    w = ndimage.gaussian_filter(m.astype(float), sig)
    b = np.stack([ndimage.gaussian_filter(n[..., c] * m, sig) / np.maximum(w, 1e-6)
                  for c in range(3)], -1)
    return b / np.linalg.norm(b, axis=-1, keepdims=True).clip(1e-9)


def ang(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def band_report(n, nt, m, sigs=(0, 2, 4, 8, 16, 32)):
    out = []
    for s in sigs:
        mm = ndimage.binary_erosion(m, iterations=max(1, int(2 * s)))
        e = ang(blur_n(n, m, s)[mm], blur_n(nt, m, s)[mm])
        out.append((s, np.median(e), (e < 5).mean() * 100))
    # detail layers: what lies between scale s and 2s
    det = []
    for s in (1, 2, 4, 8):
        mm = ndimage.binary_erosion(m, iterations=int(4 * s) + 1)
        a = blur_n(n, m, s) - blur_n(n, m, 2 * s)
        b = blur_n(nt, m, s) - blur_n(nt, m, 2 * s)
        a, b = a[mm][:, :2].ravel(), b[mm][:, :2].ravel()
        det.append((s, np.corrcoef(a, b)[0, 1], np.std(a) / max(np.std(b), 1e-9)))
    return out, det


def print_report(label, n, nt, m):
    out, det = band_report(n, nt, m)
    print(f"{label}")
    print("  blurred at  " + "  ".join(f"s{s:<2d} {e:5.1f}deg" for s, e, _ in out))
    print("  detail band " + "  ".join(f"{s}-{2*s}px r={c:.2f} amp={r:.2f}" for s, c, r in det))
