# M2c — Adding three-quarter views

M1 and M2b converged on one change from two directions. M1: the canonical six
orthographic views supply three silhouette constraints, and a diagonal view is
the only way to add another. M2b: DA3 placed the bottom view 161.5° wrong and
VGGT stacked all six views in one place, which looked like a lack of overlap
between views 90° and 180° apart.

So: eight auxiliary views in the middle of each octant, azimuth 45/135/225/315
at elevation ±45. Fourteen views total. Three things were expected of them.

| purpose | result |
|---|---|
| silhouette constraints | **confirmed** — Chamfer halved |
| registration overlap | **refuted** for VGGT; DA3 pending |
| in-distribution input | visually clear, pending measurement |

## 1. Silhouette constraints — confirmed

| | 6 views | 14 views |
|---|---|---|
| occupied voxels | 5,402,974 | 3,973,788 |
| **Chamfer** | 0.020579 | **0.010590** |
| volume ratio vs truth | 2.007× | **1.476×** |
| containment | 100.0000% | 100.0000% |
| silhouette IoU | 0.964 | 0.961 |
| carve time | 0.8 s | 6.1 s |

Halving the reconstruction error is a larger gain than any depth model measured
in M2 or M2b produced, from eight renders and six seconds of carving, with the
outer-bound guarantee intact.

The comparison render shows what changed. The six-view hull is a slab with the
wings fused to the body and no arm. The fourteen-view hull separates the wings
from the torso, recovers the left arm's silhouette, and reads as two distinct
wings from above — the "limbs webbed together" failure M1 predicted, reduced.

It is still not the truth and never will be: no face, no drapery. That ceiling
belongs to shape-from-silhouette.

### The redundancy law holds in a new view family

```
07_az45_up    8.1% removed      11_az45_dn    0.0%
08_az135_up   6.4% removed      12_az135_dn   0.0%
09_az225_up   7.6% removed      13_az225_dn   0.0%
10_az315_up   7.3% removed      14_az315_dn   0.0%
```

Each downward diagonal is the opposite of an upward one and carves nothing,
exactly as back, left and bottom did among the canonical six. **Eight diagonal
views supply four independent constraints**, bringing the total to seven.

Practically, silhouette coverage only needs one hemisphere: render the upper
four and the lower four come free. Both were rendered here because
registration and appearance may still want them — and that turned out to
matter, see below.

## 2. Registration overlap — refuted for VGGT

| | 6 views | 14 views |
|---|---|---|
| centroid spread across views | 0.2206 | 0.2391 |
| single view's extent | 1.28 | 1.61 |
| gt surface covered | 4.28% | 7.00% |
| Chamfer of the joint point map | 0.1738 | 0.1705 |

All fourteen centroids still fall within 0.24 of each other. VGGT stacks them
in one place exactly as before.

This refutes the explanation M2b gave. A 45° diagonal shares plenty of surface
with both the front and the right view, so the overlap VGGT supposedly lacked
was supplied, and nothing changed.

The remaining explanation is the projection. VGGT predicts intrinsics and
extrinsics under a pinhole model, and an orthographic image has no focal
length, no vanishing point and no perspective cue, so its pose head has no
problem to solve. DA3 managing five of six on the same input does not
contradict this — it solves poses relative to a chosen reference view and
reports scale separately, so its dependence on the camera model differs.
Orthographic tolerance is a per-model property.

Testing that costs one more render pass under a long lens, and is the last
experiment left from the original pair.

## 3. In-distribution input — pending

Visually unambiguous: the diagonal views look like photographs of a statue,
which is the standard framing for photographing sculpture, while the top-down
orthographic view is an unidentifiable blob. The measurement is DA3's
registration error and depth on the extended set.

## Status

DA3 on fourteen views needs about 20 GiB against the T4's 14.56, since
attention across views is quadratic in view count. It runs down a ladder —
large backbone at fourteen views, base at fourteen, base at ten, base at eight
— under fp16, and the rung that produced a result is recorded so a reduced-set
number is never reported as a fourteen-view one.

Two bugs were fixed getting there, both mine:

- The ladder's cleanup deleted an alias taken from `locals()` rather than
  rebinding the name, so every backbone stayed resident and three accumulated.
  Free memory fell 11.77 → 11.24 → 5.04 GiB across the rungs while the
  requests were 19.65, 14.74 and 7.52 — the last would have fit on an empty
  card.
- The reduced sets dropped all four downward diagonals, on the grounds that
  they carve 0.0%. True for silhouettes, and irrelevant here: those are exactly
  the views that bridge the ring to the bottom view, whose registration is what
  the run exists to measure. The fallback was removing the thing being
  measured. Reduced sets now halve by azimuth so a bridge to each pole
  survives.
