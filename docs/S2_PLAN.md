# S2 — Carving the hull down to the surface

The hull is a floor: it contains the truth and never fails, and it is twice too
fat. This is the stage that removes the fat. Everything before it exists to
make this step safe.

## What we have

- A watertight voxel grid whose surface provably **contains** the true one
- **14 cameras we know exactly** — we rendered them
- Per view: an exact silhouette, and an estimated surface normal per pixel
- Ground truth, so every step can be scored

The second item deserves emphasis. M2b's failures were all registration:
VGGT stacked every view in one place, DA3 put the bottom view on the far side
of the object. We never have to ask a network where a camera is. That entire
failure mode is out of scope by construction.

## The idea in one line

**A normal map says which way the surface tilts. Integrate the tilt and you get
the surface — up to an unknown offset. The silhouette rim supplies the offset.**

## Why the rim is the anchor

The visual hull is the intersection of the silhouette cones, so along the
boundary of each silhouette **the hull surface touches the true surface**. That
is not an approximation; it is what a silhouette means. The contour generator —
the curve on the true object that projects to the silhouette edge — lies on
both surfaces.

So at the rim, hull depth **is** true depth. That is a Dirichlet boundary
condition, handed to us by geometry, for free. Integration inward from a known
boundary is a solved, stable problem.

Related: Cao et al. 2022, *Bilateral Normal Integration*; Xiu et al. 2023,
*ECON*, which integrates front and back normal maps anchored on a depth prior.
This is that method with fourteen views and a hull for the prior.

## Orthographic helps here

Orthographic projection wrecked the pose heads in M2b. In integration it is the
opposite: for a surface `z = d(x, y)` under an orthographic camera, the
relation between normal and depth gradient is exact and linear.

```
∂d/∂x = n_x / n_z
∂d/∂y = n_y / n_z
```

No focal length, no per-pixel ray direction, no foreshortening correction. Under
perspective the same relation carries position-dependent terms. The projection
that cost us in one place pays us back here.

(The sign convention differs between estimators. We fix it once against the
ground-truth normals, which we have, and record it.)

## The per-view solve

For each view, find a depth field `d` minimising

```
  w_grad · ‖ ∇d − g ‖²            g = (n_x/n_z, n_y/n_z), the measured tilt
+ w_anchor · ‖ d − h ‖²           h = the hull's rendered depth
+ w_smooth · ‖ ∇²d ‖²
```

subject to

```
  d ≥ h                  never in front of the hull
  d ≤ h + Δ_max(view)    a carving budget
  d = h  on the rim      the anchor that fixes the offset
```

That is a linear least-squares problem on a grid — a Poisson solve. It has one
answer, it is found directly, and it does not diverge. No learning rate, no
step count, no "it exploded at iteration 4000".

Three details that matter:

**Grazing pixels.** Where the surface turns edge-on, `n_z → 0` and the gradient
blows up. Those pixels get dropped from the gradient term and the solve
interpolates across them. Standard, and the hull anchor keeps the gap honest.

**The anchor is everywhere, not only at the rim.** `w_anchor` is large at the
rim and small but non-zero in the interior. Normal integration accumulates
error with distance from its boundary, so a wide flat region far from any
silhouette edge would drift without it. This is the depth-aware part of d-BiNI.

**The budget is per view.** M2b measured which views produce usable normals:
the four side views beat a flat guess by 10–15°, the top and bottom lose to it.
Those measurements become `Δ_max` directly. A view that failed its baseline
gets `Δ_max = 0` and carves nothing. The bad views are not argued with, they
are switched off.

## Fusion: how the views combine

Silhouettes and depths must vote differently, and getting this wrong is how the
guarantee dies.

**A silhouette is a veto.** If one view says a voxel is outside the object, it
is outside — that is certain, and one view is enough. Unanimity not required.

**A depth estimate is an opinion.** If one view says "the surface is deeper
here", that is a guess from a network that was wrong by 69° on the top view. A
single opinion must not be able to hollow out the model.

So: a voxel is carved when **enough of the views that can see it agree**. In
practice, take the k-th most conservative opinion among the confident views
rather than the most aggressive one. One bad view then costs nothing; it takes
a conspiracy.

## The loop

```
hull
  ├─ render its depth from every view          (exact — it is our mesh)
  ├─ integrate normals, anchored on that depth  (per view, independent)
  ├─ carve where enough views agree             (voxels only ever leave)
  └─ tighter hull ──> repeat
```

Each pass the anchor is closer to the truth, so the next integration starts
from a better boundary. A shrink-wrap that tightens a little at a time, rather
than one solve that has to be right immediately.

Convergence is by construction: occupancy is monotone decreasing and bounded
below by the true shape as long as the budget is honest.

## Why this cannot explode

**Voxels only ever leave.** There is no code path that adds one.

- A spike growing out of the surface needs material that does not exist
- The mesh cannot fly away; it cannot leave the hull it started inside
- Holes cannot open; carving a closed volume leaves a closed volume
- Divergence has nothing to diverge — a Poisson solve is a direct solve

The single failure mode is over-carving, and it is bounded by `Δ_max`, which is
set per view from measurements we already have. The worst case is a model
slightly too thin, reported as containment dropping below 100%.

Compare the original S6: optimise a neural SDF by gradient descent under six
losses. More expressive, and it can fail in every way this cannot.

## Scoring, every step

We have the answer, so nothing is taken on faith.

| number | what it catches |
|---|---|
| containment | carved too deep |
| Chamfer | whether it is actually getting closer |
| volume ratio | how much fat is left (1.443× now, 1.0 is truth) |
| per-view ablation | which views help, measured rather than assumed |

That last one is cheap and worth running first: carve with **one view at a
time** and score each. It turns "which views should we trust" from an argument
into a table.

## Stage 3 — the fine detail

Scales, hairline cracks, pore-level texture: do not carve these as geometry.
They cost triangles by the million and add nothing a renderer needs as
geometry. Bake them from the image gradients into a normal and displacement
map on the carved mesh. This is the frequency split from the original plan, and
it is how games and film have done it for twenty years.

The carve handles what the silhouette and the normal integration can reach.
The bake handles everything finer.

## Order of work

1. Fix the sign convention against ground-truth normals, and measure how good
   the estimated normals are *in gradient space* rather than in degrees —
   gradient error is what the solve actually consumes.
2. Single view, single pass, largest budget. Does the carved depth beat the
   hull depth on that view? If not, stop and fix that before touching the grid.
3. Per-view ablation across all 14, to set `Δ_max` from data.
4. Full fusion, one pass. Score.
5. Iterate the loop until Chamfer stops falling.
6. Bake detail.

Step 2 is the gate. If a single view's integrated depth is not better than the
hull's own depth on that view, nothing downstream can help.
