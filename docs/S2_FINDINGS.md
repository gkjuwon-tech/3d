# S2 — first results, and where the design is wrong

Gate test, one view, ground-truth normals, resolution 512. Hull baseline
MAE 0.01985.

| gradients | formulation | MAE | vs hull |
|---|---|---|---|
| exact (from GT depth) | rim equality anchor | 0.02086 | −5% |
| exact | **inequality floor** | **0.01114** | **+43.9%** |
| GT normals | rim equality anchor | 0.01899 | +4% |
| GT normals | inequality floor | 0.01765 | +11.1% |

## What was verified

**The solver mechanics are correct.** Fed the numerical gradient of the true
depth, it improves on the hull by 43.9%. The sign convention was confirmed
exhaustively: of eight axis/sign combinations only `d_x = n_x/n_z`,
`d_row = −n_y/n_z` correlates positively on both axes, and swapping x and y
gives correlation 0.00.

**One unit bug, found and fixed.** `n_x/n_z` is depth change per unit of world
distance; the solve's finite differences are per pixel. Missing the pixel size
inflated every gradient by the resolution — 465× at res 512 — and produced an
MAE of 2.88 against a hull baseline of 0.02.

## What was refuted: the rim anchor

The plan's central premise was that the visual hull touches the true surface
along each silhouette's boundary, giving a free Dirichlet condition.

Measured, at full resolution, front view:

```
hull depth error on the rim (6 px)   0.02310
hull depth error in the interior     0.01941
hull within 0.002 of truth           5.29% of pixels
hull ever in front of truth          0.000%   <- the bound itself holds
```

**The rim is the worst place to anchor, not the best.** The reason is
geometric: near a silhouette the surface is nearly parallel to the view ray, so
a small 3D offset becomes a large depth offset. The hull is conservatively
dilated by one voxel, and at the rim that voxel turns into 0.023 of depth
error. Pinning the boundary to a value wrong by 0.023 caps the whole solve at
0.023 — which is exactly what feeding it *exact* gradients produced.

What the hull actually is, verified on every pixel, is a **lower bound**: its
depth never exceeds the true depth. So it now enters as an inequality with a
weak pull rather than an equality, and the answer settles as close to the hull
as the gradients allow. That change alone took exact-gradient performance from
−5% to +43.9%.

## What is still wrong: offsets on a fragmented domain

With exact gradients the shape is exact and only the offset is free, so the
floor should catch the surface at one or two points. Instead **68.6% of pixels
end up pinned to it.**

The cause is fragmentation. Pixels where the surface turns edge-on are dropped
from the gradient term, and those dropped pixels cut the domain into many
pieces that do not share a gradient path. Each piece then needs its own offset,
and the only thing offering one is the hull floor — which is 0.02 too near.
Each fragment settles 0.02 too near, and per-view integration cannot do better,
because within a single view there is nothing left to determine those offsets.

With estimated normals the effect is worse: the `|n_z| > 0.15` cut fragments
the domain further, and the result lands between +1% and +11%.

## The design consequence

**Per-view integration in isolation cannot solve this, and no amount of
tuning will change that.** A fragment's offset is not observable from the view
it lives in. It is observable from the *other* views that see the same 3D
region.

That is the original plan's principle 2 arriving from a new direction: the
per-view unknowns have to be solved jointly across views, not fixed one view at
a time and fused afterwards. Concretely, the shape term stays per-view and
local — gradients from normals, which work — while the per-fragment offsets
become global unknowns tied together by 3D consistency.

Two ways to get there, in increasing order of work:

1. **Carve with relative depth only.** Use each view's integrated surface for
   its *shape* and let the carve's agreement rule resolve the offset: a voxel
   is removed when enough views place the surface behind it. Views that
   disagree about absolute depth still agree about relief.
2. **Joint solve.** Per-fragment offsets as explicit unknowns, with equations
   from every pair of views that observe a shared 3D point. Bigger, and the
   right end state.

Option 1 reuses the carving machinery already built and is the next step.
