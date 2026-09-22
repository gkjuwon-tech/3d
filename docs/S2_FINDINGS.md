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


---

# Round two: relief carving, and what it measured

The plan after the gate test was to stop asserting absolute depth per view, use
each view's *relief* only, and let the carve's agreement rule resolve the
offsets. Implemented in `tools/carve_relief.py`. It does not work on this
subject, and the reason is now measured rather than suspected.

## Two bugs found first, one dangerous

**The unknown case failed open.** Where no estimate existed the depth was set
to `+inf`, which reads as "the surface is infinitely far back" and carved the
entire grid. Absence of information now carves nothing.

**Frames were mismatched.** The rendered images are square and span the
camera's ortho_scale; the grid view spans the voxel grid's extent. Resampling
one onto the other by pixel index pasted background over the subject, which is
why the first run found zero usable gradient in every view. Sampling now goes
through world coordinates.

## What the working code measures

Front view, ground-truth normals, one fragment covering 96.5% of the mask:

```
placed surface - true surface:  mean +0.1290
                                5%   +0.0126
                                95%  +0.1852
behind the truth (over-carving): 96.3% of pixels
```

The placed surface sits 0.13 behind the truth — 13% of the object's height.
A single offset cannot fix it because the relief itself has drifted: the
integrated shape is wrong over long distances, so no constant makes it fit.

## Why the relief drifts

Measured at the gate: the correlation between the gradient implied by
ground-truth normals and the actual gradient of ground-truth depth is **0.16
per pixel**, rising to 0.62 only under heavy smoothing.

The cause is scale. Lucy carries 28 million triangles across roughly 770,000
covered pixels — **36 triangles per pixel**. The depth difference between
adjacent pixel centres is a secant across 36 facets; the normal is one facet's.
They describe different things, and integrating a gradient that is 40% wrong
accumulates over a thousand pixels into the 0.13 seen above.

## Band-limiting does not rescue it

If the disagreement were pure high-frequency noise, smoothing the normals
before integrating should help. It does the opposite:

| normal blur | best MAE | vs hull | over-carve |
|---|---|---|---|
| none | 0.01765 | **+11.1%** | 41.8% |
| σ 2 | 0.01886 | +5.0% | 43.2% |
| σ 4 | 0.01920 | +3.3% | 43.2% |
| σ 8 | 0.01967 | +0.9% | 42.8% |

And in every configuration the estimate lands behind the truth on roughly 42%
of pixels. **Carving to a surface that is over-deep on 42% of its area removes
real object**, which is why containment collapsed from 100% to 0.2% the moment
the carve was wired up correctly.

## Where this leaves S2

Normal integration on this subject yields at most an 11% depth improvement, and
is not safe to carve with at any setting tried. The best result came from the
strongest anchor, which means the hull was doing the work and the normals were
a small correction — not the other way round.

Two readings, and they are distinguishable by experiment:

1. **The method is wrong for this data.** 36 triangles per pixel is extreme.
   Generated creature images are band-limited by the generator, so their
   normals and their depth would describe the same surface. The test is to run
   the same pipeline against a decimated, smoothed Lucy — band-limiting the
   *subject* rather than the normals — and see whether integration starts
   working.
2. **Per-view integration is the wrong primitive.** Carve by cross-view
   agreement without ever forming a per-view depth: a voxel goes only when
   several views independently place the surface behind it. One view being
   wrong 42% of the time matters much less when three must agree. This is the
   plan's majority-vote rule applied earlier, before depth is committed.

Reading 1 is cheap and decides whether the subject is the problem. Reading 2 is
the more robust design regardless of the answer.


---

# Round three: the actual blocker was depth discontinuities

Both readings offered at the end of round two were wrong, and saying so is the
point of writing them down. Band-limiting Lucy would have tuned the subject
until the method passed. Cross-view voting only fixes *independent* errors, and
every view's integration drifts the same way from the same kind of normals, so
a vote would have ratified the bias.

## The reference was the broken thing

The whole diagnosis rested on one statistic: the correlation between the
gradient implied by the normals and the gradient of the true depth, measured at
0.15. That was read as "the normals are bad". It is not what it means.

| exclusion | edges kept | gx correlation | RMS ratio |
|---|---|---|---|
| none | 100% | +0.150 | 0.119 |
| drop \|Δd\| > 20× median | 99.1% | **+0.947** | 0.928 |
| drop \|Δd\| > 8× median | 98.9% | **+0.984** | 0.981 |
| drop \|Δd\| > 4× median | 96.5% | +0.989 | 1.007 |

**Removing 1% of pixels moves the agreement from 0.15 to 0.98.** The normals
were never the problem. About 2% of adjacent-pixel pairs sit across a *depth
discontinuity* — a fold passing in front of the body, a wing crossing a
shoulder — where the two pixels are on different surfaces entirely. Their depth
difference has nothing to do with any local slope, and it dominated every
variance statistic and injected a false jump into every integration path that
crossed it.

Two hypotheses died getting here, both mine:

- *36 triangles per pixel makes the surface sub-pixel noise.* No: the local
  depth standard deviation in a 3×3 window is 0.00037, against a hull error of
  0.0196. The surface is smooth at pixel scale. Median slope from depth is
  0.59, from normals 0.58 — they agree to 2%.
- *The normal pass is smooth-shaded while depth is geometric.* Re-rendered one
  view flat-shaded: the numbers are identical to four decimal places.

## Finding the jumps without an answer key

Ground truth marks them; production cannot. Detectors, scored against the true
discontinuities:

| detector | recall | precision | edges cut |
|---|---|---|---|
| hull depth jump > 4× | 5.5% | 5.0% | 2.0% |
| image edge, top 1% | 23.0% | 42.7% | 1.0% |
| image edge, top 3% | 42.6% | 26.4% | 3.0% |
| \|n_z\| < 0.2 | 34.7% | **58.5%** | 1.1% |
| \|n_z\| < 0.35 | **57.3%** | 23.7% | 4.5% |

The hull's own depth is nearly useless for this. Grazing normals and image
edges both work, and they fire on different jumps.

## What that buys

Front view, ground-truth normals, against a hull baseline of 0.01985:

| n_z cut | image-edge cut | MAE | vs hull |
|---|---|---|---|
| 0.15 | none | 0.01765 | +11.1% |
| 0.15 | top 3% | 0.01680 | +15.4% |
| 0.15 | top 7% | **0.01567** | **+21.0%** |
| 0.35 | top 7% | 0.01576 | +20.6% |

Cutting the integration graph at image edges nearly doubles the gain, from
+11% to +21%, by refusing to integrate across boundaries the normals never
described.

## Still not safe to carve with

The estimate lands behind the truth on 38% of pixels, down from 42% but not
near zero, and carving to an over-deep surface removes real object. The lever
is recall: the best detector finds 57% of the jumps, so 43% of them still leak
into the solve.

That is the next thing to work on, and it is a well-posed problem with a
scoreboard — detector recall against the measured discontinuity set, and MAE
against the hull — rather than another guess about what might be wrong.
