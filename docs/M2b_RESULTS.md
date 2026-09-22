# M2b — Swapping the estimator, and what multi-view actually bought

M2 measured one diffusion depth model and concluded that depth estimation
loses to the visual hull. That conclusion judged a class from a single member
and is withdrawn here. It also concluded that the top and bottom views are
anti-informative as a property of those views; that is a property of
*monocular* estimation of those views, and is corrected below.

Four estimators, same six views, same metrics.

## Depth, oracle-calibrated

Mean absolute error after fitting a per-view affine against the **truth** —
the estimator's own quality, with the scale/shift problem removed. Units where
the object is 1.0 tall.

| view | Marigold | MoGe-2 | DA3-LARGE | VGGT-1B | hull |
|---|---|---|---|---|---|
| | *mono, diffusion* | *mono* | *multi-view* | *multi-view* | |
| 01_front | 0.0424 | 0.0361 | **0.0178** | 0.0191 | 0.0462 |
| 02_right | 0.0374 | 0.0203 | 0.0200 | **0.0181** | 0.0416 |
| 03_back | 0.0423 | 0.0421 | **0.0277** | 0.0441 | 0.0363 |
| 04_left | 0.0405 | **0.0273** | 0.0334 | 0.0514 | 0.0580 |
| 05_top | 0.2300 | **0.0778** | 0.1494 | 0.1287 | 0.1677 |
| 06_bottom | 0.1620 | 0.1171 | 0.0913 | **0.0399** | 0.0564 |
| **mean** | 0.0924 | 0.0535 | 0.0566 | **0.0502** | 0.0677 |

## Three findings

### 1. It was the model

Marigold sits at 0.0924 while three newer models cluster at 0.050–0.057, and
all three beat the hull's 0.0677. M2's "depth loses to the hull" was "Marigold
loses to the hull". On the front view the best estimator is 2.6× more accurate
than the one M2 measured.

### 2. Multi-view conditioning is not what fixed it

The control was there to separate "multi-view helps" from "a better model
helps", and it came down on the second. The best top-view result, 0.0778,
belongs to **monocular** MoGe-2; both multi-view models do worse on that view.
Monocular MoGe-2's overall mean, 0.0535, is within noise of multi-view DA3's
0.0566.

One confound, stated rather than argued away: VGGT processes at 518² and DA3 at
504², against MoGe-2's higher working resolution. On the top view the subject
covers 7.3% of the frame, so the multi-view models are resolution-starved
exactly where they are being asked to do their hardest work.

### 3. Registration is the precondition, and it is where this breaks

A multi-view model can only read the top view in the context of the side views
if it first places those views correctly. Both models were asked directly.

**VGGT failed completely.** The six per-view point clouds have centroids within
0.22 of each other while a single view spans 1.3 — six frontal shells stacked
in one place. Its joint point map scores Chamfer 0.1738 against the hull's
0.0198, and the asymmetry names the cause: predicted points lie 0.0328 from the
true surface, but the true surface lies 0.3148 from any predicted point and is
covered at only 4.28%. What it predicts is roughly right; it predicts almost
nothing. Its per-view depth was nonetheless the best of everything tested, so
this is a pose failure, not a depth failure.

**DA3 got five of six.** Registration error per view, after the best global
rotation onto the true camera axes:

| view | error |
|---|---|
| 01_front | 5.9° |
| 02_right | 5.9° |
| 03_back | 7.8° |
| 04_left | 10.8° |
| 05_top | 4.2° |
| **06_bottom** | **161.5°** |

It placed the bottom view where the top view belongs. The angle it predicts
between top and bottom is 14.6°; the true angle is 180°.

### The root cause is the one M1 already found

Under orthographic projection, opposite views have **identical silhouettes**.
M1 measured this as redundancy — back, left and bottom each carved away 0.0–0.1%
beyond their opposites, so six views supply three silhouette constraints.

The same property makes opposite views ambiguous to register. Only appearance
can break the tie, and appearance is weakest exactly where DA3 failed: a
top-down and a bottom-up orthographic view of this subject are both foreshortened
blobs with wings, at 7.3% coverage, where front and back are separated easily by
a face against the back of a head.

One geometric fact, two symptoms: the views that carve nothing are the views
that cannot be placed.

## The hull anchor has become the bottleneck

M2 validated anchoring the per-view affine on the hull: Marigold went 0.192 raw
to 0.109 hull-fit against a 0.092 oracle floor, capturing 89% of the available
gain. With a better estimator that margin inverts. VGGT is 0.0767 hull-fit
against 0.0502 oracle — a 0.0265 penalty, against Marigold's 0.0164.

The hull is twice the true volume (M1). Fitting an accurate estimate onto it
drags the estimate toward the fat. The anchor was the right idea while the
estimator was the weak link; now the estimator is better than the anchor.

**Plan change:** the affine stays a free parameter solved jointly with the
shape, as S6 always specified, rather than fitted against a fixed hull first.
The hull keeps its other two jobs — the hard envelope, and the per-view sanity
check that needs no ground truth.

## What to do next

Elevated three-quarter views now serve three separate purposes at once:

1. **Silhouette constraints.** M1: the canonical six give three, and a diagonal
   view is the only way to add one.
2. **Registration overlap.** A view between the ring and a pole shares surface
   with both, which is what DA3 lacked when it put the bottom view on top of
   the top view.
3. **In-distribution input.** A three-quarter view resembles a photograph far
   more than a top-down orthographic projection does.

Two experiments that were proposed separately have converged on one change.

## Corrections made while measuring

- Fixing the scale in the Umeyama solve computed rotation and translation for
  scale 1 and then applied a different scale, so the point-map alignment
  diverged instead of converging. Translation now uses the scale being applied.
- MoGe marks regions it declines to predict with NaN, which was turning every
  error statistic into NaN. Declined pixels are excluded and their fraction
  reported.
- Kernel package installs now run before numpy is imported; installing a
  package that pulls a different numpy afterwards left scipy failing on an ABI
  mismatch unrelated to any model.
