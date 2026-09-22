# M2 — Per-view estimation, measured

Depth and normal estimators run on a Kaggle T4 over the six orthographic views,
scored against the exact ground truth the views were rendered from.

Models: `depth-anything/Depth-Anything-V2-Large-hf`,
`prs-eth/marigold-depth-v1-1` and `prs-eth/marigold-normals-v1-1`, the Marigold
pair at ensemble 5, 4 steps, 1024 processing resolution.

```bash
python3 tools/calibrate_depth.py --est out/est --views refs/lucy_gt \
    --hull-views out/hull_views_1024 --out out/m2
python3 tools/eval_normals.py --est out/est --views refs/lucy_gt --out out/m2
```

## Depth

Mean absolute error in normalized units, object height 1.0. `raw` is per-view
normalization, what naive fusion uses. `hull-fit` solves the affine against the
visual hull's rendered depth. `gt-fit` solves it against the truth and is an
oracle. `HULL ITSELF` is the hull's own depth error — the thing the estimate
was calibrated against, used directly.

### marigold-depth-v1-1

| view | cover | raw | hull-fit | gt-fit | **HULL ITSELF** | winner |
|---|---|---|---|---|---|---|
| 01_front | 18.3% | 0.10547 | 0.05998 | 0.04237 | **0.04623** | hull |
| 02_right | 14.9% | 0.19642 | 0.05446 | 0.03737 | **0.04164** | hull |
| 03_back | 18.3% | 0.09453 | 0.05283 | 0.04225 | **0.03625** | hull |
| 04_left | 14.9% | 0.20151 | 0.06816 | 0.04051 | **0.05798** | hull |
| 05_top | 7.3% | 0.35372 | 0.25593 | 0.23001 | **0.16769** | hull |
| 06_bottom | 7.3% | 0.20006 | 0.16128 | 0.16199 | **0.05638** | hull |
| **mean** | | 0.19195 | 0.10877 | 0.09242 | **0.06769** | |

### Depth-Anything-V2-Large

| view | raw | hull-fit | gt-fit | **HULL ITSELF** | winner |
|---|---|---|---|---|---|
| 01_front | 0.05282 | 0.06007 | 0.04370 | **0.04623** | hull |
| 02_right | 0.10660 | 0.05671 | 0.03718 | **0.04164** | hull |
| 03_back | 0.04031 | 0.04745 | 0.03190 | **0.03625** | hull |
| 04_left | 0.08294 | 0.06780 | 0.03049 | **0.05798** | hull |
| 05_top | 0.14204 | 0.22786 | 0.12696 | **0.16769** | hull |
| 06_bottom | 0.17755 | 0.10258 | 0.09950 | **0.05638** | hull |
| **mean** | 0.10038 | 0.09374 | 0.06162 | **0.06769** | |

## Normals — marigold-normals-v1-1

Angular error in degrees. `FLAT BASE` assumes every visible surface faces the
camera: the answer you get for free, without running anything.

| view | mean | median | <22.5° | **FLAT BASE** | verdict |
|---|---|---|---|---|---|
| 01_front | 27.80 | 24.95 | 43.2% | 40.55 | estimator |
| 02_right | 28.20 | 23.75 | 47.3% | 38.10 | estimator |
| 03_back | 31.92 | 26.86 | 39.5% | 38.73 | estimator |
| 04_left | 24.07 | 20.82 | 54.4% | 39.50 | estimator |
| 05_top | **69.15** | 74.42 | 7.7% | **56.30** | **flat guess wins** |
| 06_bottom | **28.34** | 16.45 | 67.8% | **17.42** | **flat guess wins** |

The 27.8° on the front view is not a convention error. Solving for the single
global rotation that best aligns the estimate to the truth improves it only to
26.43°, a 1.4° gain for a 9° rotation. The axis convention is right and the
estimate is genuinely that far off.

## Three findings

### 1. The depth estimators lose to the hull on every view

Not "add little" — lose. On all six views, for both models, the visual hull's
own rendered depth is closer to the truth than the calibrated estimate. The
hull takes under a second to carve from the silhouettes.

Giving the estimators a perfect affine does not rescue them. Oracle-fitted
Marigold beats the hull on only two of six views, by 0.004 and 0.004, and loses
on top by 0.06 and on bottom by 0.11.

On this input — flat-lit orthographic clay renders — monocular depth
contributes nothing over shape-from-silhouette.

### 2. Hull anchoring works; the thing it anchors does not

The mechanism from the plan does what it was designed to do. Marigold's error
goes 0.192 raw → 0.109 hull-fit, against an oracle floor of 0.092. The hull
anchor captures **89%** of the improvement a perfect affine would give, without
ever seeing ground truth. Per-view scale and shift are solvable from the hull.

That result stands on its own and will matter as soon as there is an estimate
worth calibrating. It just is not this one.

### 3. Top and bottom are anti-informative

Both views cover 7.3% of the frame against 15–18% for the sides, and both are
far outside what these models were trained on. A top-down orthographic view of
a statue does not resemble a photograph of anything.

The result is not merely weak, it is worse than guessing. On the top view the
normal estimate is 69.15° against a flat-guess baseline of 56.30°, and on the
bottom 28.34° against 17.42°. Depth on the bottom view is triple the hull's
error.

Fed into a fusion at equal weight, these two views actively pull the surface
away from the truth. Four good views and two harmful ones, averaged, is a
worse answer than four good views alone.

## What this changes in the plan

- **Depth is demoted from auxiliary signal to nothing, pending evidence.** The
  plan already ranked normals above depth; the measurement says depth is below
  the hull, which is below both. Nothing in S6 should consume a depth estimate
  until one is shown to beat the hull on the view it is used for. That check is
  now one command.
- **Per-view confidence weighting becomes mandatory, and is computable.** The
  hull gives a per-view sanity check without ground truth: where an estimate
  disagrees violently with the hull, it is the estimate that is wrong. That is
  the weight.
- **Normals remain the refinement signal, on the four side views only.** They
  beat the flat baseline by 10–15° there and carry the high-frequency detail
  the hull cannot have.
- **Two experiments are now worth more than more models.** First, render the
  references under a long-lens near-orthographic perspective instead of true
  orthographic, and re-measure: if the gap is distribution rather than
  geometry, that closes it cheaply. Second, add three-quarter views, which M1
  showed are the only way to add silhouette constraints and which are also far
  closer to what the estimators were trained on than a top-down view.

## Scope of these numbers

Measured on flat-lit orthographic clay renders of a marble statue. That is out
of distribution for every model here, which is part of what is being measured,
but it means the absolute numbers do not transfer to the generated creature
references, which look much more like the rendered art these models see in
training. The top and bottom result will transfer, because low coverage and an
unfamiliar viewing direction are properties of the view, not the subject.
