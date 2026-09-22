# 3d — Cinder Basilisk

Reference-driven 3D creature pipeline. Stage 1 is the orthographic reference
sheet: six aligned views of one consistent creature, generated with a
**reference chain** so the design locks instead of drifting between views.

## The creature

**Cinder Basilisk** — hexapodal volcanic apex predator, ~4 m long, part
deep-sea crustacean part wingless dragon. Matte black basalt plating with
molten seams, seven-spine crown fan, four eyes per side, six legs, twelve-ring
counterweight tail ending in a basalt crystal cluster. Perfectly bilaterally
symmetric, held in a rigid neutral A-pose so the views stay modelable.

## Stage 2 — 6 views to one mesh

`docs/PLAN.md` is the design for the hard part: fusing the six views into a
single mesh without it exploding. Short version of the diagnosis — TSDF-style
depth fusion assumes many observations per surface point, and six orthographic
views give you one. The plan replaces statistical redundancy with constraints
(visual hull envelope, watertight implicit surface, bilateral symmetry),
treats per-view drift and per-view depth scale/shift as free parameters solved
jointly with the shape instead of as errors to eliminate, and splits the
frequency bands so surface detail never has to come out of a depth map.

## Two ways to run stage 1

- **Manual (Gemini app / subscription)** — `docs/PROMPT_PACK.md` has the
  copy-paste prompt blocks, the turn-by-turn chain order, the pass/fail count
  checklist and repair prompts for each drift symptom. This is the path to use
  when the API key has no image quota.
- **Scripted (API)** — `tools/gen_orthoviews.py`, described below. Needs an
  image-generation quota on the key; the free tier's daily allowance for
  `gemini-*-image` models runs out fast.

## Stage 1 — orthographic reference sheet

`tools/gen_orthoviews.py` generates, in order:

| # | view | references fed back in |
|---|------|------------------------|
| 1 | front  | — (this one is the master) |
| 2 | right  | front |
| 3 | back   | front, right |
| 4 | left   | front, right, back |
| 5 | top    | right, back, left (+front) |
| 6 | bottom | back, left, top (+right) |

Each request carries the previously accepted views as image inputs plus an
explicit "do not redesign anything" instruction. That is the whole trick:
view *n* is a **rotation** of views 1..n-1, not a fresh interpretation of the
text prompt.

The prompt is split into three fixed blocks so only one of them ever varies:

- `SUBJECT` — anatomy, locked counts (7 spines, 6 legs, 12 tail rings, 4 eyes
  per side), pose. Identical in every call.
- `VIEW` — the only part that changes: camera direction and what must be
  visible.
- `STYLE` — orthographic projection, flat even light, no cast shadows, empty
  #808080 background, one view per image, matched scale and alignment.

### Run it

```bash
cp .env.example .env        # put your key in .env — it is gitignored
export $(grep -v '^#' .env | xargs)

python3 tools/gen_orthoviews.py --out refs/cinder_basilisk --size 2K
```

Regenerate a single view without losing the chain:

```bash
python3 tools/gen_orthoviews.py --only 05_top
```

Kept views are re-read from disk and still fed forward as references, so one
bad frame costs one call instead of six.

`tools/contact_sheet.py` tiles the six PNGs into `refs/<name>/_sheet.jpg` for
a quick consistency check.

### Flags

| flag | default | meaning |
|------|---------|---------|
| `--out` | `refs/cinder_basilisk` | output directory |
| `--size` | `2K` | `1K` / `2K` / `4K` |
| `--only` | – | regenerate just this view |
| `--max-refs` | `4` | how many prior views to attach |

Model: `gemini-3-pro-image`, 1:1 aspect on every view so the six frames share
one square canvas and line up when loaded as background planes.

Retries: 429/500/503 back off at 2s, 4s, 8s, 16s.

## Secrets

The API key is read from `GEMINI_API_KEY` only. `.env`, `*.key` and `secrets/`
are gitignored — no key is ever committed.

## Ground truth: `tools/render_orthoviews.py`

The fusion stage is gated on reconstructing a *known* mesh, so the repo carries
its own renderer rather than relying on generated images to debug geometry
code. It takes any mesh and emits six axis-aligned orthographic views sharing
one camera scale and one centered object, so the frames are aligned by
construction:

```
refs/<name>/
  rgb/<view>.png        flat-lit clay render on a #808080 plate
  mask/<view>.png       exact silhouette
  depth/<view>.exr      orthographic depth, float32, normalized object units
  normal/<view>.exr     world-space normals, float32
  preview/              viewable depth + camera-space normal maps
  cameras.json          camera basis, ortho scale, normalization record
```

Lighting is a uniform white environment dome — even from every direction, no
cast shadows, and Cycles' global illumination supplies the crevice darkening
that makes form readable. Bounce count is deliberately low, because light
bouncing back out of crevices is what flattens a clay render.

```bash
assets/blender/blender -b -P tools/render_orthoviews.py -- \
    --mesh assets/lucy_le.ply --out refs/lucy_gt \
    --res 2048 --samples 128 --yaw 180

assets/blender/blender -b -P tools/inspect_gt.py -- --dir refs/lucy_gt
```

`inspect_gt.py` is the actual test: it measures each view's silhouette extents
and checks that the shared axes agree across views (front/back/top/bottom must
report one width, and so on), that depth stays inside the normalized bounding
box, and that normals are unit length. A view set that fails this is not worth
feeding to a reconstruction.

### The test subject

Stanford's **Lucy** — 14,027,872 vertices / 28,055,742 triangles, a full-body
winged figure. Limbs, wings, drapery folds and a raised arm give the deep
self-occlusion and concavity that a silhouette-based method is worst at, which
is the point. The distributed PLY is big-endian, which Blender 4.x will not
read, so `tools/ply_be2le.py` converts it and reports the bounding box.

Lucy is **not** bilaterally symmetric, so the symmetry constraint in the plan
is an option rather than a premise.

## Mirroring the left side

The creature is perfectly bilaterally symmetric, so the left side view is the
horizontal mirror of the right — generating it is strictly worse than flipping
it, because generation reintroduces drift:

```bash
magick refs/cinder_basilisk/02_right.png -flop refs/cinder_basilisk/04_left.png
```

The scripted pipeline generates it anyway (for comparison); the manual pack
tells you to flip instead.
