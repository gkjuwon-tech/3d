# M1 — Visual hull floor

Stage 1 of the reconstruction, measured against the mesh the views were
rendered from. Subject: Stanford Lucy, 28,055,742 triangles, six orthographic
views at 2048².

```bash
python3 tools/visual_hull.py --views refs/lucy_gt --out out/lucy_hull --res 512
python3 tools/eval_hull.py --gt-mesh assets/lucy_le.ply --views refs/lucy_gt \
    --recon out/lucy_hull.ply --recon-views out/hull_views \
    --occ out/lucy_hull_occ.npz
```

## Result

| | res 512 | res 1024 |
|---|---|---|
| grid | 319 × 192 × 533 (32.6M) | 638 × 384 × 1065 (260.9M) |
| occupied voxels | 5,402,974 | 42,341,529 |
| mesh | 783,008 tris | 3,132,272 tris |
| carve time | 0.8 s | 7 s |
| **containment** | **100.0000%** | **100.0000%** |
| silhouette IoU | 0.96428 | 0.97665 |
| Chamfer (symmetric mean) | 0.020579 | 0.019788 |
| volume ratio vs truth | 2.007× | 1.966× |

Chamfer and containment are over 400,000 points sampled per surface, in units
where the object's longest axis is 1.0.

**The floor is established.** There is now a watertight mesh that provably
contains the true surface, takes under a second to produce, and cannot fly
apart no matter what the inputs do. Every later stage refines inside it.

## Three findings

### 1. Conservative voxelization is what makes the bound real

Marking a voxel occupied when its **centre** projects inside every silhouette
leaves **99.032%** containment — 1% of the true surface falls outside the
result. The hull is then no longer an outer bound, and the guarantee the whole
plan leans on is gone.

A voxel spans about 3.6 pixels at res 512, so its centre is not a proxy for
the voxel. Dilating each silhouette by the voxel footprint before the lookup
makes the test conservative — a voxel survives if *any* part of it projects
inside — and containment becomes **100.0000%** on all 400,000 sampled points.
The cost is one voxel of fat, which is what the silhouette IoU drop from 0.981
to 0.964 is measuring. That is the correct trade: the bound is the point.

### 2. Six orthographic views carry only three silhouette constraints

Carving with all six, in order:

```
01_front    32,645,184 -> 11,591,424   (64.5% removed)
02_right    11,591,424 ->  5,918,769   (48.9% removed)
03_back      5,918,769 ->  5,914,523   ( 0.1% removed)
04_left      5,914,523 ->  5,911,653   ( 0.0% removed)
05_top       5,911,653 ->  5,404,137   ( 8.6% removed)
06_bottom    5,404,137 ->  5,402,974   ( 0.0% removed)
```

Back, left and bottom remove essentially nothing. This is not a bug and not a
property of this model: **under orthographic projection the silhouette along
+Y and along −Y are the same set**, so opposite views carve identical prisms.
The 0.0–0.1% they do remove is anti-aliasing noise at the boundary.

Carving with only front, right and top gives 5,409,095 voxels against 5,402,974
for all six — a **0.11% difference**.

The three opposite views are not wasted: they see surfaces the first three
cannot, and they carry the depth, normal and appearance information those
surfaces need. They are simply worth nothing for shape-from-silhouette.

This settles decision 1 in the plan. Auxiliary three-quarter views are not a
nice-to-have for hard subjects — they are the **only** way to add silhouette
constraints to this setup, because the canonical six provide three.

### 3. The fat is the hull's limit, not a resolution artifact

Doubling the grid resolution moves silhouette IoU 0.964 → 0.977, which is
discretization converging as expected. But Chamfer moves only 0.0206 → 0.0198
and the volume ratio only 2.007× → 1.966×.

**Twice the true volume is where the visual hull converges, not where it
starts.** More voxels will not fix it. Only information the silhouettes do not
contain — depth, normals, or additional viewing directions — can.

The comparison render shows what that means: the outline is right in every
view, and the interior is a slab. The drapery folds, the gap between the
raised arm and the body, and the space under the wings are all filled solid.
This is the predicted failure mode, and it is why the plan's later stages
exist.

## What this changes in the plan

- Decision 1 resolved: add auxiliary three-quarter views. The canonical six
  give three silhouette constraints, and each diagonal view adds a genuinely
  new one.
- The hull envelope constraint is validated as a hard bound, so S6 can use it
  with confidence rather than as a soft prior.
- The refinement stages now have a number to beat: Chamfer 0.0198 and volume
  ratio 1.97×, from a reconstruction that never fails.

---

## Extended in M1b

Eight three-quarter views were added later, and the floor moved with them.
See [M1b](M1b_RESULTS.md).

| | 6 views | 14 views |
|---|---|---|
| Chamfer | 0.019788 | **0.009948** |
| volume ratio | 1.966× | **1.443×** |
| containment | 100.0000% | 100.0000% |
| triangles | 3,132,272 | 2,854,890 |

The conclusion in finding 3 above — that the fat is the method's limit rather
than a resolution artifact — held. Doubling the grid moved Chamfer by about
0.001 at both view counts. What halved it was more viewing directions.
