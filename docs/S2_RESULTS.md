# S2 — carving the hull down to a surface

The hull was a floor, not a result. It contained the statue with certainty and
nothing else: 100.0000% containment, 1.443x the true volume, Chamfer 0.009948.
S2 was the step that had to turn that block into a surface.

## What finally worked

Three things, in this order. None of them is a learned 3D model.

**1. Photometric stereo replaces the learned normal estimator.** Four lit
renders per view, a Lambertian solve, and the darkest observations per pixel
dropped so attached shadows do not poison the fit. Median angular error
5.88 deg against Marigold's 27.80 deg -- a factor of five. With those normals
every one of the 14 views integrates to a depth map that beats the hull's own
depth, by +7.4% to +31.3%, with no view coming out worse. With Marigold the
same pipeline ranged from +36% to -275% and only 3 views of 14 were safe to
use at all.

**2. A quorum, not an intersection.** Carving is an intersection, so one bad
view decides a voxel for everyone -- which is what shredded one side of the
first result while the other side came out clean. Each view instead votes,
and a voxel is removed only once two views agree it is empty.

The vote histogram over the 31,079,206 hull voxels:

    0 votes  18,544,777
    1 vote    9,628,336
    2 votes   2,733,841
    3 votes     168,738
    4 votes       3,514

Four is the maximum across fourteen views, and that is not a defect. The
hull's excess volume sits almost entirely in concavities -- between the legs,
under the arms, in the gap behind a wing -- and a concavity is by definition
visible from only a few directions. "At least two views agree" is a
structural threshold, not a tuned one.

**3. A thickness gate, because Chamfer distance cannot see a wing.** The
quorum carve beat the hull on Chamfer, 0.009084 against 0.009948, and looked
worse. Rendering it showed why: it thins Lucy's wings correctly -- one comes
out as the blade it should be, where the hull had a slab -- and tears the
other open, and shreds the hand into filaments. A membrane four voxels thick
only has to be over-carved by two from each side to stop being a surface, and
a membrane holds so few surface samples that the metric barely moves.

The damage is not incidental. Measured on the hull, thin voxels are marked for
removal at 94% (16,530 of 17,607 at T=3) against 12% for the hull as a whole.
The carve is specifically eating plates and filaments, because a plate seen
near edge-on integrates to a depth that lands behind it, and the plate then
looks like empty space in front of a surface.

So the carve is gated on local thickness: erode the hull by T, dilate the
survivors back, and whatever fails to return has no core of its own and is
left exactly as the hull had it. A closing afterwards fills the holes that
get punched through anyway -- gaps narrower than 2r close, while the armpits
and the gap between the legs, which are far wider, stay carved -- intersected
with the hull so the repair can never spend containment putting voxels back.

## Numbers

Against the hull at Chamfer 0.009948 / containment 100.0000% / volume 1.443x:

| variant                   | Chamfer  | containment | volume |
|---------------------------|----------|-------------|--------|
| hull (the floor)          | 0.009948 | 100.00%     | 1.443x |
| quorum k=1                | 0.013774 |  60.39%     | 0.858x |
| quorum k=2                | 0.009084 |  88.45%     | 1.306x |
| quorum k=3                | 0.009860 |  99.78%     | 1.435x |
| guard T=8                 | 0.007065 |  89.80%     | 1.280x |
| guard T=8  + close 1      | 0.006999 |  91.19%     | 1.285x |
| **guard T=12 + close 1**  | **0.007336** | **95.85%** | **1.303x** |
| guard T=16                | 0.007538 |  96.56%     | 1.311x |
| guard T=16 + close 1      | 0.007563 |  97.67%     | 1.317x |
| guard T=16 + close 2      | 0.007947 |  98.38%     | 1.334x |

T=12 with a radius-1 closing is the shipped result: 26.3% better than the hull
on Chamfer while giving up 4.15 points of containment. T=8 is the best
Chamfer on offer and T=16 the safest surface; the whole curve is one
parameter and costs about two minutes to move along, because the votes are
saved once and thresholded afterwards.

For contrast, the best result the learned normal estimator ever produced was
Chamfer 0.010946 at 74.6% containment -- worse than the hull it started from.

## What is still wrong

**Terracing.** The concentric rectangular steps across the drapery are
inherited from the hull, not created by the carve; they are visible in the
hull render too. They are the signature of an axis-aligned boolean decision
per voxel, and no amount of quorum tuning removes them. The fix is to stop
carving booleans and fuse a truncated signed distance field instead, so the
isosurface can land between voxels rather than on a face.

**Crust.** Filament debris along the torso and around the base survives the
radius-1 opening. Only 8,543 voxels of it are actually disconnected; the rest
is attached, so a connected-component pass will not reach it.

**The face.** Still mush. It is small, concave, and seen face-on by exactly
one view.

## Reproducing

    python3 tools/photometric.py --render ...        # 4 lit renders per view
    python3 tools/photometric.py --solve  ...        # camera-space normals
    python3 tools/carve_depth.py --normals-dir out/ps_normals \
        --nz-floor 0.35 --budget 15 --margin-voxels 2 --save-votes votes.npz
    python3 tools/thin_guard.py --votes votes.npz --quorum 2 --thickness 12 --out g12
    python3 tools/repair.py --carved g12_occ.npz --hull out/final_mesh_occ.npz \
        --close 1 --open 1 --out out/s2_mesh

No GPU, no learned 3D model.
