# Known issues

## Holes in small, thin, protruding features (open)

First integrated Lucy mesh (stage 2, `data/lucy_k`, slow CPU pipeline on
Kaggle). The overall shape is right -- drapery, belt, wings, plinth -- and
the scores say so:

|              | Chamfer  | F@1   | F@2   | normal median | volume |
|--------------|----------|-------|-------|---------------|--------|
| hull         | 0.009177 | 18.8% | 30.2% | 26.3 deg      | 1.402x |
| fused mesh   | 0.001846 | 74.0% | 83.4% |  7.2 deg      | 0.981x |

(skirt F@1 93.5%, plinth 88.0%, chest 80.6%, face 63.6%)

But it is holed in exactly one kind of place, and they are one problem, not
several:

- the face: a crater from the cheek through the nose and mouth
- the fingers of the outstretched hand
- the hand holding the torch, which is eaten away so the flame floats free
- the toes

All of them are small, thin and stick out. Working hypothesis, not yet
tested:

- A pixel placed too deep in one view claims the space in front of it is
  empty. On a large surface other views outvote it; on a finger, few views
  see the same spot squarely, so one confident wrong view decides.
- The per-view "behind the truth by > 3 voxels" rate is 0.7% to 7.2%, and
  those errors cluster at depth discontinuities -- which is where a finger,
  a nose or a toe is.
- The TSDF weighted mean has no notion of majority: one view's +trunc
  outweighs two views' small negatives if its confidence weight is higher.

Candidate fixes, to be measured before any is adopted:

1. free-space consistency filter on the depth maps (tools/depth_filter.py
   exists; it caught 27-94% of too-deep pixels but also flagged 5-21% of good
   ones on the first five views)
2. robust fusion: weighted median, or require agreement from a second view
   before a voxel may be emptied (the quorum idea, applied inside the TSDF)
3. a thin-structure guard like S2's, on the continuous field

Deferred: speed first (see docs/SPEED.md when written), with quality held.

## Floating debris in the fused mesh (open)

The confirmed v3 Lucy mesh has 2,771 disconnected pieces; the small ones add
up to 76,146 of its 3,884,286 faces. Retopology drops them, but the raw mesh
carries them. Likely the same mechanism as the holes -- a view's wrong depth
leaving a sliver on the wrong side of the surface -- so it should be looked at
together with them.
