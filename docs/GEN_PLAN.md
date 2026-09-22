# Generated input: control, then correction

Everything measured so far leaned on a luxury that disappears next. The views
came from one mesh, so the cameras were exact, the silhouettes were exact, and
there was an answer key to score against. Generated images give none of that.

Two plans, in order. The first stops most of the error from being created. The
second cleans up what still is.

---

# Plan A — Control: the code owns the geometry

## The inversion

Reference chaining asks the diffusion model to be geometrically consistent
across fourteen images. It cannot be, and no prompt makes it so: nothing in the
model represents "the same object from another angle" as a constraint. It
resamples a plausible creature each time and we hope the samples agree.

So stop asking.

> **A 3D proxy owns the geometry. Diffusion owns the surface.**

Build a rough creature as an actual 3D object in code. Render its depth,
normals and silhouette from the fourteen cameras. Those renders are consistent
*by construction* — one object, fourteen cameras, no model involved. Then hand
each render to diffusion as a structural constraint and let it paint a detailed
creature onto that skeleton.

The generated images inherit the proxy's consistency. Drift stops being
something to detect and correct, and becomes something that mostly cannot
happen.

This is the shape of [Coin3D](https://arxiv.org/pdf/2405.08054), which
conditions multiview diffusion on a coarse proxy assembled from basic volumes,
and of the mesh-guided conditioning in
[MeSS](https://arxiv.org/pdf/2508.15169).

## A1. The proxy

Not sculpted — **specified**. A creature is a skeleton plus volumes:

```python
spine   = bezier([...])            # head to tail
limbs   = [chain(joint, joint, joint) for _ in range(6)]
body    = metaballs along spine, radius r(t)
crown   = 7 cones, mirrored, fanning back
tail    = 12 rings, tapering
```

Blender does this natively: metaballs, skin modifier on an armature,
curve-to-mesh. A hundred lines and a few minutes of parameter fiddling.

It will be ugly. It does not matter. Its job is to be **exactly one object**
with the right proportions, limb count and topology. The diffusion supplies
everything else.

The three properties that matter:

- **Consistent by construction.** Fourteen renders of one object cannot
  disagree about how many legs it has.
- **Silhouettes are exact.** M1 measured silhouettes as the most valuable
  signal in this pipeline; now they are authored rather than estimated.
- **Cameras are known.** M2b's failures were all registration. That entire
  failure mode is gone again, for the same reason it was gone before: we
  placed the cameras.

## A2. Conditioning

Per view, the proxy gives a stack of control signals:

| signal | what it pins |
|---|---|
| depth | the surface's distance and overall form |
| normal | local orientation — the signal S2 consumes |
| silhouette / mask | the outline, exactly |
| canny on plate seams | where the hard edges go |

Feed these to ControlNet ([Zhang et al. 2023](https://arxiv.org/abs/2302.05543),
SDXL/FLUX variants available). Two or three stacked at moderate weight is
typically enough to lock structure without crushing detail.

Identity across views is a separate problem from geometry: ControlNet holds the
pose, IP-Adapter or a fixed reference image holds the look. The existing
reference chain still helps here — it is just no longer carrying the geometry
on its own.

## A3. The upgrade: models that are multi-view native

The strongest option skips generic ControlNet for models built for this.

[**Wonder3D**](https://openaccess.thecvf.com/content/CVPR2024/papers/Long_Wonder3D_Single_Image_to_3D_using_Cross-Domain_Diffusion_CVPR_2024_paper.pdf)
generates multi-view **normal maps and colour images together**, with
cross-domain attention keeping the two aligned, and — critically for us —
**defines its views in orthographic space**, which is the projection our whole
pipeline uses. [Wonder3D++](https://arxiv.org/abs/2511.01767) extends it and
supports both projections.
[**Era3D**](https://penghtyx.github.io/Era3D/) does high-resolution multiview
diffusion with row-wise attention in a canonical camera setting.

That is a direct hit on this project: it generates exactly the signal S2 eats,
already cross-view consistent, already orthographic.

The catch is these models generate *their* canonical view set, not our
fourteen. So the sequence is:

```
proxy (ours, code)
  → multi-view diffusion with the proxy as structural condition
  → consistent normals + colour in the model's views
  → our renderer re-projects to our fourteen cameras
```

The proxy is what bridges the two view sets, because it lives in 3D.

## A4. The loop the proxy makes possible

The proxy does not have to be right the first time:

```
proxy v0  → condition → generate → S1 carve + S2 integrate → mesh v1
proxy v1 := mesh v1   → condition → generate → mesh v2
```

Each round the structural condition is a better object, so the images get more
consistent, so the reconstruction gets better. This is the Coin3D idea with the
reconstruction closing the loop.

It also has a natural stopping rule: iterate until the mesh stops changing.

---

# Plan B — Correction: what control does not catch

Control reduces the error. It does not zero it. Diffusion will still add a
horn that the proxy did not have, or shift the head three pixels left.

Every mechanism below already exists in this repo, built for the ground-truth
case. This lists what changes when the answer key is taken away.

## B1. The proxy replaces ground truth as the reference

Not as an answer — it is wrong about detail by design — but as a **consistency
reference**. Every view can be scored against it without any external truth:

| check | what it catches |
|---|---|
| silhouette IoU vs proxy render | the creature changed shape in this view |
| normal agreement vs proxy normals | the surface turned the wrong way |
| shared-axis extents across views | the M0 alignment test, unchanged |
| limb and part counts | diffusion grew a leg |

A view failing badly is **regenerated, not repaired**. Generation is cheap;
propagating a bad view through the pipeline is not. This is the automated
version of the manual checklist already in `docs/PROMPT_PACK.md`.

## B2. Drift as free parameters, initialised by the proxy

The plan's original principle 2 — do not fight the inconsistency, solve for it
— survives intact and gets easier. Per view: a 2D similarity, a low-order warp,
and a depth affine, optimised jointly with the shape.

What changes is the starting point. Before, those parameters started at
identity and had to find their way. Now the proxy gives an initial estimate per
view: align each generated silhouette to the proxy's silhouette and read the
transform off directly. The optimiser starts close and only has to clean up.

## B3. The existing safety rails, unchanged

This is the part that does not need redesigning, and the reason the ground-truth
work was worth doing:

- **Conservative silhouette carving** still yields a hull that contains the
  object, because it depends on the silhouettes being *outer bounds*, not on
  them being *correct*. A generated silhouette that is slightly too fat is
  still a valid bound. One that is too thin is the danger, so silhouettes get
  dilated by the measured drift magnitude rather than one voxel.
- **Voxels only ever leave.** Spikes, holes, flyaway geometry and divergence
  remain structurally impossible regardless of how bad the images are.
- **Per-view carving budgets** already exist and are already driven by measured
  per-view quality. On generated input the quality measure changes from
  "agreement with truth" to "agreement with the proxy and with the other
  views", and the machinery is identical.
- **Carving by agreement** — silhouettes veto, depths vote — is exactly the
  robustness that [SparseFlex](https://arxiv.org/html/2503.21732v1) and the
  sparse-view literature build for. One hallucinating view cannot carve.

## B4. Normals over depth, now for a second reason

Normals were chosen over depth because they carry no scale or shift ambiguity.
The multi-view literature adds another: **normals, being first-order
derivatives, are inherently more tolerant of cross-view inconsistency than raw
depth**. A depth map that is globally offset is wrong everywhere; a normal map
from a slightly shifted view is still locally right.

Exactly the property wanted when the inputs are generated.

## B5. Calibrating the whole thing without an answer key

Once, on a known mesh:

1. Take Lucy's fourteen ground-truth renders.
2. Run them through the **generation** path — proxy from Lucy's own hull,
   conditioned diffusion, back out as images.
3. Reconstruct from the generated images and score against Lucy.

That measures end-to-end what generation costs, in the same units used
throughout: Chamfer, containment, volume ratio. Every threshold in Plan B —
rejection cut-off, dilation margin, carving budget — gets a number instead of a
guess, and those numbers transfer to creatures with no ground truth.

The harness built in M0 turns out to be the calibration rig for the generative
pipeline. That was not the plan, and it is the best thing about it.

---

## Order of work

1. Finish S2 on ground truth. Nothing below is meaningful until the
   reconstruction works with good input.
2. Proxy builder in Blender: skeleton, metaballs, parameters.
3. Control render pass: depth, normal, mask, canny from the fourteen cameras.
4. ControlNet generation, one view, tuned until structure holds and detail
   survives.
5. All fourteen. Score against the proxy with the B1 checks.
6. Full reconstruction from generated images, scored against Lucy (B5).
7. Turn the resulting numbers into the thresholds, then run the creature.

## Sources

- [Coin3D: Controllable and Interactive 3D Assets Generation with Proxy-Guided Conditioning](https://arxiv.org/pdf/2405.08054)
- [Wonder3D: Single Image to 3D using Cross-Domain Diffusion (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Long_Wonder3D_Single_Image_to_3D_using_Cross-Domain_Diffusion_CVPR_2024_paper.pdf)
- [Wonder3D++: Cross-domain Diffusion for High-fidelity 3D Generation](https://arxiv.org/abs/2511.01767)
- [Era3D: High-Resolution Multiview Diffusion using Efficient Row-wise Attention](https://penghtyx.github.io/Era3D/)
- [MeSS: City Mesh-Guided Outdoor Scene Generation with Cross-View Consistent Diffusion](https://arxiv.org/pdf/2508.15169)
- [SparseFlex: High-Resolution and Arbitrary-Topology 3D Shape Modeling](https://arxiv.org/html/2503.21732v1)
- [Sparse-View 3D Reconstruction: Recent Advances and Open Challenges](https://arxiv.org/pdf/2507.16406)
