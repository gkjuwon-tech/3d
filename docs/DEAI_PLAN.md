# Removing the models from the geometry stage

Image generation stays generative — that is what diffusion is for. This is
about everything after it: how much of the 3D path can run on algorithms that
are deterministic, auditable, and cannot hallucinate a horn.

## Audit: where the models actually are

| stage | today | status |
|---|---|---|
| silhouette / mask | exact from render; SAM or BiRefNet on generated input | **replaceable, trivially** |
| visual hull carving | pure algorithm | already clean — and the best performer in the project |
| camera registration | none. We place the cameras | already clean, twice over |
| depth estimation | dropped | measured worse than the hull; out |
| **normal estimation** | **Marigold / MoGe-2** | **the only real dependency left** |
| normal integration | screened Poisson | already clean |
| discontinuity detection | grazing + normal turn | already clean |
| meshing, smoothing, remesh | marching cubes | already clean |

The geometry stage is already mostly algorithmic, and not by design — it ended
up that way because every model that was tried lost to an algorithm. Depth
estimators lost to the hull's own depth. Multi-view networks could not place
fourteen orthographic views and were replaced by knowing where the cameras are.

**One model is left: the normal estimator.** Everything below is about that
one, plus a path that does not need normals at all.

---

## Route A — Photometric stereo

The classical answer, and the strongest one.

[Photometric stereo](https://www.sciencedirect.com/science/article/abs/pii/S1077314212000483)
recovers surface normals from three or more images of the same view under
*different known lighting directions*. With Lambertian shading, each pixel
gives a linear system

```
    I_k = albedo · (L_k · n)      k = 1..K
```

and `n` falls out of a 3×3 solve. No training, no priors, no drift, no
per-view offset ambiguity. It is a 1980 algorithm and it is exact.

Its historical problem is that it needs controlled lighting, which nobody has
outside a lab. **We are not outside a lab.** Under the proxy-guided generation
in `GEN_PLAN.md`, ControlNet pins the geometry while the prompt controls the
light, so asking for the same view lit from three directions is asking for
exactly the thing that is easy to vary and hard to get wrong. Geometry is held
by the control signal; only shading moves.

### Why this is worth doing first

Shape from shading on a single image is ill-posed — one equation, two unknowns
per pixel, which is why it needs a network to guess. Three lights make it
**over-determined**. The estimator stops being a guess and becomes a solve.

### The test costs nothing

Lucy can be rendered under any lighting, exactly. So before any generated
image is involved:

1. Render three light directions of one view.
2. Run photometric stereo.
3. Compare the recovered normals to the ground-truth normal pass.

If that returns normals good to a degree or two, the remaining question is
purely how faithfully diffusion preserves photometric consistency — which is
also measurable, by doing the same thing to generated images and scoring
against the proxy.

This is the first experiment when S2 is done, and it needs no GPU.

---

## Route B — Photo-consistency carving, with no normals at all

[Space carving](https://www.cs.toronto.edu/~kyros/pubs/00.ijcv.carve.pdf)
(Kutulakos & Seitz) removes a voxel when the views that see it *disagree about
its colour*. A voxel inside the true surface is seen consistently; a voxel
floating in a concavity the silhouettes could not reach projects onto different
surfaces in different views and gives different colours.

That is precisely the failure the visual hull has. M1 measured it: the hull
converges to 1.44× the true volume and more silhouettes will not fix it,
because the missing information is not in any silhouette. It is in the colours.

Properties that suit this project:

- **No normals, no depth, no network.** Colour comparison and a voxel grid.
- **Voxels only ever leave**, so it inherits the safety argument already built.
- It fails where the surface is featureless, and succeeds exactly where the
  surface has shading variation — which is where the hull is worst, since
  concavities are where ambient occlusion writes the most contrast.

The caveat is honest: our renders are matte and evenly lit by design, so the
photometric signal is weaker than a textured photograph. Whether it is strong
enough is measurable on Lucy in an afternoon, and the answer does not depend on
any model.

---

## Route C — Constraints that cost nothing

Neither of these needs a pixel of new input.

**Symmetry.** A bilaterally symmetric subject halves the unknowns and doubles
the effective observations. The plan dropped it because Lucy is asymmetric; the
actual target creature was designed symmetric on purpose.

**More silhouettes.** M1b measured this directly: eight extra views halved the
hull's Chamfer, from 0.0206 to 0.0106. That is a larger improvement than any
depth model in M2 or M2b produced, for the cost of rendering. The redundancy
law says only one hemisphere is needed, so the extra views are half price.

---

## Route D — Keep the model, demote it

If photometric stereo works, the learned normals do not have to be deleted.
They become a **second opinion**: where the two agree, confidence is high and
the carve may be aggressive; where they disagree, the algorithmic answer wins
and the carving budget shrinks.

That is strictly better than either alone, and it turns the model from a
dependency into an input that has to earn its place per pixel — which is how
every other estimator in this project has been treated.

---

## The target

```
image generation   diffusion, unchanged -- that is its job
      |
      v
segmentation       threshold on a flat plate. No model.
silhouette carve   algorithm. Already the best thing here.
normals            photometric stereo from 3 lights. No model.
photo-consistency  colour agreement across views. No model.
integration        screened Poisson. No model.
meshing            marching cubes. No model.
```

**Models in the geometry path: zero.** Diffusion supplies pixels; everything
that touches geometry is a solve with a known answer and a scoreboard.

That is not purism. It is what the measurements have been saying all along:
every model tried in this stage lost to an algorithm, and the one still
standing has never been compared against the classical method that solves its
exact problem.

## Order of work, after S2

1. Photometric stereo on Lucy, three lights, one view. Score against the true
   normals. Zero AI, zero GPU.
2. If it holds, run it through the full carve and compare against the learned
   normals on the same views.
3. Photo-consistency carving on Lucy, and measure how much of the hull's 1.44×
   volume it can remove that silhouettes cannot.
4. Symmetry, on a symmetric test subject.
5. Then generated input, where the only remaining question is how well
   diffusion preserves photometric consistency across a lighting change — and
   that has a number too.

## Sources

- [Kutulakos & Seitz, A Theory of Shape by Space Carving](https://www.cs.toronto.edu/~kyros/pubs/00.ijcv.carve.pdf)
- [Surface reflectance and normal estimation from photometric stereo](https://www.sciencedirect.com/science/article/abs/pii/S1077314212000483)
- [Shape from Shading and Photometric Stereo Methods](https://www.researchgate.net/publication/2423527_Shape_from_Shading_and_Photometric_Stereo_Methods)
- [Recent Advances in Image-Based 3D Reconstruction: conventional and learning-based](https://link.springer.com/article/10.1007/s41064-026-00412-y)
- [Self-supervised view synthesis with efficient multi-scale voxel carving](https://arxiv.org/html/2306.14709)
