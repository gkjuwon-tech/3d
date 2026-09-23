#!/usr/bin/env python3
"""Recover surface normals from several images of one view under known lights.

Run inside Blender to render the lit images:
  assets/blender/blender -b -P tools/photometric.py -- render \
      --mesh assets/lucy_le.ply --out out/photo --view 01_front --res 1024

Then, in plain Python, to solve and score:
  python3 tools/photometric.py solve --dir out/photo --views refs/lucy_gt \
      --view 01_front

Under Lambertian shading a pixel lit by direction L with albedo a satisfies
I = a (L . n). Three lights make that an over-determined linear system in the
scaled normal a*n, so the normal is a 3x3 solve per pixel -- not an estimate, a
solution. It has no training, no prior, no drift and no per-view offset
ambiguity, which are the four things that have cost this project the most.

The classical blocker is that it needs controlled lighting. Under the
proxy-guided generation plan the geometry is pinned by a control signal while
the prompt moves the light, so controlled lighting is the easy part to vary.
Whether diffusion honours it is a separate measurement; this one establishes
what the method is worth when the lighting really is known.
"""
import argparse
import math
import os
import sys

import numpy as np

# light directions in camera space: right, left-ish, above. Kept well separated
# so the 3x3 system stays far from singular.
LIGHTS = {
    "a": (0.4, 0.2, 0.894),
    "b": (-0.5, 0.1, 0.860),
    "c": (0.05, 0.55, 0.834),
    "d": (0.0, -0.5, 0.866),
}


def light_set(n):
    """Camera-space light directions. 4: the original rig (kept so old
    renders still solve). Otherwise n lights evenly around the view axis,
    alternating between 25 and 45 degrees off it: with shadows, a pixel needs
    three lights that reach it, and next to an arm or under a wing four lights
    left 17% of Lucy's view 14 with two or fewer."""
    if n == 4:
        return dict(LIGHTS)
    out = {}
    for i in range(n):
        az = 2 * math.pi * i / n
        el = math.radians(25.0 if i % 2 == 0 else 45.0)
        out[chr(ord("a") + i)] = (math.sin(el) * math.cos(az),
                                  math.sin(el) * math.sin(az), math.cos(el))
    return out


def render(argv):
    import bpy
    from mathutils import Euler, Matrix, Vector

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views-json", default=None,
                   help="cameras.json; renders every view it lists")
    ap.add_argument("--view", default="01_front")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--no-denoise", action="store_true",
                    help="a shadowless sun on a pure diffuse surface shades every "
                         "sample identically, so the only noise is edge "
                         "coverage; the denoiser can only move good values")
    ap.add_argument("--yaw", type=float, default=180.0)
    ap.add_argument("--lights", type=int, default=4,
                    help="number of lights per view (see light_set)")
    ap.add_argument("--no-shadows", action="store_true",
                    help="lights cast no shadows (not physical; for isolating "
                         "the Lambertian model). The default keeps them, as a "
                         "real capture has them -- and so did every render "
                         "made before this flag: the old switch set "
                         "light.cycles.cast_shadow, which Blender 4 ignores")
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.ply_import(filepath=os.path.abspath(args.mesh))
    obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
    bpy.context.view_layer.objects.active = obj
    obj.rotation_euler[2] += math.radians(args.yaw)
    bpy.context.view_layer.update()
    c = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    lo = Vector((min(p[i] for p in c) for i in range(3)))
    hi = Vector((max(p[i] for p in c) for i in range(3)))
    s = 1.0 / max(hi - lo)
    obj.location = -((lo + hi) / 2) * s
    obj.scale = (s, s, s)
    bpy.context.view_layer.update()
    bpy.ops.object.shade_smooth()

    mat = bpy.data.materials.new("lambert")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    diff.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    outp = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(diff.outputs[0], outp.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    world = bpy.data.worlds.new("black")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = world

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = not args.no_denoise
    sc.cycles.max_bounces = 0          # direct light only: pure Lambertian
    sc.render.resolution_x = sc.render.resolution_y = args.res
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "32"

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.clip_start, cam_data.clip_end = 0.01, 100.0
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam

    if args.views_json:
        import json
        meta = json.load(open(os.path.abspath(args.views_json)))
        cam_data.ortho_scale = meta["ortho_scale"]
        plan = [(v, Matrix(i["matrix_world"])) for v, i in meta["views"].items()]
    else:
        cam_data.ortho_scale = 1.10
        m = Matrix.Identity(4)
        m = Matrix.Translation(Vector((0, -2, 0))) @ \
            Euler([math.radians(90), 0, 0], "XYZ").to_matrix().to_4x4()
        plan = [(args.view, m)]

    import json as _json
    lights = light_set(args.lights)
    with open(os.path.join(out, "lights.json"), "w") as f:
        _json.dump(lights, f)
    for view, mw in plan:
        cam.matrix_world = mw
        bpy.context.view_layer.update()
        R = cam.matrix_world.to_3x3()
        render_lights(sc, view, R, out, args, lights)
    print("[done]", out, flush=True)


def render_lights(sc, view, R, out, args, lights):
    import bpy
    from mathutils import Matrix, Vector
    for name, d in lights.items():
        for o in list(sc.objects):
            if o.type == "LIGHT":
                bpy.data.objects.remove(o, do_unlink=True)
        import math
        world_dir = R @ Vector(d).normalized()
        ld = bpy.data.lights.new(f"L{name}", type="SUN")
        ld.energy = math.pi          # so a facing surface returns albedo
        ld.angle = 0.0
        if args.no_shadows:
            ld.use_shadow = False
        lo_ = bpy.data.objects.new(f"L{name}", ld)
        sc.collection.objects.link(lo_)
        z = world_dir
        x = Vector((0, 0, 1)).cross(z)
        x = x.normalized() if x.length > 1e-6 else Vector((1, 0, 0))
        y = z.cross(x)
        lo_.matrix_world = Matrix(((x.x, y.x, z.x, 0.0),
                                   (x.y, y.y, z.y, 0.0),
                                   (x.z, y.z, z.z, 0.0),
                                   (0, 0, 0, 1)))
        sc.render.filepath = os.path.join(out, f"{view}_{name}.exr")
        bpy.ops.render.render(write_still=True)
    print(f"[view] {view} lit {len(lights)} ways", flush=True)


def solve(argv):
    """Solve normals from the lit renders. Runs under Blender, which reads EXR.

    Per pixel, only the lights that reach it are used (see below); a pixel
    with fewer than three has no normal (zero vector), which every consumer
    downstream already reads as "no information".
    """
    import itertools
    import json

    import bpy

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of lit renders")
    ap.add_argument("--out", required=True, help="directory for normal .npy")
    ap.add_argument("--views-json", required=True)
    ap.add_argument("--shadow-frac", type=float, default=0.02,
                    help="a light reading below this fraction of the pixel's "
                         "brightest counts as shadowed")
    ap.add_argument("--shadow-abs", type=float, default=1e-3)
    ap.add_argument("--score-against", default=None,
                    help="a view set with normal_npy, to report accuracy")
    args = ap.parse_args(argv)

    src = os.path.abspath(args.dir)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.abspath(args.views_json)))

    def load(path):
        img = bpy.data.images.load(path)
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        bpy.data.images.remove(img)
        return buf.reshape(h, w, 4)[::-1]

    lp = os.path.join(src, "lights.json")
    lights = json.load(open(lp)) if os.path.exists(lp) else dict(LIGHTS)
    names = list(lights)
    L = np.array([lights[k] for k in names], dtype=np.float64)
    L /= np.linalg.norm(L, axis=1, keepdims=True)

    for view in meta["views"]:
        paths = [os.path.join(src, f"{view}_{k}.exr") for k in names]
        present = [i for i, p in enumerate(paths) if os.path.exists(p)]
        if len(present) < 3:
            print(f"  {view}: only {len(present)} lights, skipped")
            continue
        obs, alpha = [], None
        for i in present:
            a = load(paths[i])
            obs.append(a[..., 0])
            alpha = a[..., 3] if alpha is None else alpha
        obs = np.stack(obs, -1)
        lit = alpha > 0.99
        H, W = lit.shape
        B = obs[lit]
        Ls = L[present]

        # A light in shadow -- cast (the render keeps shadows, as a real
        # capture would) or attached (facing away) -- reads zero, which is
        # not a measurement. Each pixel is solved from the lights that do
        # reach it, all of them, and a pixel fewer than three reach has no
        # normal at all: two equations do not fix three unknowns, and the
        # "three brightest" rule this replaces then solved with a zero in the
        # system and returned garbage without a sign of it -- on Lucy's view
        # 14, 17% of the object, nearly all of it beside the raised arm.
        Ls_ = Ls
        litm = B > np.maximum(args.shadow_frac * B.max(1, keepdims=True),
                              args.shadow_abs)
        g = np.zeros((len(B), 3))
        pats, inv = np.unique(litm, axis=0, return_inverse=True)
        inv = inv.ravel()
        for j, pat in enumerate(pats):
            if pat.sum() < 3:
                continue
            sel = inv == j
            M = Ls_[pat]
            g[sel] = np.linalg.lstsq(M, B[sel][:, pat].T, rcond=None)[0].T
        n_unmeasured = int((litm.sum(1) < 3).sum())
        normal = np.zeros((H, W, 3), dtype=np.float32)
        a = np.linalg.norm(g, axis=1)
        ok = a > 1e-4
        n = np.zeros_like(g)
        n[ok] = g[ok] / a[ok, None]
        full = np.zeros((H, W, 3))
        full[lit] = n
        normal[:] = full

        np.save(os.path.join(out, f"{view}.npy"), normal)
        note = ""
        if args.score_against:
            ref_p = os.path.join(args.score_against, "normal_npy", f"{view}.npy")
            if os.path.exists(ref_p):
                R = np.array(meta["views"][view]["matrix_world"])[:3, :3]
                ref = np.load(ref_p)
                ref = (ref.reshape(-1, 3) @ R).reshape(ref.shape)
                ref /= np.linalg.norm(ref, axis=2, keepdims=True).clip(1e-9)
                m = lit & (np.linalg.norm(normal, axis=2) > 0.5)
                ang = np.degrees(np.arccos(
                    np.clip((normal[m] * ref[m]).sum(1), -1, 1)))
                note = (f"  mean {ang.mean():5.2f} deg  median "
                        f"{np.median(ang):5.2f}  <5 {100*(ang<5).mean():4.1f}%")
        print(f"  {view:<13} {len(present)} lights -> {lit.sum():,} px, "
              f"{100*n_unmeasured/max(len(B), 1):.1f}% with < 3 lit (no normal){note}",
              flush=True)
    print(f"[done] {out}", flush=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if not argv:
        sys.exit(__doc__)
    cmd, rest = argv[0], argv[1:]
    if cmd == "render":
        render(rest)
    else:
        solve(rest)


if __name__ == "__main__":
    main()
