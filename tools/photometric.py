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
    ap.add_argument("--yaw", type=float, default=180.0)
    ap.add_argument("--shadows", action="store_true",
                    help="leave cast shadows on; off by default so the first "
                         "measurement isolates the Lambertian assumption")
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
    sc.cycles.use_denoising = True
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

    for view, mw in plan:
        cam.matrix_world = mw
        bpy.context.view_layer.update()
        R = cam.matrix_world.to_3x3()
        render_lights(sc, view, R, out, args)
    print("[done]", out, flush=True)


def render_lights(sc, view, R, out, args):
    import bpy
    from mathutils import Matrix, Vector
    for name, d in LIGHTS.items():
        for o in list(sc.objects):
            if o.type == "LIGHT":
                bpy.data.objects.remove(o, do_unlink=True)
        import math
        world_dir = R @ Vector(d).normalized()
        ld = bpy.data.lights.new(f"L{name}", type="SUN")
        ld.energy = math.pi          # so a facing surface returns albedo
        ld.angle = 0.0
        if not args.shadows:
            ld.cycles.cast_shadow = False
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
    print(f"[view] {view} lit {len(LIGHTS)} ways", flush=True)


def solve(argv):
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--view", default="01_front")
    ap.add_argument("--lights", default="a,b,c,d")
    args = ap.parse_args(argv)

    import OpenEXR  # noqa: F401  (only to make the failure explicit)


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
