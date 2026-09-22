#!/usr/bin/env python3
"""Render a mesh for looking at, rather than for measuring.

The orthographic view sets exist to be fed to algorithms: flat light, no
shadow, no perspective, all of which make a shape hard to read by eye. This
lights the same mesh the way a sculpture is photographed -- a large soft key,
a cool fill, a rim to lift the silhouette off the background, and a lens with
enough perspective to give depth -- so what the reconstruction actually looks
like is visible.

Run:
  assets/blender/blender -b -P tools/showcase.py -- \
      --mesh out/hull14_1024.ply --out out/showcase --res 1400
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

# name -> (azimuth, elevation, lens mm)
SHOTS = {
    "a_three_quarter": (300, 8, 60),
    "b_front":         (270, 5, 70),
    "c_profile":       (5, 6, 70),
    "d_low_hero":      (320, -12, 45),
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--res", type=int, default=1400)
    p.add_argument("--samples", type=int, default=160)
    p.add_argument("--albedo", type=float, default=0.55)
    p.add_argument("--focus", default=None,
                   help="x,y,z to aim at, in the mesh's own coordinates. "
                        "Implies --keep-coords so that two meshes framed on "
                        "the same point are framed identically")
    p.add_argument("--radius", type=float, default=None,
                   help="radius of the sphere to frame around --focus")
    p.add_argument("--keep-coords", action="store_true",
                   help="do not re-normalise the mesh to its own bounding box; "
                        "a reconstruction's box differs slightly from the "
                        "ground truth's, which would shift every close-up")
    p.add_argument("--shots", default=None,
                   help="name:az:el:lens,... replacing the default four")
    return p.parse_args(argv)


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    {".ply": bpy.ops.wm.ply_import,
     ".obj": bpy.ops.wm.obj_import,
     ".stl": bpy.ops.wm.stl_import}[ext](filepath=path)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if len(objs) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
    return objs[0]


def normalize(obj):
    bpy.context.view_layer.update()
    c = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    lo = Vector((min(p[i] for p in c) for i in range(3)))
    hi = Vector((max(p[i] for p in c) for i in range(3)))
    s = 1.0 / max(hi - lo)
    obj.scale = (s, s, s)
    obj.location = -((lo + hi) / 2) * s
    bpy.context.view_layer.update()
    return ((hi - lo) * s).length / 2  # bounding sphere radius


def light(name, kind, loc, energy, size, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, type=kind)
    d.energy = energy
    d.color = color
    if kind == "AREA":
        d.size = size
        d.shape = "DISK"
    o = bpy.data.objects.new(name, d)
    o.location = loc
    # aim at the origin
    z = Vector(loc).normalized()
    x = Vector((0, 0, 1)).cross(z)
    x = x.normalized() if x.length > 1e-6 else Vector((1, 0, 0))
    y = z.cross(x)
    o.matrix_world = Matrix(((x.x, y.x, z.x, loc[0]),
                             (x.y, y.y, z.y, loc[1]),
                             (x.z, y.z, z.z, loc[2]),
                             (0, 0, 0, 1)))
    bpy.context.scene.collection.objects.link(o)
    return o


def main():
    args = parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = import_mesh(os.path.abspath(args.mesh))
    print(f"[mesh] {len(obj.data.vertices):,} verts / "
          f"{len(obj.data.polygons):,} faces", flush=True)
    if args.focus:
        args.keep_coords = True
    if args.keep_coords:
        bpy.context.view_layer.update()
        c = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
        radius = (Vector((max(p[i] for p in c) for i in range(3)))
                  - Vector((min(p[i] for p in c) for i in range(3)))).length / 2
    else:
        radius = normalize(obj)
    target = Vector((0.0, 0.0, 0.0))
    if args.focus:
        target = Vector(tuple(float(t) for t in args.focus.split(",")))
        radius = args.radius or 0.08
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()

    mat = bpy.data.materials.new("plaster")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    a = args.albedo
    b.inputs["Base Color"].default_value = (a, a * 0.99, a * 0.96, 1.0)
    b.inputs["Roughness"].default_value = 0.55
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = 0.3
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    world = bpy.data.worlds.new("studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = \
        (0.045, 0.048, 0.055, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world

    # key from upper left, cool fill from the right, rim from behind
    light("key", "AREA", (-1.7, -1.9, 1.9), 220, 2.6)
    light("fill", "AREA", (2.2, -1.2, 0.2), 45, 3.0, (0.75, 0.82, 1.0))
    light("rim", "AREA", (0.9, 2.3, 1.4), 130, 1.6, (1.0, 0.95, 0.88))

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = args.samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.render.resolution_x = args.res
    sc.render.resolution_y = int(args.res * 1.25)
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Filmic" if "Filmic" in [
        v.name for v in sc.view_settings.bl_rna.properties["view_transform"].enum_items
    ] else "Standard"
    sc.render.image_settings.file_format = "PNG"

    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam

    shots = SHOTS
    if args.shots:
        shots = {}
        for spec in args.shots.split(","):
            name, az, el, lens = spec.split(":")
            shots[name] = (float(az), float(el), float(lens))
    # the lights were placed for a subject at the origin; carry them along
    for o in sc.objects:
        if o.type == "LIGHT":
            o.matrix_world.translation += target

    for name, (az, el, lens) in shots.items():
        cam_data.lens = lens
        # Frame the bounding sphere rather than guessing a distance: back off
        # exactly far enough for it to fit the vertical field of view, plus a
        # margin. The previous constant left the subject adrift in the frame.
        cam_data.sensor_fit = "VERTICAL"
        cam_data.sensor_height = 24.0
        half_fov = math.atan(cam_data.sensor_height / (2.0 * lens))
        dist = radius / math.sin(half_fov) * 1.06
        a_, e_ = math.radians(az), math.radians(el)
        d = Vector((math.cos(e_) * math.cos(a_),
                    math.cos(e_) * math.sin(a_),
                    math.sin(e_)))
        loc = target + d * dist
        z = d.normalized()
        x = Vector((0, 0, 1)).cross(z).normalized()
        y = z.cross(x)
        cam.matrix_world = Matrix(((x.x, y.x, z.x, loc.x),
                                   (x.y, y.y, z.y, loc.y),
                                   (x.z, y.z, z.z, loc.z),
                                   (0, 0, 0, 1)))
        sc.render.filepath = os.path.join(out, name + ".png")
        print(f"[shot] {name}  az={az} el={el} {lens}mm", flush=True)
        bpy.ops.render.render(write_still=True)

    print("[done]", out, flush=True)


if __name__ == "__main__":
    main()
