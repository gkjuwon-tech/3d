#!/usr/bin/env python3
"""Render a mesh as six axis-aligned orthographic views, with exact ground
truth for every signal the reconstruction pipeline consumes.

Each view is rendered twice, because a beauty pass and a geometry pass want
opposite things from the pixel filter:

  beauty pass   wide filter, many samples, denoised -> a clean image
  geometry pass one sample, point filter, no denoise -> exact depth/normals

Filtering a depth buffer averages foreground depth with the background
sentinel wherever a pixel straddles the silhouette, and averages unit normals
into non-unit ones wherever a pixel spans curvature. On this mesh a pixel
covers several triangles, so that is most of the frame. Those blended edge
samples are precisely the flying pixels that put skirts on a back-projected
depth map, so ground truth must be point-sampled or it carries the artifact it
exists to measure.

Per view it writes:
  rgb/<view>.png      flat-lit beauty render on a #808080 plate
  mask/<view>.png     anti-aliased silhouette, for sub-pixel extent fitting
  depth/<view>.exr    orthographic depth, point-sampled, float32
  normal/<view>.exr   world-space surface normal, point-sampled unit vectors
  cameras.json        per-view camera basis + ortho scale + normalization

All six views share one ortho scale and one centered object, so the frames are
aligned by construction — which is what makes them a usable unit test for the
fusion stage.

Run:
  assets/blender/blender -b -P tools/render_orthoviews.py -- \
      --mesh assets/lucy_le.ply --out refs/lucy_gt --res 2048 --samples 128
"""
import argparse
import json
import math
import os
import shutil
import sys

import bpy
from mathutils import Euler, Vector

# camera name -> (euler XYZ in degrees, unit vector from object to camera)
VIEWS = {
    "01_front":  ((90, 0, 0),     (0, -1, 0)),
    "02_right":  ((90, 0, 90),    (1, 0, 0)),
    "03_back":   ((90, 0, 180),   (0, 1, 0)),
    "04_left":   ((90, 0, -90),   (-1, 0, 0)),
    "05_top":    ((0, 0, 0),      (0, 0, 1)),
    "06_bottom": ((180, 0, 0),    (0, 0, -1)),
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--res", type=int, default=2048)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--margin", type=float, default=1.10,
                   help="ortho scale multiplier over the longest bbox axis")
    p.add_argument("--only", default=None)
    p.add_argument("--yaw", type=float, default=0.0,
                   help="rotate the object about Z (degrees) before rendering, "
                        "to put its actual front toward the front camera")
    p.add_argument("--albedo", type=float, default=0.45,
                   help="clay base colour; keep it above the 0.216 linear "
                        "plate so the subject separates from the background")
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    else:
        sys.exit(f"unsupported mesh format: {ext}")
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        sys.exit("no mesh object after import")
    if len(objs) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
    return bpy.context.scene.objects[objs[0].name]


def normalize(obj):
    """Center on the bbox center and scale the longest axis to 1.0."""
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c[i] for c in corners) for i in range(3)))
    hi = Vector((max(c[i] for c in corners) for i in range(3)))
    center = (lo + hi) / 2
    size = hi - lo
    longest = max(size)
    scale = 1.0 / longest

    obj.location = -center * scale
    obj.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    return {
        "source_bbox_min": list(lo),
        "source_bbox_max": list(hi),
        "source_size": list(size),
        "applied_scale": scale,
        "applied_offset": list(-center * scale),
        "normalized_size": [s * scale for s in size],
    }


def shade_smooth(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()


def make_material(obj, albedo):
    mat = bpy.data.materials.new("clay")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (albedo, albedo, albedo * 1.01, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def setup_world():
    """Uniform environment light: even from every direction, no cast shadows,
    and Cycles' global illumination supplies the contact darkening that makes
    the form readable."""
    world = bpy.data.worlds.new("flat")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world


def setup_render(scene, res, samples):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.use_denoising = True
    # Few bounces on purpose: light bouncing back out of crevices is what
    # flattens a clay render, and the crevices are the detail we want to read.
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 2
    scene.cycles.transmission_bounces = 0
    scene.cycles.transparent_max_bounces = 1

    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"

    # Standard transform so the grey plate lands on exactly #808080.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_normal = True
    vl.use_pass_combined = True


def setup_compositor(scene, out_dir):
    """Render layers -> three file outputs: grey-plated RGB, alpha mask,
    and the raw float passes."""
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")

    # 0.5 sRGB == 0.2140 linear; the compositor works in linear.
    plate = tree.nodes.new("CompositorNodeRGB")
    plate.outputs[0].default_value = (0.2140, 0.2140, 0.2140, 1.0)

    over = tree.nodes.new("CompositorNodeAlphaOver")
    over.premul = 1.0
    tree.links.new(plate.outputs[0], over.inputs[1])
    tree.links.new(rl.outputs["Image"], over.inputs[2])

    def file_out(name, fmt, color_mode, depth):
        n = tree.nodes.new("CompositorNodeOutputFile")
        n.name = name
        n.base_path = os.path.join(out_dir, name)
        n.format.file_format = fmt
        n.format.color_mode = color_mode
        n.format.color_depth = depth
        return n

    scratch = os.path.join(out_dir, ".unused")

    rgb = file_out("rgb", "PNG", "RGB", "8")
    tree.links.new(over.outputs[0], rgb.inputs[0])

    mask = file_out("mask", "PNG", "BW", "8")
    tree.links.new(rl.outputs["Alpha"], mask.inputs[0])

    depth = file_out("depth", "OPEN_EXR", "BW", "32")
    depth.format.exr_codec = "ZIP"
    tree.links.new(rl.outputs["Depth"], depth.inputs[0])

    normal = file_out("normal", "OPEN_EXR", "RGB", "32")
    normal.format.exr_codec = "ZIP"
    tree.links.new(rl.outputs["Normal"], normal.inputs[0])

    return {"rgb": rgb, "mask": mask, "depth": depth, "normal": normal,
            "_scratch": scratch, "_out": out_dir}


def set_pass(scene, nodes, which, samples):
    """Point the outputs we do not want at a scratch directory, and set the
    sampling and pixel filter this pass needs."""
    beauty = which == "beauty"
    if beauty:
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.render.filter_size = 1.5
        keep, drop = ("rgb", "mask"), ("depth", "normal")
    else:
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        # A filter this narrow is a point sample at the pixel centre, which is
        # what makes the depth and normal exact rather than blended.
        scene.render.filter_size = 0.01
        keep, drop = ("depth", "normal"), ("rgb", "mask")
    for k in keep:
        nodes[k].base_path = os.path.join(nodes["_out"], k)
    for k in drop:
        nodes[k].base_path = os.path.join(nodes["_scratch"], k)
    return keep


def setup_camera(scene, ortho_scale):
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    # Clip generously: the camera sits outside the object on every axis.
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def rename_frame_output(node_dir, view, ext):
    """File Output nodes append the frame number; fold it back into the name."""
    written = [f for f in os.listdir(node_dir) if f.endswith(ext)]
    if not written:
        return None
    src = os.path.join(node_dir, sorted(written)[-1])
    dst = os.path.join(node_dir, f"{view}{ext}")
    if src != dst:
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
    return dst


def main():
    args = parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    clear_scene()
    print(f"[import] {args.mesh}", flush=True)
    obj = import_mesh(os.path.abspath(args.mesh))
    n_tri = len(obj.data.loop_triangles) or len(obj.data.polygons)
    print(f"[mesh  ] {len(obj.data.vertices):,} verts / "
          f"{len(obj.data.polygons):,} faces", flush=True)

    if args.yaw:
        obj.rotation_euler[2] += math.radians(args.yaw)
        bpy.context.view_layer.update()
        print(f"[yaw   ] rotated {args.yaw} deg about Z", flush=True)

    norm = normalize(obj)
    print(f"[norm  ] normalized size {['%.3f' % s for s in norm['normalized_size']]}",
          flush=True)
    shade_smooth(obj)
    make_material(obj, args.albedo)
    setup_world()

    scene = bpy.context.scene
    ortho_scale = args.margin  # longest axis is 1.0 after normalization
    setup_render(scene, args.res, args.samples)
    nodes = setup_compositor(scene, out)
    cam = setup_camera(scene, ortho_scale)

    cameras = {}
    todo = [args.only] if args.only else list(VIEWS)
    for view in todo:
        euler_deg, direction = VIEWS[view]
        cam.rotation_euler = Euler([math.radians(a) for a in euler_deg], "XYZ")
        cam.location = Vector(direction) * 2.0
        bpy.context.view_layer.update()

        for key, node in nodes.items():
            if not key.startswith("_"):
                node.file_slots[0].path = view + "_"

        ext_of = {"rgb": ".png", "mask": ".png",
                  "depth": ".exr", "normal": ".exr"}
        paths = {}
        for which in ("geometry", "beauty"):
            written = set_pass(scene, nodes, which, args.samples)
            print(f"[render] {view:<10} {which:<8} dir={direction}", flush=True)
            bpy.ops.render.render(write_still=False)
            for key in written:
                p = rename_frame_output(os.path.join(out, key), view,
                                        ext_of[key])
                paths[key] = os.path.relpath(p, out) if p else None

        m = cam.matrix_world
        cameras[view] = {
            "type": "ORTHO",
            "ortho_scale": ortho_scale,
            "location": list(cam.location),
            "rotation_euler_deg": list(euler_deg),
            "matrix_world": [list(r) for r in m],
            "view_direction": list((Vector((0, 0, -1))) @ m.to_3x3().inverted()),
            "outputs": paths,
        }
        print(f"         wrote {paths}", flush=True)

    meta = {
        "mesh": os.path.relpath(os.path.abspath(args.mesh),
                                os.path.dirname(out)),
        "resolution": [args.res, args.res],
        "samples": args.samples,
        "faces": len(obj.data.polygons),
        "vertices": len(obj.data.vertices),
        "normalization": norm,
        "yaw_deg": args.yaw,
        "ortho_scale": ortho_scale,
        "depth_units": "normalized object units; longest bbox axis == 1.0",
        "normal_space": "world",
        "geometry_pass": "1 sample, filter_size 0.01 (point sampled), no denoise",
        "beauty_pass": f"{args.samples} samples, filter_size 1.5, denoised",
        "background_depth": 1e10,
        "background": "#808080",
        "views": cameras,
    }
    shutil.rmtree(nodes["_scratch"], ignore_errors=True)
    with open(os.path.join(out, "cameras.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[done  ] " + out, flush=True)


if __name__ == "__main__":
    main()
