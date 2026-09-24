"""Coarse proxy from a mesh: voxel remesh + smoothing, same object space.

Stands in for the proxy of a generated object when calibrating on a known one:
the large form survives, the surface detail (folds, hair, feathers) does not.

  assets/blender/blender -b -P tools/genlab/make_proxy.py -- \
      --mesh data/lucy12/recon/mesh.ply --out data/genlab/proxy/lucy_proxy.ply --voxel 0.012
"""
import argparse
import os
import sys

import bpy

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--voxel", type=float, default=0.012, help="object units (height 1.0)")
ap.add_argument("--smooth", type=int, default=10, help="Laplacian smoothing iterations")
ap.add_argument("--lam", type=float, default=0.5, help="Laplacian smoothing strength")
ap.add_argument("--decimate", type=float, default=0.0,
                help="instead of a voxel remesh, collapse to this face ratio first "
                     "(no voxel ripple; thin parts keep their topology)")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.ply_import(filepath=os.path.abspath(a.mesh))
obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
bpy.context.view_layer.objects.active = obj
if a.decimate:
    m = obj.modifiers.new("decimate", "DECIMATE"); m.ratio = a.decimate
else:
    m = obj.modifiers.new("remesh", "REMESH")
    m.mode = "VOXEL"; m.voxel_size = a.voxel
s = obj.modifiers.new("smooth", "LAPLACIANSMOOTH")
s.iterations = a.smooth; s.lambda_factor = a.lam; s.use_volume_preserve = True
obj.modifiers.new("tri", "TRIANGULATE")
for mod in list(obj.modifiers):
    bpy.ops.object.modifier_apply(modifier=mod.name)
print(f"[proxy] {len(obj.data.vertices):,} verts {len(obj.data.polygons):,} faces", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
bpy.ops.wm.ply_export(filepath=os.path.abspath(a.out), export_normals=False,
                      export_colors="NONE", export_uv=False, ascii_format=False)
