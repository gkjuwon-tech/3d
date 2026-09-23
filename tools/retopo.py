#!/usr/bin/env python3
"""Scripted retopology: an all-quad base mesh, and a subdivided quad mesh that
carries the reconstruction's detail. Runs inside Blender, no hand steps.

  1. The marching-cubes surface is triangle soup at voxel density (millions of
     faces), which is the wrong input for a quad remesher. Loose islands are
     dropped and a copy is voxel-remeshed to a working budget first.
  2. QuadriFlow (Huang et al. 2018, bundled with Blender) remeshes the copy to
     a target quad count: `<name>_quads.obj`.
  3. That base is subdivided (Catmull-Clark) and each new vertex is moved to
     the nearest point of the full-resolution surface (shrinkwrap), so the
     detail lives on clean quad topology: `<name>_quads_detail.obj`.

Run:
  assets/blender/blender -b -P tools/retopo.py -- --mesh out/fused.ply \
      --out-dir out/retopo --faces 40000 --levels 2
"""
import argparse
import os
import sys

import bmesh
import bpy


def args_():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--faces", type=int, default=40000, help="target quads")
    p.add_argument("--work", type=int, default=400000,
                   help="triangles the remesher is given")
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--min-island", type=float, default=0.002,
                   help="islands under this fraction of the faces are dropped")
    return p.parse_args(argv)


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    {".ply": bpy.ops.wm.ply_import, ".obj": bpy.ops.wm.obj_import}[ext](filepath=path)
    return [o for o in bpy.context.scene.objects if o.type == "MESH"][-1]


def drop_islands(obj, frac):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    seen, islands = set(), []
    for f in bm.faces:
        if f.index in seen:
            continue
        stack, comp = [f], []
        seen.add(f.index)
        while stack:
            g = stack.pop()
            comp.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index)
                        stack.append(h)
        islands.append(comp)
    total = len(bm.faces)
    doomed = [f for comp in islands if len(comp) < frac * total for f in comp]
    bmesh.ops.delete(bm, geom=doomed, context="FACES")
    bm.to_mesh(obj.data)
    bm.free()
    return len(islands), len(doomed)


def select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def export_obj(obj, path):
    select_only(obj)
    bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True,
                          export_normals=False, export_uv=False,
                          export_materials=False, export_triangulated_mesh=False,
                          apply_modifiers=True)


def main():
    a = args_()
    os.makedirs(a.out_dir, exist_ok=True)
    name = a.name or os.path.splitext(os.path.basename(a.mesh))[0]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    src = import_mesh(os.path.abspath(a.mesh))
    src.name = "source"
    n_isl, n_drop = drop_islands(src, a.min_island)
    print(f"[retopo] source {len(src.data.polygons):,} faces, {n_isl} islands, "
          f"dropped {n_drop:,} faces of small ones", flush=True)

    select_only(src)
    bpy.ops.object.duplicate()
    work = bpy.context.active_object
    work.name = "work"
    # QuadriFlow refuses anything non-manifold, and collapse decimation of a
    # marching-cubes surface produces non-manifold edges. The voxel remesher
    # (OpenVDB) always returns a closed manifold, and its voxel size can be
    # chosen from the surface area to land near the working budget: about
    # one quad per voxel-sized patch of surface.
    area = sum(p.area for p in work.data.polygons)
    work.data.remesh_voxel_size = (area / a.work) ** 0.5
    select_only(work)
    bpy.ops.object.voxel_remesh()
    # The remesher also closes internal voids into surfaces of their own, and
    # QuadriFlow rejects the lot: a void's surface is consistent with itself
    # but faces inward. Only the largest shell is remeshed; normals are then
    # made consistent and outward.
    n_shell, _ = drop_islands(work, 0.5)
    select_only(work)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[retopo] working copy {len(work.data.polygons):,} faces "
          f"(voxel remesh at {work.data.remesh_voxel_size:.5f}, kept the "
          f"largest of {n_shell} shells)", flush=True)

    # QuadriFlow's input check also rejects any edge shorter than 1e-4 in
    # object units, and in a frame where the model is one unit tall the voxel
    # remesher makes plenty of those. Scale up for the remesh, back down after.
    K = 1000.0
    select_only(work)
    work.scale = (K, K, K)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.quadriflow_remesh(mode="FACES", target_faces=a.faces,
                                     use_mesh_symmetry=False,
                                     use_preserve_sharp=False,
                                     use_preserve_boundary=False,
                                     smooth_normals=False, seed=0)
    work.scale = (1 / K, 1 / K, 1 / K)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if len(work.data.polygons) > 3 * a.faces:
        sys.exit("[retopo] QuadriFlow did not run (see the warning above)")
    quads = sum(1 for p in work.data.polygons if len(p.vertices) == 4)
    print(f"[retopo] quadriflow: {len(work.data.polygons):,} faces, "
          f"{100*quads/max(len(work.data.polygons),1):.1f}% quads", flush=True)
    export_obj(work, os.path.join(a.out_dir, f"{name}_quads.obj"))

    sub = work.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = a.levels
    sw = work.modifiers.new("wrap", "SHRINKWRAP")
    sw.target = src
    sw.wrap_method = "NEAREST_SURFACEPOINT"
    select_only(work)
    bpy.ops.object.modifier_apply(modifier="sub")
    bpy.ops.object.modifier_apply(modifier="wrap")
    print(f"[retopo] detail: {len(work.data.polygons):,} quads at level "
          f"{a.levels}", flush=True)
    export_obj(work, os.path.join(a.out_dir, f"{name}_quads_detail.obj"))
    print("[retopo] done", flush=True)


if __name__ == "__main__":
    main()
