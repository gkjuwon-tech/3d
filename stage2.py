#!/usr/bin/env python3
"""Stage 2 -- images in, a 3D mesh out. No mesh, no learned model, no GPU.

Reads only data/<name>/views/{mask,rgb,cameras.json} and data/<name>/normals.

  1. hull     visual hull as an exact distance field from the silhouettes,
              and its depth per view by sphere tracing (tools/hull_field.py)
  2. depth    per view: normals integrated robustly, placed in depth by
              matching normals across the other 13 views, re-solved against
              those anchors (tools/depth_mv.py), several views at a time
  3. fuse     hull field and depth maps fused into one continuous signed
              distance -- a robust one: weighted median over the views, then
              the mean of those that agree -- and meshed at its zero level
              (tools/fuse_field.py)
  3b. pass 2  (--passes 2, the default) where two or more anchored views
              agree in that fusion, the surface is rendered back into every
              view as extra anchors (tools/consensus.py), each view's depth
              is re-solved against them (depth_mv.py --pass1, solve only),
              and the result is fused again. A view's depth is only right
              near its anchors; this gives it the anchors the other views
              found.
  4. retopo   QuadriFlow quad base plus a subdivided, shrink-wrapped quad
              mesh carrying the detail (tools/retopo.py)
  5. score    only if --gt-mesh is given: Chamfer, containment, volume,
              F-score and normal error against the mesh stage 1 rendered,
              and showcase renders of both

Every step is skipped when its output already exists, so an interrupted run
resumes where it stopped; --force redoes everything.

Confirmed version (v3). On Lucy, 14 views of 2048^2:
  GPU, 2x T4    5.0 min  (hull 0.7, depth 3.9, fuse 0.4)
  CPU, 4 cores  about 30 min
  Chamfer 0.00169, F@1 72.7%, normal median 8.1 deg

Run:
  python3 stage2.py --name bunny --gpu              # on a CUDA machine
  python3 stage2.py --name bunny                    # CPU
  python3 stage2.py --name bunny --gt-mesh assets/bunny.ply   # and score it
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
BLENDER = os.path.join(ROOT, "assets", "blender", "blender")
PY = sys.executable


def run(cmd, log, parallel=False, env=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    f = open(log, "a")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                         env=env)
    if parallel:
        return p
    if p.wait():
        sys.exit(f"failed ({p.returncode}); see {log}")


def tool(name):
    return os.path.join(ROOT, "tools", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--views", default=None, help="override data/<name>/views")
    ap.add_argument("--normals", default=None, help="override data/<name>/normals")
    ap.add_argument("--gt-mesh", default=None, help="score against this mesh")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--quads", type=int, default=40000)
    ap.add_argument("--relief-scale", type=int, default=2,
                    help="passed to depth_mv: integrate the relief on k x k "
                         "blocks (2 = at 1024 for 2048 images)")
    ap.add_argument("--gpu", action="store_true",
                    help="run every step on an NVIDIA GPU through CuPy "
                         "(same code, same results; THREED_GPU=1 does the same)")
    ap.add_argument("--stop-after", default=None, choices=["hull", "depth", "fuse"],
                    help="end early, e.g. on a machine without Blender")
    ap.add_argument("--passes", type=int, default=2, choices=[1, 2],
                    help="2: re-solve every view against the surface the "
                         "others agreed on in the first fusion, and fuse again")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.gpu:
        os.environ["THREED_GPU"] = "1"   # inherited by every step below

    d = os.path.join(a.data, a.name)
    views = a.views or os.path.join(d, "views")
    normals = a.normals or os.path.join(d, "normals")
    rec = os.path.join(d, "recon")
    hull, depth = os.path.join(rec, "hull"), os.path.join(rec, "depth")
    os.makedirs(depth, exist_ok=True)
    log = os.path.join(rec, "stage2.log")
    meta = json.load(open(os.path.join(views, "cameras.json")))
    names = list(meta["views"])
    T = {}
    t_all = time.time()
    # a fused mesh fetched from elsewhere (tools/kaggle_stage2.py pull) means
    # hull, depth and fusion are done, even though their bulky intermediates
    # were never copied here
    fused = os.path.exists(os.path.join(rec, "mesh.ply")) and not a.force

    # 1. hull -----------------------------------------------------------------
    t0 = time.time()
    if not fused and (a.force or not os.path.exists(
            os.path.join(hull, "views", "depth_npy", f"{names[-1]}.npy"))):
        run([PY, tool("hull_field.py"), "--views", views, "--out", hull], log)
    T["hull"] = time.time() - t0
    if a.stop_after == "hull":
        return

    # 2. depth, several views at once ------------------------------------------
    # on a multi-GPU machine each worker gets a device of its own; sharing
    # one serialises their solves
    ngpu = 0
    if os.environ.get("THREED_GPU") == "1":
        try:
            ngpu = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                      text=True).stdout.strip().splitlines())
        except OSError:
            ngpu = 0

    def depth_pass(out, extra):
        os.makedirs(out, exist_ok=True)
        todo = [] if fused else [v for v in names if a.force or
                                 not os.path.exists(os.path.join(out, f"{v}_cost.npy"))]
        chunks = [todo[i::a.workers] for i in range(a.workers)]
        procs = []
        for i, c in enumerate(chunks):
            if not c:
                continue
            env = dict(os.environ)
            if ngpu > 1:
                env["CUDA_VISIBLE_DEVICES"] = str(i % ngpu)
            procs.append(run([PY, tool("depth_mv.py"), "--views", views,
                              "--hull-views", os.path.join(hull, "views"),
                              "--normals-dir", normals, "--out", out,
                              "--relief-scale", str(a.relief_scale),
                              "--only-views", ",".join(c)] + extra, log,
                             parallel=True, env=env))
        for p in procs:
            if p.wait():
                sys.exit(f"depth worker failed; see {log}")

    def fuse(depth_dir, out, extra=()):
        if a.force or not os.path.exists(out + ".ply"):
            run([PY, tool("fuse_field.py"), "--occ", os.path.join(hull, "grid.npz"),
                 "--views", views, "--depth-dir", depth_dir, "--normals-dir", normals,
                 "--hull-cache", os.path.join(hull, "H.npy"), "--out", out]
                + list(extra), log)

    t0 = time.time()
    depth_pass(depth, ["--save-pass1"] if a.passes == 2 else [])
    T["depth"] = time.time() - t0
    if a.stop_after == "depth":
        return

    # 3. fuse (and the second pass) ---------------------------------------------
    t0 = time.time()
    mesh = os.path.join(rec, "mesh")
    if a.passes == 1:
        fuse(depth, mesh)
    elif not fused:
        f1 = os.path.join(rec, "fuse1")
        fuse(depth, f1, ["--support-out", f1 + "_support.npy"])
        T["fuse1"] = time.time() - t0
        t0 = time.time()
        cons = os.path.join(rec, "consensus")
        if a.force or not os.path.exists(os.path.join(cons, f"{names[-1]}.npy")):
            run([PY, tool("consensus.py"), "--field", f1 + "_field.npy",
                 "--support", f1 + "_support.npy", "--grid", f1 + "_occ.npz",
                 "--views", views, "--hull-views", os.path.join(hull, "views"),
                 "--out", cons], log)
        depth2 = os.path.join(rec, "depth2")
        depth_pass(depth2, ["--pass1", depth, "--consensus", cons])
        T["pass2"] = time.time() - t0
        t0 = time.time()
        fuse(depth2, mesh)
    T["fuse"] = time.time() - t0
    if a.stop_after == "fuse":
        print("timings: " + "  ".join(f"{k} {v/60:.1f}min" for k, v in T.items()))
        return

    # 4. retopo ---------------------------------------------------------------
    t0 = time.time()
    rdir = os.path.join(rec, "retopo")
    if a.force or not os.path.exists(os.path.join(rdir, f"{a.name}_quads_detail.obj")):
        run([BLENDER, "-b", "-P", tool("retopo.py"), "--", "--mesh", mesh + ".ply",
             "--out-dir", rdir, "--name", a.name, "--faces", str(a.quads)], log)
    T["retopo"] = time.time() - t0

    # 5. score ----------------------------------------------------------------
    if a.gt_mesh:
        t0 = time.time()
        rep = os.path.join(rec, "report.txt")
        with open(rep, "w") as f:
            for target in (os.path.join(hull, "hull.ply"), mesh + ".ply"):
                f.write(f"== {os.path.basename(target)}\n")
                f.flush()
                subprocess.run([PY, tool("eval_hull.py"), "--gt-mesh", a.gt_mesh,
                                "--views", views, "--recon", target],
                               cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            f.flush()
            subprocess.run([PY, tool("eval_surface.py"), "--gt-mesh", a.gt_mesh,
                            "--views", views, "--cache",
                            os.path.join(d, "gt_samples.npz"),
                            "--regions", "lucy" if a.name.startswith("lucy") else "none",
                            "--recon", os.path.join(hull, "hull.ply"), mesh + ".ply"],
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            f.flush()
            if os.path.exists(mesh + "_field.npy"):   # not fetched from Kaggle
                subprocess.run([PY, tool("containment.py"), "--field",
                                mesh + "_field.npy", "--occ",
                                os.path.join(hull, "grid.npz"),
                                "--gt-mesh", a.gt_mesh, "--views", views,
                                "--cache", os.path.join(d, "gt_samples.npz")],
                               cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        for tag, target in (("gt", a.gt_mesh), ("hull", os.path.join(hull, "hull.ply")),
                            ("recon", mesh + ".ply")):
            out = os.path.join(rec, "show", tag)
            if not os.path.exists(os.path.join(out, "d_low_hero.png")):
                run([BLENDER, "-b", "-P", tool("showcase.py"), "--", "--mesh",
                     os.path.abspath(target), "--out", out, "--res", "900",
                     "--samples", "64"], log)
        T["score"] = time.time() - t0
        print(open(rep).read())
    print("timings: " + "  ".join(f"{k} {v/60:.1f}min" for k, v in T.items())
          + f"  total {(time.time()-t_all)/60:.1f}min")


if __name__ == "__main__":
    main()
