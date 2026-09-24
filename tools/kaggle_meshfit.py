#!/usr/bin/env python3
"""Run tools/mesh_fit.py on a Kaggle GPU.

Uploads the view set (cameras, masks) and one normal set with the code,
installs nvdiffrast without its dependencies (it compiles its CUDA plugin on
first use; PyTorch is never touched), runs the fit under a time limit, and
brings back mesh.ply, the progress snapshots and the log.

  push <name> --views data/cat6/views --normals data/cat6_nirne/normals [--fit-args "..."]
       [--extra data/cathead/views:data/cathead/normals] [--init data/x/fit/mesh.ply]
  status <name>
  pull <name>   -> data/<name>/fit/
"""
import argparse
import glob
import json
import os
import shlex
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_stage2 import ROOT, STAGE, owner, sh, upload_dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

RUNNER = r'''
import os, sys, glob, shutil, subprocess, time
W = "/kaggle/working"
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
subprocess.run("which nvcc; nvcc --version | tail -1", shell=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/code__mesh_fit.py", recursive=True)[0])
root = "/tmp/job"
for f in os.listdir(src):
    if "__" in f:
        dst = os.path.join(root, *f.split("__"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(src, f), dst)
t0 = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", "ninja",
                    "git+https://github.com/NVlabs/nvdiffrast"], capture_output=True, text=True)
print("nvdiffrast install:", r.returncode, (r.stdout + r.stderr)[-800:], round(time.time() - t0), "s", flush=True)
cmd = [sys.executable, "/tmp/job/code/mesh_fit.py", "--views", "/tmp/job/views",
       "--normals", "/tmp/job/normals", "--out", W + "/fit"] + __ARGS__
print("$", " ".join(cmd), flush=True)
try:
    r = subprocess.run(cmd, timeout=__LIMIT__, capture_output=True, text=True, cwd="/tmp/job/code")
    print(r.stdout[-6000:], r.stderr[-4000:], flush=True)
    print("exit", r.returncode, flush=True)
except subprocess.TimeoutExpired as e:
    print("killed after __LIMIT__ s", (e.stdout or b"")[-3000:], flush=True)
'''


def kid(user, name):
    return f"{user}/threed-meshfit-{name.replace('_', '-')}"


def push(a):
    user = owner()
    d = os.path.join(STAGE, f"meshfit_{a.name}")
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    def add(prefix, views, normals):
        shutil.copy(os.path.join(views, "cameras.json"), os.path.join(d, f"{prefix}views__cameras.json"))
        for sub in ("mask", "valid"):
            for p in glob.glob(os.path.join(views, sub, "*.png")):
                shutil.copy(p, os.path.join(d, f"{prefix}views__{sub}__" + os.path.basename(p)))
        for p in glob.glob(os.path.join(normals, "*.npy")):
            shutil.copy(p, os.path.join(d, f"{prefix}normals__" + os.path.basename(p)))
    add("", a.views, a.normals)
    extra = []
    for i, pair in enumerate(a.extra):
        ev, en = pair.split(":")
        add(f"extra{i}__", ev, en)
        extra.append(i)
    fit_args = shlex.split(a.fit_args)
    if extra:
        fit_args += ["--extra-views"] + [f"/tmp/job/extra{i}/views" for i in extra]
        fit_args += ["--extra-normals"] + [f"/tmp/job/extra{i}/normals" for i in extra]
    if a.init:
        shutil.copy(a.init, os.path.join(d, "init__mesh.ply"))
        fit_args += ["--init", "/tmp/job/init/mesh.ply"]
    shutil.copy(os.path.join(HERE, "mesh_fit.py"), os.path.join(d, "code__mesh_fit.py"))
    for p in glob.glob(os.path.join(HERE, "meshfit", "*.py")) + [os.path.join(HERE, "meshfit", "LICENSE_THIRD_PARTY")]:
        shutil.copy(p, os.path.join(d, "code__meshfit__" + os.path.basename(p)))
    slug = f"threed-meshfit-{a.name.replace('_', '-')}-input"
    upload_dataset(user, slug, f"threed meshfit {a.name} input", d)
    kdir = os.path.join(STAGE, f"kernel_meshfit_{a.name}")
    shutil.rmtree(kdir, ignore_errors=True); os.makedirs(kdir)
    open(os.path.join(kdir, "run.py"), "w").write(
        RUNNER.replace("__ARGS__", repr(fit_args)).replace("__LIMIT__", str(a.limit)))
    k = kid(user, a.name)
    json.dump({"id": k, "title": k.split("/")[1].replace("-", " "), "code_file": "run.py",
               "language": "python", "kernel_type": "script", "is_private": True,
               "enable_gpu": True, "enable_internet": True,
               "dataset_sources": [f"{user}/{slug}"], "competition_sources": [], "kernel_sources": []},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
    sh(["kaggle", "kernels", "push", "-p", kdir])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status", "pull"])
    ap.add_argument("name")
    ap.add_argument("--views")
    ap.add_argument("--normals")
    ap.add_argument("--fit-args", default="")
    ap.add_argument("--extra", nargs="*", default=[], help="views_dir:normals_dir pairs")
    ap.add_argument("--init", default=None, help="mesh to start from")
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args()
    if a.cmd == "push":
        push(a)
    elif a.cmd == "status":
        sh(["kaggle", "kernels", "status", kid(owner(), a.name)], check=False)
    else:
        out = os.path.join(ROOT, "data", a.name)
        os.makedirs(out, exist_ok=True)
        sh(["kaggle", "kernels", "output", kid(owner(), a.name), "-p", out, "-o"], check=False)


if __name__ == "__main__":
    main()
