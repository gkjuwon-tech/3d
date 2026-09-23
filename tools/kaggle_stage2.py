#!/usr/bin/env python3
"""Run the heavy half of stage 2 on Kaggle CPU sessions, one model per session.

Locally a model's stage 2 is hours of single-threaded solves competing for
four cores. Kaggle gives each session four more, and several sessions run at
once. Only images leave the machine -- masks, rgb, float16 normals and the
camera file; the mesh stage 1 rendered never does -- and everything is
private.

  push <name>    flatten data/<name>/{views,normals} into a private dataset
                 (Kaggle uploads subdirectories as archives, so files are
                 renamed dir__file and rebuilt on the other side), refresh the
                 code dataset, and start a private CPU script that runs
                 stage2.py --stop-after fuse
  status <name>
  pull <name>    fetch recon/{hull,depth,mesh...} into data/<name>/recon, after
                 which a local stage2.py run finds those steps done and goes on
                 to retopology and scoring

Run:
  python3 tools/kaggle_stage2.py push bunny
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.path.join(ROOT, "data", "_kaggle")
CODE_SLUG = "threed-stage2-code"


def owner():
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True,
                         text=True).stdout
    for line in out.splitlines():
        if "username" in line:
            return line.split(":", 1)[1].strip()
    # fall back to the owner of the account's own kernels
    out = subprocess.run(["kaggle", "kernels", "list", "--mine", "--csv"],
                         capture_output=True, text=True).stdout.splitlines()
    return out[1].split("/")[0] if len(out) > 1 else sys.exit("no kaggle user")


def sh(cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-2000:], flush=True)
    if check and r.returncode:
        sys.exit(f"failed: {' '.join(cmd)}")
    return r


def upload_dataset(user, slug, title, folder):
    json.dump({"title": title, "id": f"{user}/{slug}",
               "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(folder, "dataset-metadata.json"), "w"))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{user}/{slug}"],
                            capture_output=True, text=True).returncode == 0
    if exists:
        sh(["kaggle", "datasets", "version", "-p", folder, "-m", "update",
            "--dir-mode", "skip"])
    else:
        sh(["kaggle", "datasets", "create", "-p", folder, "--dir-mode", "skip"])
    for _ in range(120):   # a kernel started before processing ends sees nothing
        r = subprocess.run(["kaggle", "datasets", "status", f"{user}/{slug}"],
                           capture_output=True, text=True)
        if "ready" in r.stdout.lower():
            return
        time.sleep(10)
    sys.exit(f"dataset {slug} never became ready")


RUNNER = r'''
import os, subprocess, sys, shutil, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyamg", "plyfile"],
               check=True)
if %(gpu)d:
    os.environ["THREED_GPU"] = "1"
    try:
        import cupy
        cupy.zeros(1)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"],
                       check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
NAME = "%(name)s"
def locate(fname):
    for root, _, files in os.walk("/kaggle/input"):
        if fname in files:
            return root
    raise SystemExit("missing " + fname)
code = locate("stage2.py")
data = locate("views__cameras.json")
W = "/kaggle/working"
os.makedirs(W + "/code/tools", exist_ok=True)
for f in os.listdir(code):
    if f.endswith(".py"):
        dst = "code/" if f == "stage2.py" else "code/tools/"
        shutil.copy(os.path.join(code, f), W + "/" + dst + (f if f == "stage2.py" else f.replace("tools__", "")))
d = W + "/code/data/" + NAME
for f in os.listdir(data):
    if "__" not in f:
        continue
    parts = f.split("__")
    dst = os.path.join(d, *parts)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(data, f), dst)
t0 = time.time()
r = subprocess.run([sys.executable, W + "/code/stage2.py", "--name", NAME,
                    "--workers", "%(workers)d", "--stop-after", "fuse",
                    "--relief-scale", "%(relief)d"], cwd=W + "/code")
print("stage2 exit", r.returncode, "minutes", (time.time() - t0) / 60)
rec = d + "/recon"
out = W + "/out"
os.makedirs(out, exist_ok=True)
for rel in ["stage2.log", "mesh.ply", "mesh_occ.npz", "hull/hull.ply", "hull/grid.npz"]:
    p = os.path.join(rec, rel)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(out, rel.replace("/", "__")))
import numpy as np
for f in os.listdir(rec + "/depth"):
    if f.endswith(".npy") and "_" in f:
        a = np.load(os.path.join(rec, "depth", f))
        np.save(os.path.join(out, "depth__" + f), a.astype(np.float16) if "_cost" not in f else a)
shutil.rmtree(W + "/code")
'''


def push(name, workers, gpu=False, reuse_data=False, relief=2):
    user = owner()
    code_dir = os.path.join(STAGE, "code")
    shutil.rmtree(code_dir, ignore_errors=True)
    os.makedirs(code_dir)
    shutil.copy(os.path.join(ROOT, "stage2.py"), code_dir)
    for f in glob.glob(os.path.join(ROOT, "tools", "*.py")):
        shutil.copy(f, os.path.join(code_dir, "tools__" + os.path.basename(f)))
    upload_dataset(user, CODE_SLUG, "threed stage2 code", code_dir)

    slug = f"threed-{name.replace('_', '-')}-images"
    have = subprocess.run(["kaggle", "datasets", "status", f"{user}/{slug}"],
                          capture_output=True, text=True).returncode == 0
    if not (reuse_data and have):
        src = os.path.join(ROOT, "data", name)
        ddir = os.path.join(STAGE, name)
        shutil.rmtree(ddir, ignore_errors=True)
        os.makedirs(ddir)
        shutil.copy(os.path.join(src, "views", "cameras.json"),
                    os.path.join(ddir, "views__cameras.json"))
        for sub in ("mask", "rgb"):
            for f in glob.glob(os.path.join(src, "views", sub, "*.png")):
                shutil.copy(f, os.path.join(ddir, f"views__{sub}__{os.path.basename(f)}"))
        for f in glob.glob(os.path.join(src, "normals", "*.npy")):
            np.save(os.path.join(ddir, "normals__" + os.path.basename(f)),
                    np.load(f).astype(np.float16))
        upload_dataset(user, slug, f"threed {name} images", ddir)
        shutil.rmtree(ddir, ignore_errors=True)

    kname = f"threed-stage2-{name.replace('_', '-')}" + ("-gpu" if gpu else "")
    kdir = os.path.join(STAGE, f"kernel_{name}{'_gpu' if gpu else ''}")
    shutil.rmtree(kdir, ignore_errors=True)
    os.makedirs(kdir)
    open(os.path.join(kdir, "run.py"), "w").write(
        RUNNER % {"name": name, "workers": workers, "gpu": int(gpu),
                  "relief": relief})
    json.dump({"id": f"{user}/{kname}",
               "title": kname.replace("-", " "),
               "code_file": "run.py", "language": "python",
               "kernel_type": "script", "is_private": True,
               "enable_gpu": bool(gpu), "enable_internet": True,
               "dataset_sources": [f"{user}/{CODE_SLUG}", f"{user}/{slug}"],
               "competition_sources": [], "kernel_sources": []},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
    sh(["kaggle", "kernels", "push", "-p", kdir])


def kernel_id(user, name, gpu):
    return f"{user}/threed-stage2-{name.replace('_', '-')}" + ("-gpu" if gpu else "")


def status(name, gpu=False):
    sh(["kaggle", "kernels", "status", kernel_id(owner(), name, gpu)], check=False)


def pull(name, gpu=False, into=None):
    user = owner()
    tmp = os.path.join(STAGE, f"pull_{name}")
    shutil.rmtree(tmp, ignore_errors=True)
    sh(["kaggle", "kernels", "output", kernel_id(user, name, gpu), "-p", tmp])
    rec = os.path.join(ROOT, "data", into or name, "recon")
    for f in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
        base = os.path.basename(f)
        if not os.path.isfile(f):
            continue
        dst = os.path.join(rec, *base.split("__"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if base.startswith("depth__") and base.endswith(".npy") and "_cost" not in base:
            np.save(dst, np.load(f).astype(np.float32))
        else:
            shutil.copy(f, dst)
    print("pulled into", rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status", "pull"])
    ap.add_argument("name")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu", action="store_true",
                    help="a GPU session running the CuPy backend")
    ap.add_argument("--reuse-data", action="store_true",
                    help="keep an already uploaded image dataset")
    ap.add_argument("--relief-scale", type=int, default=2)
    ap.add_argument("--into", default=None,
                    help="pull into data/<into>/recon instead of data/<name>")
    a = ap.parse_args()
    {"push": lambda: push(a.name, a.workers, a.gpu, a.reuse_data, a.relief_scale),
     "status": lambda: status(a.name, a.gpu),
     "pull": lambda: pull(a.name, a.gpu, a.into)}[a.cmd]()


if __name__ == "__main__":
    main()
