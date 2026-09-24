#!/usr/bin/env python3
"""Per-view normal maps on a Kaggle GPU: Marigold normals (diffusers built-in).

Nothing is pip-installed: the Kaggle image's diffusers already carries the
Marigold pipeline (installing another estimator's pinned requirements once
reinstalled PyTorch under a running kernel and hung it). Views go in with the
background flattened to mid-grey; axis signs are fixed afterwards from the
silhouette (tools/mvgen_views.py normals).

  push <name> --views data/<name>/views        (rgb/ + mask/)
  status <name>
  pull <name>   -> data/<name>/normals_raw/<view>.npy (+ .png)
  then: python3 tools/mvgen_views.py normals --src data/<name>/normals_raw --out data/<name>
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_stage2 import ROOT, STAGE, owner, sh, upload_dataset  # noqa: E402

RUNNER = r'''
import os, sys, glob, json, subprocess, time
import numpy as np, torch
from PIL import Image
W = "/kaggle/working"
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
src = os.path.dirname(glob.glob("/kaggle/input/**/*.rgba.png", recursive=True)[0])
files = sorted(glob.glob(src + "/*.rgba.png"))
t0 = time.time()
from diffusers import MarigoldNormalsPipeline
try:
    mp = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", variant="fp16",
                                                 torch_dtype=torch.float16).to("cuda")
except Exception as e:
    print("no fp16 variant:", repr(e)[:200], flush=True)
    mp = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1",
                                                 torch_dtype=torch.float16).to("cuda")
name = "marigold-normals-v1-1"
def normal(im):
    a = np.asarray(im.convert("RGBA")).astype(np.float32) / 255
    rgb = a[..., :3] * a[..., 3:] + 0.5 * (1 - a[..., 3:])
    r = mp(Image.fromarray((rgb * 255).astype(np.uint8)), num_inference_steps=4, ensemble_size=10,
           processing_resolution=768)
    return np.asarray(r.prediction[0]).astype(np.float32)
print("estimator", name, "loaded in", int(time.time() - t0), "s", flush=True)
for f in files:
    v = os.path.basename(f)[:-len(".rgba.png")]
    n = normal(Image.open(f))
    np.save(f"{W}/{v}.npy", n.astype(np.float16))
    Image.fromarray(((n + 1) / 2 * 255).clip(0, 255).astype(np.uint8)).save(f"{W}/{v}.png")
    print("normal", v, n.shape, flush=True)
open(W + "/estimator.txt", "w").write(name)
print("all normals done", flush=True)
'''


def kid(user, name):
    return f"{user}/threed-normals-{name.replace('_', '-')}"


def push(a):
    user = owner()
    ddir = os.path.join(STAGE, f"normals_{a.name}")
    shutil.rmtree(ddir, ignore_errors=True); os.makedirs(ddir)
    for p in sorted(glob.glob(f"{a.views}/rgb/*.png")):
        v = os.path.basename(p)[:-4]
        rgb = np.asarray(Image.open(p).convert("RGB"))
        m = np.asarray(Image.open(f"{a.views}/mask/{v}.png").convert("L"))
        Image.fromarray(np.dstack([rgb, m])).save(os.path.join(ddir, f"{v}.rgba.png"))
    slug = f"threed-normals-{a.name.replace('_', '-')}-input"
    upload_dataset(user, slug, f"threed normals {a.name} input", ddir)
    kdir = os.path.join(STAGE, f"kernel_normals_{a.name}")
    shutil.rmtree(kdir, ignore_errors=True); os.makedirs(kdir)
    open(os.path.join(kdir, "run.py"), "w").write(RUNNER)
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
    a = ap.parse_args()
    if a.cmd == "push":
        push(a)
    elif a.cmd == "status":
        sh(["kaggle", "kernels", "status", kid(owner(), a.name)], check=False)
    else:
        out = os.path.join(ROOT, "data", a.name, "normals_raw")
        os.makedirs(out, exist_ok=True)
        sh(["kaggle", "kernels", "output", kid(owner(), a.name), "-p", out, "-o"], check=False)


if __name__ == "__main__":
    main()
