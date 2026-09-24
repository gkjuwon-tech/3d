#!/usr/bin/env python3
"""Per-view normal maps for a generated view set, on a Kaggle GPU.

Estimators (each in its own process under a time limit; nothing that could
reinstall PyTorch is pip-installed -- that once hung a kernel for an hour):

  normalcrafter  the views in turntable order as one video: a video normal
                 model keeps its frames consistent with each other, which is
                 what depth placement across views needs (Apache-2.0 weights,
                 on Stable Video Diffusion)
  nirne          Hi3DGen's single-image estimator (StableNormal_turbo,
                 yoso-normal-v1-8-1): the sharpest of the single-image ones
  stablenormal   lowest median error on the zoo benchmark (19.6 deg)
  marigold       diffusers built-in

  push <name> --views data/<name>/views [--estimators normalcrafter nirne]
  status <name>
  pull <name>   -> data/<name>/normals_raw/<estimator>/<view>.npy
  then: python3 tools/mvgen_views.py normals --src data/<name>/normals_raw/<estimator> --out data/<name>
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

SHIM = r'''
import sys, importlib
# remote code that imports diffusers modules from paths newer diffusers moved
for old, new in [("diffusers.models.controlnet", "diffusers.models.controlnets.controlnet"),
                 ("diffusers.models.unet_2d_condition", "diffusers.models.unets.unet_2d_condition"),
                 ("diffusers.models.unet_2d_blocks", "diffusers.models.unets.unet_2d_blocks")]:
    try:
        sys.modules.setdefault(old, importlib.import_module(new))
    except Exception as e:
        print("alias", old, "skipped:", e)
'''

# each defines run_all(images) -> list of HxWx3 float arrays, camera frame
EST = {
    "normalcrafter": r'''
import torch, subprocess
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/Binyr/NormalCrafter",
                "/tmp/normalcrafter"], check=True)
sys.path.insert(0, "/tmp/normalcrafter")
from diffusers import AutoencoderKLTemporalDecoder
from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
    "Yanrui95/NormalCrafter", subfolder="unet", low_cpu_mem_usage=True).to(dtype=torch.float16)
vae = AutoencoderKLTemporalDecoder.from_pretrained("Yanrui95/NormalCrafter", subfolder="vae").to(dtype=torch.float16)
pipe = NormalCrafterPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
                                             unet=unet, vae=vae, torch_dtype=torch.float16,
                                             variant="fp16")
pipe.enable_model_cpu_offload()
def run_all(images):
    frames = [flat(im) for im in images]
    with torch.inference_mode():
        res = pipe(frames, decode_chunk_size=4, time_step_size=len(frames),
                   window_size=len(frames)).frames[0]
    return [np.asarray(r, np.float32) for r in res]
''',
    "nirne": SHIM + r'''
import torch
p = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True,
                   yoso_version="yoso-normal-v1-8-1")
def run_all(images):
    return [np.asarray(p(im, resolution=768, data_type="object")).astype(np.float32) / 255 * 2 - 1
            for im in images]
''',
    "stablenormal": SHIM + r'''
import torch
p = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
def run_all(images):
    return [np.asarray(p(im, resolution=768, data_type="object")).astype(np.float32) / 255 * 2 - 1
            for im in images]
''',
    "marigold": r'''
import torch
from diffusers import MarigoldNormalsPipeline
p = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", variant="fp16",
                                            torch_dtype=torch.float16).to("cuda")
def run_all(images):
    return [np.asarray(p(flat(im), num_inference_steps=4, ensemble_size=10,
                         processing_resolution=768).prediction[0]).astype(np.float32)
            for im in images]
''',
}

RUNNER = r'''
import os, sys, glob, json, subprocess, time
W = "/kaggle/working"
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
src = os.path.dirname(glob.glob("/kaggle/input/**/order.txt", recursive=True)[0])
order = open(src + "/order.txt").read().split()
EST = __EST__
LIMIT = __LIMIT__
COMMON = """
import os, sys, time, numpy as np
from PIL import Image
def flat(im):
    a = np.asarray(im.convert("RGBA")).astype(np.float32) / 255
    return Image.fromarray(((a[..., :3] * a[..., 3:] + 0.5 * (1 - a[..., 3:])) * 255).astype(np.uint8))
"""
for name, code in EST.items():
    out = f"{W}/{name}"
    os.makedirs(out, exist_ok=True)
    body = COMMON + code + """
t0 = time.time()
views = ORDER
ims = [Image.open(SRC + "/" + v + ".rgba.png") for v in views]
res = run_all(ims)
for v, n in zip(views, res):
    np.save(OUT + "/" + v + ".npy", n.astype(np.float16))
    Image.fromarray(((n + 1) / 2 * 255).clip(0, 255).astype(np.uint8)).save(OUT + "/" + v + ".png")
print("done", len(res), "views in", round(time.time() - t0), "s", flush=True)
"""
    body = body.replace("SRC", repr(src)).replace("OUT", repr(out)).replace("ORDER", repr(order))
    open(f"{W}/run_{name}.py", "w").write(body)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, f"{W}/run_{name}.py"], timeout=LIMIT,
                           capture_output=True, text=True)
        print(f"== {name}: exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        print((r.stdout + r.stderr)[-3000:], flush=True)
    except subprocess.TimeoutExpired:
        print(f"== {name}: killed after {LIMIT}s", flush=True)
'''


def kid(user, name):
    return f"{user}/threed-normals-{name.replace('_', '-')}"


def push(a):
    user = owner()
    ddir = os.path.join(STAGE, f"normals_{a.name}")
    shutil.rmtree(ddir, ignore_errors=True); os.makedirs(ddir)
    order = list(json.load(open(f"{a.views}/cameras.json"))["views"])   # turntable order
    for v in order:
        rgb = np.asarray(Image.open(f"{a.views}/rgb/{v}.png").convert("RGB"))
        m = np.asarray(Image.open(f"{a.views}/mask/{v}.png").convert("L"))
        Image.fromarray(np.dstack([rgb, m])).save(os.path.join(ddir, f"{v}.rgba.png"))
    open(os.path.join(ddir, "order.txt"), "w").write("\n".join(order))
    slug = f"threed-normals-{a.name.replace('_', '-')}-input"
    upload_dataset(user, slug, f"threed normals {a.name} input", ddir)
    kdir = os.path.join(STAGE, f"kernel_normals_{a.name}")
    shutil.rmtree(kdir, ignore_errors=True); os.makedirs(kdir)
    est = {k: EST[k] for k in a.estimators}
    open(os.path.join(kdir, "run.py"), "w").write(
        RUNNER.replace("__EST__", repr(est)).replace("__LIMIT__", str(a.limit)))
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
    ap.add_argument("--estimators", nargs="+", default=["normalcrafter", "nirne"],
                    choices=list(EST))
    ap.add_argument("--limit", type=int, default=1500, help="seconds per estimator")
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
