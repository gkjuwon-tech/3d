#!/usr/bin/env python3
"""Era3D on a Kaggle GPU: one image in, six views of colour AND normals out,
drawn jointly (cross-domain attention), so the normals agree with each other
and with the colours -- unlike a single-image estimator run view by view.

Era3D (weights Apache-2.0, pengHTYX/MacLab-Era3D-512-6view) draws 512 px views
at azimuths 0, 45, 90, 180, 270, 315, level, orthographic, and estimates the
input's elevation and focal length itself. Its code was written for diffusers
0.26; the kernel keeps the image's own diffusers (a pinned install once
reinstalled PyTorch under a running kernel) and aliases the modules that moved,
and uses PyTorch attention instead of xformers.

  push <name> --image hero.png --mask hero_mask.png
  status <name>
  pull <name>   -> data/<name>/era3d/{color,normal}_<view>.png, normal_<view>.npy
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_stage2 import ROOT, STAGE, owner, sh, upload_dataset  # noqa: E402

RUNNER = r'''
import os, sys, glob, subprocess, importlib, time, json
W = "/kaggle/working"
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/pengHTYX/Era3D", "/tmp/era"],
               check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "omegaconf", "einops", "icecream"], check=True)
import diffusers
print("diffusers", diffusers.__version__, flush=True)
for old, new in [("diffusers.models.dual_transformer_2d", "diffusers.models.transformers.dual_transformer_2d"),
                 ("diffusers.models.unet_2d_blocks", "diffusers.models.unets.unet_2d_blocks"),
                 ("diffusers.models.unet_2d_condition", "diffusers.models.unets.unet_2d_condition"),
                 ("diffusers.models.transformer_2d", "diffusers.models.transformers.transformer_2d"),
                 ("diffusers.models.controlnet", "diffusers.models.controlnets.controlnet")]:
    try:
        sys.modules.setdefault(old, importlib.import_module(new))
    except Exception as e:
        print("alias", old, "skipped:", repr(e)[:120], flush=True)
import transformers
if not hasattr(transformers, "CLIPFeatureExtractor"):     # removed in newer transformers
    transformers.CLIPFeatureExtractor = transformers.CLIPImageProcessor
sys.path.insert(0, "/tmp/era")
os.chdir("/tmp/era")
import torch, numpy as np
from PIL import Image
from einops import rearrange
from mvdiffusion.data.single_image_dataset import SingleImageDataset
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
inp = glob.glob("/kaggle/input/**/input.png", recursive=True)[0]
os.makedirs("/tmp/in", exist_ok=True)
Image.open(inp).save("/tmp/in/input.png")
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("pengHTYX/MacLab-Era3D-512-6view",
                                                   torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
ds = SingleImageDataset(root_dir="/tmp/in", num_views=6, img_wh=[512, 512], bg_color="white",
                        crop_size=420, prompt_embeds_path="mvdiffusion/data/fixed_prompt_embeds_6view")
b = ds[0]
imgs_in = torch.cat([b["imgs_in"][None]] * 2, 0)
imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
pe = torch.cat([b["normal_prompt_embeddings"][None], b["color_prompt_embeddings"][None]], 0)
pe = rearrange(pe, "B Nv N C -> (B Nv) N C")
VIEWS = ["front", "front_right", "right", "back", "left", "front_left"]
g = torch.Generator(device="cuda").manual_seed(42)
t0 = time.time()
with torch.autocast("cuda"):
    out = pipe(imgs_in, None, prompt_embeds=pe, generator=g, guidance_scale=3.0, output_type="pt",
               num_images_per_prompt=1, num_inference_steps=40, eta=1.0).images
print("generated in", round(time.time() - t0), "s", flush=True)
n = out.shape[0] // 2
for j, v in enumerate(VIEWS):
    nrm = out[j].float().permute(1, 2, 0).cpu().numpy()
    clr = out[n + j].float().permute(1, 2, 0).cpu().numpy()
    Image.fromarray((nrm.clip(0, 1) * 255).astype(np.uint8)).save(f"{W}/normal_{v}.png")
    Image.fromarray((clr.clip(0, 1) * 255).astype(np.uint8)).save(f"{W}/color_{v}.png")
    np.save(f"{W}/normal_{v}.npy", (nrm * 2 - 1).astype(np.float16))
Image.fromarray((b["imgs_in"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(f"{W}/input_512.png")
print("all done", flush=True)
'''


def kid(user, name):
    return f"{user}/threed-era3d-{name.replace('_', '-')}"


def push(a):
    user = owner()
    ddir = os.path.join(STAGE, f"era3d_{a.name}")
    shutil.rmtree(ddir, ignore_errors=True); os.makedirs(ddir)
    rgb = np.asarray(Image.open(a.image).convert("RGB"))
    m = np.asarray(Image.open(a.mask).convert("L"))
    if m.shape != rgb.shape[:2]:
        m = np.asarray(Image.fromarray(m).resize(rgb.shape[1::-1], Image.NEAREST))
    Image.fromarray(np.dstack([rgb, m])).save(os.path.join(ddir, "input.png"))
    slug = f"threed-era3d-{a.name.replace('_', '-')}-input"
    upload_dataset(user, slug, f"threed era3d {a.name} input", ddir)
    kdir = os.path.join(STAGE, f"kernel_era3d_{a.name}")
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
    ap.add_argument("--image")
    ap.add_argument("--mask")
    a = ap.parse_args()
    if a.cmd == "push":
        push(a)
    elif a.cmd == "status":
        sh(["kaggle", "kernels", "status", kid(owner(), a.name)], check=False)
    else:
        out = os.path.join(ROOT, "data", a.name, "era3d")
        os.makedirs(out, exist_ok=True)
        sh(["kaggle", "kernels", "output", kid(owner(), a.name), "-p", out, "-o"], check=False)


if __name__ == "__main__":
    main()
