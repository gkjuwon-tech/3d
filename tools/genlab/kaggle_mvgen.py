#!/usr/bin/env python3
"""Multi-view consistent generation on a Kaggle GPU: images only, no 3D.

MV-Adapter (SDXL, Apache-2.0) turns one image into orthographic views around
the vertical axis at chosen azimuths, drawn jointly so they agree; StableNormal
(Apache-2.0 weights; Marigold as a fallback) then estimates each view's normal
map. Everything 3D stays in this repo (stage2 reconstructs from these images).

  push <name> --image hero.png --mask hero_mask.png --prompt "..."
  status <name>
  pull <name>        -> data/<name>/mvgen/{view_<az>.png, normal_<az>.npy/png}

Run:
  python3 tools/genlab/kaggle_mvgen.py push catmv --image data/cat/gen/hero_front_a.png \
      --mask data/cat/gen/hero_mask.png --prompt "a matte white clay sculpture of ..."
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kaggle_stage2 import ROOT, STAGE, owner, sh, upload_dataset  # noqa: E402

RUNNER = r'''
import os, sys, subprocess, types, json, glob
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
W = "/kaggle/working"
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/huanngzh/MV-Adapter", "/tmp/mva"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers>=0.31", "transformers", "peft",
                "huggingface_hub", "accelerate", "safetensors", "einops", "omegaconf", "timm", "kornia",
                "trimesh", "jaxtyping", "typeguard", "sentencepiece", "pytorch-lightning", "imageio"], check=True)
# texturing-only dependencies the package imports at load time: stub them out
class Stub(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return type(k, (), {})
for m in ["nvdiffrast", "nvdiffrast.torch", "cvcuda", "open3d", "pymeshlab", "spandrel"]:
    mod = Stub(m); mod.__path__ = []
    sys.modules[m] = mod
sys.path.insert(0, "/tmp/mva")
# accelerate's offload hooks rebuild every dict they pass to a module, so the
# reference pass would fill a copy of its cache and leave the original empty;
# a non-Mapping holder survives that
_pp = "/tmp/mva/mvadapter/pipelines/pipeline_mvadapter_i2mv_sdxl.py"
_src = open(_pp).read()
if "class _Cache" not in _src:
    _src = _src.replace("ref_hidden_states = {}\n", "ref_hidden_states = _Cache()\n", 1)
    _src = ("class _Cache:\n"
            "    def __init__(self): self.d = {}\n"
            "    def __setitem__(self, k, v): self.d[k] = v\n"
            "    def __getitem__(self, k): return self.d[k]\n"
            "    def items(self): return self.d.items()\n\n") + _src
    open(_pp, "w").write(_src)
import torch, numpy as np
from PIL import Image
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils.mesh_utils.camera import get_orthogonal_camera
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from diffusers import AutoencoderKL

CFG = json.loads(%(cfg)r)
inp = [p for p in glob.glob("/kaggle/input/**/input.png", recursive=True)][0]
dev, dt = "cuda", torch.float16
pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix"))
pipe.scheduler = ShiftSNRScheduler.from_scheduler(pipe.scheduler, shift_mode="interpolated", shift_scale=8.0)
az = CFG["azimuths"]; n = len(az)
pipe.init_custom_adapter(num_views=n)
pipe.load_custom_adapter("huanngzh/mv-adapter", weight_name="mvadapter_i2mv_sdxl.safetensors")
pipe.to(dtype=dt)
# a T4 has 15 GB: keep only the part that is running on the GPU
pipe.enable_model_cpu_offload()
pipe.cond_encoder.to(device=dev, dtype=dt)
pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
H = Wd = 768
cams = get_orthogonal_camera(elevation_deg=[0] * n, distance=[1.8] * n, left=-0.55, right=0.55,
                             bottom=-0.55, top=0.55, azimuth_deg=[x - 90 for x in az], device=dev)
pl = get_plucker_embeds_from_cameras_ortho(cams.c2w, [1.1] * n, Wd)
ctrl = ((pl + 1.0) / 2.0).to(dev, dtype=dt)

img = Image.open(inp)                       # RGBA: our own mask, no background remover
a = np.array(img); alpha = a[..., 3]
ys, xs = np.nonzero(alpha > 127)
crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
h, w = crop.shape[:2]
s = min(H * 0.9 / h, Wd * 0.9 / w)
crop = np.array(Image.fromarray(crop).resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS))
canvas = np.zeros((H, Wd, 4), np.uint8)
y0 = (H - crop.shape[0]) // 2; x0 = (Wd - crop.shape[1]) // 2
canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
rgb = canvas[..., :3].astype(np.float32) / 255; al = canvas[..., 3:4].astype(np.float32) / 255
ref = Image.fromarray(((rgb * al + 0.5 * (1 - al)) * 255).astype(np.uint8))
ref.save(W + "/reference.png")
json.dump({"scale": s, "crop_box": [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())],
           "offset": [int(y0), int(x0)], "size": [H, Wd], "azimuths": az,
           "ortho": [-0.55, 0.55], "distance": 1.8}, open(W + "/frame.json", "w"))
for seed in CFG["seeds"]:
    out = pipe(CFG["prompt"], height=H, width=Wd, num_inference_steps=50, guidance_scale=3.0,
               num_images_per_prompt=n, control_image=ctrl, control_conditioning_scale=1.0,
               reference_image=ref, reference_conditioning_scale=1.0,
               negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
               cross_attention_kwargs={"scale": 1.0},
               generator=torch.Generator(device=dev).manual_seed(seed)).images
    for a_, im in zip(az, out):
        im.save(f"{W}/view_s{seed}_{a_:03d}.png")
    print("views done, seed", seed, flush=True)
del pipe; torch.cuda.empty_cache()

# normals, one deterministic estimator for every view
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/Stable-X/StableNormal.git"], check=False)
    est = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
    name = "stablenormal"
    def normal(im):
        return np.asarray(est(im)).astype(np.float32) / 255 * 2 - 1
except Exception as e:
    print("StableNormal unavailable:", e, flush=True)
    from diffusers import MarigoldNormalsPipeline
    mp = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", variant="fp16",
                                                 torch_dtype=dt).to(dev)
    name = "marigold"
    def normal(im):
        r = mp(im, num_inference_steps=4, ensemble_size=5)
        return np.asarray(r.prediction[0]).astype(np.float32)
for f in sorted(glob.glob(W + "/view_*.png")):
    n_ = normal(Image.open(f).convert("RGB"))
    np.save(f.replace("view_", "normal_").replace(".png", ".npy"), n_.astype(np.float16))
    Image.fromarray(((n_ + 1) / 2 * 255).clip(0, 255).astype(np.uint8)).save(f.replace("view_", "normal_"))
print("normals by", name, flush=True)
open(W + "/estimator.txt", "w").write(name)
'''


def kernel_id(user, name):
    return f"{user}/threed-mvgen-{name.replace('_', '-')}"


def push(a):
    user = owner()
    ddir = os.path.join(STAGE, f"mvgen_{a.name}")
    shutil.rmtree(ddir, ignore_errors=True); os.makedirs(ddir)
    rgb = np.asarray(Image.open(a.image).convert("RGB"))
    m = np.asarray(Image.open(a.mask).convert("L"))
    if m.shape != rgb.shape[:2]:
        m = np.asarray(Image.fromarray(m).resize(rgb.shape[1::-1], Image.NEAREST))
    Image.fromarray(np.dstack([rgb, m])).save(os.path.join(ddir, "input.png"))
    slug = f"threed-mvgen-{a.name.replace('_', '-')}-input"
    upload_dataset(user, slug, f"threed mvgen {a.name} input", ddir)
    kdir = os.path.join(STAGE, f"kernel_mvgen_{a.name}")
    shutil.rmtree(kdir, ignore_errors=True); os.makedirs(kdir)
    cfg = json.dumps({"prompt": a.prompt, "azimuths": a.azimuths, "seeds": a.seeds})
    open(os.path.join(kdir, "run.py"), "w").write(RUNNER % {"cfg": cfg})
    kid = kernel_id(user, a.name)
    json.dump({"id": kid, "title": kid.split("/")[1].replace("-", " "), "code_file": "run.py",
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
    ap.add_argument("--prompt", default="")
    ap.add_argument("--azimuths", type=int, nargs="+", default=[0, 45, 90, 180, 270, 315])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    if a.cmd == "push":
        push(a)
    elif a.cmd == "status":
        sh(["kaggle", "kernels", "status", kernel_id(owner(), a.name)], check=False)
    else:
        out = os.path.join(ROOT, "data", a.name, "mvgen")
        os.makedirs(out, exist_ok=True)
        sh(["kaggle", "kernels", "output", kernel_id(owner(), a.name), "-p", out, "-o"], check=False)


if __name__ == "__main__":
    main()
