#!/usr/bin/env python3
"""Multi-view consistent generation on a Kaggle GPU: images only, no 3D.

MV-Adapter (SDXL, Apache-2.0) turns one image into orthographic views around
the vertical axis at chosen azimuths, drawn jointly so they agree. Normals are
a separate kernel (kaggle_normals.py): installing a normal estimator's pinned
requirements inside this one reinstalled PyTorch under a running process and
hung it. Everything 3D stays in this repo (stage2 reconstructs from these images).

  push <name> --image hero.png --mask hero_mask.png --prompt "..."
  status <name>
  pull <name>        -> data/<name>/mvgen/view_s<seed>_<view>.png + frame.json

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

sys.path.insert(0, os.path.dirname(__file__))
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
# Two patches to the reference-attention cache:
#  - accelerate's offload hooks rebuild every dict they pass to a module, so
#    the reference pass would fill a copy of its cache and leave the original
#    empty; a non-Mapping holder survives that
#  - the pipeline repeats every cached layer once per view and again for
#    guidance, and keeps all of it on the GPU: 28 copies for 14 views, which
#    ran a T4 out of memory. One copy is kept; each layer gets its expanded
#    batch when it asks for it
_pp = "/tmp/mva/mvadapter/pipelines/pipeline_mvadapter_i2mv_sdxl.py"
_src = open(_pp).read()
if "class _Cache" not in _src:
    _src = _src.replace("ref_hidden_states = {}\n", "ref_hidden_states = _Cache()\n", 1)
    a0 = _src.index("            ref_hidden_states = {\n                k: v.repeat_interleave(")
    a1 = _src.index("        cross_attention_kwargs = {", a0)
    _src = (_src[:a0] + "            ref_hidden_states.n = num_images_per_prompt\n"
            "        ref_hidden_states.cfg = self.do_classifier_free_guidance\n\n" + _src[a1:])
    _src = _src.replace('"ref_hidden_states": {k: v.clone() for k, v in ref_hidden_states.items()},',
                        '"ref_hidden_states": ref_hidden_states,', 1)
    _src = ("import torch as _t\n"
            "class _Cache:\n"
            "    def __init__(self): self.d = {}; self.n = 1; self.cfg = False\n"
            "    def __setitem__(self, k, v): self.d[k] = v\n"
            "    def __getitem__(self, k):\n"
            "        v = self.d[k]\n"
            "        r = v.expand(self.n, *v.shape[1:])\n"
            "        return _t.cat([_t.zeros_like(r), r], 0) if self.cfg else r.contiguous()\n"
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
VIEWS = CFG["views"]                 # [name, azimuth, elevation] in MV-Adapter's convention
az = [v[1] for v in VIEWS]; el = [v[2] for v in VIEWS]; n = len(VIEWS)
pipe.init_custom_adapter(num_views=n)
pipe.load_custom_adapter("huanngzh/mv-adapter", weight_name="mvadapter_i2mv_sdxl.safetensors")
pipe.to(dtype=dt)
# a T4 has 15 GB: keep only the part that is running on the GPU
pipe.enable_model_cpu_offload()
pipe.cond_encoder.to(device=dev, dtype=dt)
pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
H = Wd = 768
cams = get_orthogonal_camera(elevation_deg=el, distance=[1.8] * n, left=-0.55, right=0.55,
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
           "offset": [int(y0), int(x0)], "size": [H, Wd], "views": VIEWS,
           "ortho": [-0.55, 0.55], "distance": 1.8}, open(W + "/frame.json", "w"))
for seed in CFG["seeds"]:
    out = pipe(CFG["prompt"], height=H, width=Wd, num_inference_steps=30, guidance_scale=3.0,
               num_images_per_prompt=n, control_image=ctrl, control_conditioning_scale=1.0,
               reference_image=ref, reference_conditioning_scale=1.0,
               negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
               cross_attention_kwargs={"scale": 1.0},
               generator=torch.Generator(device=dev).manual_seed(seed)).images
    for v, im in zip(VIEWS, out):
        im.save(f"{W}/view_s{seed}_{v[0]}.png")
    print("views done, seed", seed, flush=True)
print('all views done', flush=True)
'''


# name, azimuth, elevation in MV-Adapter's convention: azimuth 0 is the front
# camera, 90 the camera on +x (this pipeline's azimuth 0), elevation > 0 above.
# zoo14 is the layout the zoo was rendered in: four level views, top and
# bottom, and the eight three-quarter views at +-45 degrees.
LAYOUTS = {
    "six": [["01_front", 0, 0], ["d045", 45, 0], ["02_right", 90, 0], ["03_back", 180, 0],
            ["04_left", 270, 0], ["d315", 315, 0]],
    "eight": [["01_front", 0, 0], ["d045", 45, 0], ["02_right", 90, 0], ["d135", 135, 0],
              ["03_back", 180, 0], ["d225", 225, 0], ["04_left", 270, 0], ["d315", 315, 0]],
    # zoo14 does NOT work with MV-Adapter i2mv: it was trained on level views
    # only and draws the +-45 and polar views level too (on the cat the four
    # level views agree 99.5%, the "elevated" ones fit better as level ones)
    "zoo14": [["01_front", 0, 0], ["02_right", 90, 0], ["03_back", 180, 0], ["04_left", 270, 0],
              ["05_top", 0, 89.99], ["06_bottom", 0, -89.99],
              ["07_az45_up", 45, 45], ["08_az135_up", 135, 45], ["09_az225_up", 225, 45],
              ["10_az315_up", 315, 45], ["11_az45_dn", 45, -45], ["12_az135_dn", 135, -45],
              ["13_az225_dn", 225, -45], ["14_az315_dn", 315, -45]],
}


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
    cfg = json.dumps({"prompt": a.prompt, "views": LAYOUTS[a.layout], "seeds": a.seeds})
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
    ap.add_argument("--layout", default="zoo14", choices=list(LAYOUTS))
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
