#!/usr/bin/env python3
"""Views drawn over the mesh's own geometry, from any camera (Kaggle GPU).

MV-Adapter's image+geometry mode (ig2mv, Apache-2.0) takes, per view, the
mesh's position map and normal map rendered from that camera, and draws the
reference image's object onto exactly that geometry. So the views cannot
disagree about shape -- the shape is the mesh -- and the camera can be any
camera: level views at azimuths the image-only mode was not trained on, the
top and the bottom it never drew, or a zoomed-in camera on one region.

The control maps are rendered here, in this pipeline's world frame (z up,
front camera on -y, the same frame MV-Adapter's cameras use), after the same
rescale MV-Adapter applies (largest |coordinate| -> 0.5). Every group of six
cameras is one joint generation. Out come the views and, for each, the
camera in the mesh's own frame, ready for mesh_fit.py.

  push <name> --mesh fit/mesh.ply --image hero.png --mask hero_mask.png \
       --groups ring6,diag6 --prompt "..."
  status <name>
  pull <name>   -> data/<name>/ig2mv/{cameras.json, rgb/, geom_mask/, control/}

Camera groups (azimuth in MV-Adapter's convention: 0 = front; elevation up +):
  ring6   0, 90, 180, 270 level, top, bottom  (what ig2mv was trained on)
  diag6   45, 135, 225, 315 level, top, bottom (turned 45 degrees)
  zoom:<cx>,<cy>,<cz>,<ortho>  the ring6 cameras centred on a world point with
          a narrower frame, for one region at higher resolution
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


def groups_from(spec):
    out = []
    for g in spec.split(";"):
        g = g.strip()
        if g == "ring6":
            cams = [[0, 0], [90, 0], [180, 0], [270, 0], [0, 89.99], [0, -89.99]]
            out.append({"name": "ring6", "cams": cams, "center": [0, 0, 0], "ortho": 1.1, "frame": "mv"})
        elif g == "diag6":
            cams = [[45, 0], [135, 0], [225, 0], [315, 0], [45, 89.99], [45, -89.99]]
            out.append({"name": "diag6", "cams": cams, "center": [0, 0, 0], "ortho": 1.1, "frame": "mv"})
        elif g.startswith("zoom:"):
            cx, cy, cz, o = (float(x) for x in g[5:].split(","))
            cams = [[0, 0], [90, 0], [180, 0], [270, 0], [0, 89.99], [0, -89.99]]
            out.append({"name": f"zoom{len(out)}", "cams": cams, "center": [cx, cy, cz], "ortho": o,
                        "frame": "mesh"})
        else:
            raise SystemExit(f"unknown camera group {g}")
    return out


RUNNER = r'''
import os, sys, subprocess, types, json, glob, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
W = "/kaggle/working"
t0 = time.time()
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/huanngzh/MV-Adapter", "/tmp/mva"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers>=0.31", "transformers", "peft",
                "huggingface_hub", "accelerate", "safetensors", "einops", "omegaconf", "timm", "kornia",
                "trimesh", "jaxtyping", "typeguard", "sentencepiece", "pytorch-lightning", "imageio"], check=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", "ninja",
                    "git+https://github.com/NVlabs/nvdiffrast"], capture_output=True, text=True)
print("installs", r.returncode, round(time.time() - t0), "s", flush=True)
class Stub(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return type(k, (), {})
for m in ["cvcuda", "open3d", "pymeshlab", "spandrel"]:
    mod = Stub(m); mod.__path__ = []
    sys.modules[m] = mod
sys.path.insert(0, "/tmp/mva")
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
import torch, numpy as np, trimesh
import nvdiffrast.torch as dr
from PIL import Image
from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from diffusers import AutoencoderKL

CFG = json.loads(__CFG__)
src = os.path.dirname(glob.glob("/kaggle/input/**/input.png", recursive=True)[0])
dev, dt = "cuda", torch.float16
H = Wd = 768

# --- geometry, rescaled as MV-Adapter's load_mesh(rescale=True) does ---------
mesh = trimesh.load(src + "/mesh.ply", process=False)
V = np.asarray(mesh.vertices, np.float64); F = np.asarray(mesh.faces, np.int64)
s = 0.5 / np.abs(V).max()
Vn = np.asarray(mesh.vertex_normals, np.float64)
vs = torch.tensor(V * s, device=dev, dtype=torch.float32)
vn = torch.tensor(Vn, device=dev, dtype=torch.float32)
fi = torch.tensor(F, device=dev, dtype=torch.int32)
glctx = dr.RasterizeCudaContext(device=dev)
print(f"mesh {len(V):,} vertices, {len(F):,} faces, rescale {s:.4f}", flush=True)

def camera(az, el, center):
    """ours: az 270 = front; MV-Adapter az a is ours 270 + a"""
    a = np.radians((270 + az) % 360); e = np.radians(el)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross([0, 0, 1], d); right /= np.linalg.norm(right)
    up = np.cross(d, right)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2] = right, up, d
    M[:3, 3] = np.asarray(center) + 2 * d
    return M

def clip(M, ortho, near=0.1, far=4.0):
    h = ortho / 2
    P = np.array([[1 / h, 0, 0, 0], [0, 1 / h, 0, 0],
                  [0, 0, -2 / (far - near), -(far + near) / (far - near)], [0, 0, 0, 1]])
    return P @ np.linalg.inv(M)

pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix"))
pipe.scheduler = ShiftSNRScheduler.from_scheduler(pipe.scheduler, shift_mode="interpolated", shift_scale=8.0)
pipe.init_custom_adapter(num_views=6, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)
pipe.load_custom_adapter("huanngzh/mv-adapter", weight_name="mvadapter_ig2mv_sdxl.safetensors")
pipe.to(dtype=dt)
pipe.enable_model_cpu_offload()
pipe.cond_encoder.to(device=dev, dtype=dt)
pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
print("pipeline ready", round(time.time() - t0), "s", flush=True)

img = np.array(Image.open(src + "/input.png"))
alpha = img[..., 3] > 127
ys, xs = np.nonzero(alpha)
crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
h, w = crop.shape[:2]; k = min(H * 0.9 / h, Wd * 0.9 / w)
crop = np.array(Image.fromarray(crop).resize((max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS))
canvas = np.zeros((H, Wd, 4), np.uint8)
y0 = (H - crop.shape[0]) // 2; x0 = (Wd - crop.shape[1]) // 2
canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
rgb = canvas[..., :3] / 255.0; al = canvas[..., 3:4] / 255.0
ref = Image.fromarray(((rgb * al + 0.5 * (1 - al)) * 255).astype(np.uint8))

for d in ("rgb", "geom_mask", "control"):
    os.makedirs(f"{W}/ig2mv/{d}", exist_ok=True)
cams_out = {}
for g in CFG["groups"]:
    # full-object groups use MV-Adapter's own frame (+-0.55 around the
    # rescaled mesh); zooms are given in the mesh's units
    osc = g["ortho"] if g["frame"] == "mv" else g["ortho"] * s
    Ms = [camera(az, el, np.asarray(g["center"]) * s) for az, el in g["cams"]]
    mvp = torch.tensor(np.stack([clip(M, osc) for M in Ms]), device=dev, dtype=torch.float32)
    vh = torch.cat([vs, torch.ones_like(vs[:, :1])], -1)
    with torch.no_grad():
        clp = vh @ mvp.transpose(-2, -1)
        rast, _ = dr.rasterize(glctx, clp, fi, resolution=[H, Wd])
        pos, _ = dr.interpolate(vs, rast, fi)
        nrm, _ = dr.interpolate(vn, rast, fi)
        nrm = torch.nn.functional.normalize(nrm, dim=-1)
        m = rast[..., 3:] > 0
        pos = torch.where(m, pos, torch.zeros_like(pos))
        nrm = torch.where(m, nrm, torch.zeros_like(nrm))
        ctrl = torch.cat([(pos + 0.5).clamp(0, 1), (nrm / 2 + 0.5).clamp(0, 1)], -1)
        ctrl = ctrl.flip(1)                                   # first row = top, like the images
        m = m.flip(1)
    control = ctrl.permute(0, 3, 1, 2).to(dev, dtype=dt)
    t1 = time.time()
    out = pipe(CFG["prompt"], height=H, width=Wd, num_inference_steps=CFG["steps"], guidance_scale=3.0,
               num_images_per_prompt=6, control_image=control, control_conditioning_scale=1.0,
               reference_image=ref, reference_conditioning_scale=1.0,
               negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
               cross_attention_kwargs={"scale": 1.0},
               generator=torch.Generator(device=dev).manual_seed(CFG["seed"])).images
    print(f"group {g['name']}: generated in {time.time() - t1:.0f}s", flush=True)
    for (az, el), M, im, mk, ct in zip(g["cams"], Ms, out, m, ctrl):
        name = f"{g['name']}_az{int(round(az)):03d}_el{int(round(el)):+03d}"
        im.save(f"{W}/ig2mv/rgb/{name}.png")
        Image.fromarray((mk[..., 0].cpu().numpy() * 255).astype(np.uint8)).save(f"{W}/ig2mv/geom_mask/{name}.png")
        Image.fromarray((ct[..., 3:].cpu().numpy() * 255).astype(np.uint8)).save(f"{W}/ig2mv/control/{name}_normal.png")
        # camera back in the mesh's own frame: same directions, centre and
        # frame undone from the rescale
        Mo = M.copy(); Mo[:3, 3] = M[:3, 3] / s
        cams_out[name] = {"matrix_world": Mo.tolist(), "ortho_scale": osc / s,
                          "group": g["name"], "az_mv": az, "el": el}
json.dump({"ortho_scale": 1.1 / s, "resolution": [H, Wd], "views": cams_out, "rescale": s},
          open(f"{W}/ig2mv/cameras.json", "w"), indent=1)
print("all done", round(time.time() - t0), "s", flush=True)
'''


def kid(user, name):
    return f"{user}/threed-ig2mv-{name.replace('_', '-')}"


def push(a):
    user = owner()
    d = os.path.join(STAGE, f"ig2mv_{a.name}")
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    rgb = np.asarray(Image.open(a.image).convert("RGB"))
    m = np.asarray(Image.open(a.mask).convert("L"))
    Image.fromarray(np.dstack([rgb, m])).save(os.path.join(d, "input.png"))
    shutil.copy(a.mesh, os.path.join(d, "mesh.ply"))
    slug = f"threed-ig2mv-{a.name.replace('_', '-')}-input"
    upload_dataset(user, slug, f"threed ig2mv {a.name} input", d)
    kdir = os.path.join(STAGE, f"kernel_ig2mv_{a.name}")
    shutil.rmtree(kdir, ignore_errors=True); os.makedirs(kdir)
    cfg = json.dumps({"prompt": a.prompt, "groups": groups_from(a.groups), "steps": a.steps, "seed": a.seed})
    open(os.path.join(kdir, "run.py"), "w").write(RUNNER.replace("__CFG__", repr(cfg)))
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
    ap.add_argument("--mesh")
    ap.add_argument("--image")
    ap.add_argument("--mask")
    ap.add_argument("--prompt", default="high quality")
    ap.add_argument("--groups", default="ring6;diag6", help="';'-separated camera groups")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
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
