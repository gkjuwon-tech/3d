#!/usr/bin/env python3
"""Which normal estimator to trust: measured against ground truth on a Kaggle GPU.

Renders from the zoo (buddha, dragon; clay, known camera-space normals) go in
at 768 px with their masks; every estimator runs in its own process under a
time limit, with nothing pip-installed that could touch PyTorch (a PyTorch
reinstall once hung a kernel for an hour). Each estimator's axis convention
is found once, as the sign flip that fits all images best, then the angular
error inside the mask is reported per image.

  push                          (data/_kaggle/normbench: <obj>__<view>.rgba.png + .gt.npy)
  status
  pull   -> data/_kaggle/normbench_out/{report.json, <est>/<image>.npy}
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_stage2 import ROOT, STAGE, owner, sh, upload_dataset  # noqa: E402

KERNEL = "threed-normbench"

EST = {
    "marigold": r'''
import torch
from diffusers import MarigoldNormalsPipeline
p = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", variant="fp16",
                                            torch_dtype=torch.float16).to("cuda")
def run(im):
    r = p(flat(im), num_inference_steps=4, ensemble_size=10, processing_resolution=768)
    return np.asarray(r.prediction[0]).astype(np.float32)
''',
    "stablenormal": r'''
import torch, sys, importlib
# their remote code imports diffusers modules from paths newer diffusers moved
for old, new in [("diffusers.models.controlnet", "diffusers.models.controlnets.controlnet"),
                 ("diffusers.models.unet_2d_condition", "diffusers.models.unets.unet_2d_condition"),
                 ("diffusers.models.unet_2d_blocks", "diffusers.models.unets.unet_2d_blocks")]:
    try:
        sys.modules.setdefault(old, importlib.import_module(new))
    except Exception as e:
        print("alias", old, "skipped:", e)
p = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
def run(im):
    return np.asarray(p(im, resolution=768, data_type="object")).astype(np.float32) / 255 * 2 - 1
''',
    "nirne": r'''
import torch, sys, importlib
# their remote code imports diffusers modules from paths newer diffusers moved
for old, new in [("diffusers.models.controlnet", "diffusers.models.controlnets.controlnet"),
                 ("diffusers.models.unet_2d_condition", "diffusers.models.unets.unet_2d_condition"),
                 ("diffusers.models.unet_2d_blocks", "diffusers.models.unets.unet_2d_blocks")]:
    try:
        sys.modules.setdefault(old, importlib.import_module(new))
    except Exception as e:
        print("alias", old, "skipped:", e)
p = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True,
                   yoso_version="yoso-normal-v1-8-1")
def run(im):
    return np.asarray(p(im, resolution=768, data_type="object")).astype(np.float32) / 255 * 2 - 1
''',
    "lotus": r'''
import torch, subprocess, sys
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/EnVision-Research/Lotus",
                "/tmp/lotus"], check=True)
sys.path.insert(0, "/tmp/lotus")
from pipeline import LotusGPipeline
p = LotusGPipeline.from_pretrained("jingheya/lotus-normal-g-v1-1", torch_dtype=torch.float16).to("cuda")
@torch.no_grad()
def run(im):
    x = torch.from_numpy(np.asarray(flat(im)).astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    x = x.cuda()
    task = torch.tensor([1, 0]).float().unsqueeze(0).cuda()
    task = torch.cat([torch.sin(task), torch.cos(task)], dim=-1)
    with torch.autocast("cuda"):
        out = p(rgb_in=x, prompt="", num_inference_steps=1, output_type="np",
                timesteps=[999], task_emb=task).images[0]
    return (out.astype(np.float32) * 2 - 1)
''',
}

RUNNER = r'''
import os, sys, glob, json, subprocess, time
import numpy as np
W = "/kaggle/working"
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
src = os.path.dirname(glob.glob("/kaggle/input/**/*.rgba.png", recursive=True)[0])
EST = __EST__
LIMIT = __LIMIT__
COMMON = """
import os, sys, glob, time, numpy as np
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
for f in sorted(glob.glob(SRC + "/*.rgba.png")):
    b = os.path.basename(f)[:-len(".rgba.png")]
    n = run(Image.open(f))
    np.save(OUT + "/" + b + ".npy", n.astype(np.float16))
    print("done", b, n.shape, round(time.time() - t0, 1), flush=True)
"""
    body = body.replace("SRC", repr(src)).replace("OUT", repr(out))
    open(f"{W}/run_{name}.py", "w").write(body)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, f"{W}/run_{name}.py"], timeout=LIMIT,
                           capture_output=True, text=True)
        print(f"== {name}: exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        print((r.stdout + r.stderr)[-2500:], flush=True)
    except subprocess.TimeoutExpired:
        print(f"== {name}: killed after {LIMIT}s", flush=True)

def err_deg(p, g, m):
    p = p / np.linalg.norm(p, axis=-1, keepdims=True).clip(1e-9)
    g = g / np.linalg.norm(g, axis=-1, keepdims=True).clip(1e-9)
    return np.degrees(np.arccos(np.clip((p * g).sum(-1), -1, 1)))[m]

from PIL import Image
report = {}
flips = [np.array([a, b, c]) for a in (1, -1) for b in (1, -1) for c in (1, -1)]
for name in EST:
    files = sorted(glob.glob(f"{W}/{name}/*.npy"))
    if not files:
        report[name] = "no output"
        continue
    data = []
    for f in files:
        b = os.path.basename(f)[:-4]
        p = np.load(f).astype(np.float32)
        g = np.load(f"{src}/{b}.gt.npy").astype(np.float32)
        a = np.asarray(Image.open(f"{src}/{b}.rgba.png"))
        m = (a[..., 3] > 200) & (np.linalg.norm(g, axis=-1) > 0.5)
        if p.shape[:2] != g.shape[:2]:
            p = np.stack([np.asarray(Image.fromarray(p[..., c]).resize(g.shape[1::-1], Image.BILINEAR))
                          for c in range(3)], -1)
        data.append((b, p, g, m))
    best = min(flips, key=lambda s: np.mean([np.median(err_deg(p * s, g, m)) for _, p, g, m in data]))
    per = {}
    for b, p, g, m in data:
        e = err_deg(p * best, g, m)
        per[b] = {"median": float(np.median(e)), "mean": float(e.mean()),
                  "within_11.25": float((e < 11.25).mean()), "within_22.5": float((e < 22.5).mean())}
    allm = [v["median"] for v in per.values()]
    report[name] = {"signs": best.tolist(), "median_of_medians": float(np.median(allm)),
                    "mean_error": float(np.mean([v["mean"] for v in per.values()])),
                    "within_11.25": float(np.mean([v["within_11.25"] for v in per.values()])),
                    "per_image": per}
    print(f"{name:14s} median {np.median(allm):5.1f} deg  mean {report[name]['mean_error']:5.1f}  "
          f"<11.25: {100*report[name]['within_11.25']:4.1f}%", flush=True)
json.dump(report, open(f"{W}/report.json", "w"), indent=1)
'''


def push(a):
    user = owner()
    src = os.path.join(STAGE, "normbench")
    upload_dataset(user, "threed-normbench-input", "threed normbench input", src)
    kdir = os.path.join(STAGE, "kernel_normbench")
    shutil.rmtree(kdir, ignore_errors=True); os.makedirs(kdir)
    est = {k: v for k, v in EST.items() if k in a.estimators}
    open(os.path.join(kdir, "run.py"), "w").write(RUNNER.replace("__EST__", repr(est)).replace("__LIMIT__", str(a.limit)))
    json.dump({"id": f"{user}/{KERNEL}", "title": KERNEL.replace("-", " "), "code_file": "run.py",
               "language": "python", "kernel_type": "script", "is_private": True,
               "enable_gpu": True, "enable_internet": True,
               "dataset_sources": [f"{user}/threed-normbench-input"],
               "competition_sources": [], "kernel_sources": []},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=1)
    sh(["kaggle", "kernels", "push", "-p", kdir])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status", "pull"])
    ap.add_argument("--estimators", nargs="+", default=list(EST))
    ap.add_argument("--limit", type=int, default=1200, help="seconds per estimator")
    a = ap.parse_args()
    if a.cmd == "push":
        push(a)
    elif a.cmd == "status":
        sh(["kaggle", "kernels", "status", f"{owner()}/{KERNEL}"], check=False)
    else:
        out = os.path.join(STAGE, "normbench_out")
        os.makedirs(out, exist_ok=True)
        sh(["kaggle", "kernels", "output", f"{owner()}/{KERNEL}", "-p", out, "-o"], check=False)


if __name__ == "__main__":
    main()
