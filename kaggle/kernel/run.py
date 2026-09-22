"""M2 per-view estimation on GPU.

Runs the depth and normal estimators over the six orthographic views and saves
raw float arrays. No fusion here and no calibration here -- this kernel only
produces the per-view estimates; the scale/shift solve and the scoring against
ground truth happen locally, where the ground truth lives.

Outputs to /kaggle/working/est/<model>/<view>.npy plus a manifest.
"""
import json
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image

IN = "/kaggle/input/lucy-m2-views"
OUT = "/kaggle/working/est"
VIEWS = ["01_front", "02_right", "03_back", "04_left", "05_top", "06_bottom"]

manifest = {"models": {}, "views": VIEWS, "errors": {}}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_input_root():
    """Locate the attached view set. The mount point is not assumed: a dataset
    still processing when the kernel starts simply is not there, and the failure
    reads as a missing file rather than a missing mount."""
    root = "/kaggle/input"
    if not os.path.isdir(root):
        raise FileNotFoundError("/kaggle/input does not exist; no data attached")
    entries = sorted(os.listdir(root))
    log("attached datasets:", entries)
    for e in entries:
        d = os.path.join(root, e)
        names = set(os.listdir(d)) if os.path.isdir(d) else set()
        log(f"  {e}: {sorted(names)[:8]}")
        if {"rgb", "rgb.zip"} & names:
            return d
    raise FileNotFoundError(
        f"no attached dataset contains rgb/ or rgb.zip; saw {entries}")


def resolve_input():
    """Directories may arrive zipped depending on how the dataset was uploaded;
    accept either form."""
    import zipfile
    src_root = find_input_root()
    base = "/kaggle/working/views"
    os.makedirs(base, exist_ok=True)
    for name in ("rgb", "mask"):
        src_dir = os.path.join(src_root, name)
        if os.path.isdir(src_dir):
            os.symlink(src_dir, os.path.join(base, name))
            log(f"  {name}/ found as a directory")
            continue
        src_zip = os.path.join(src_root, name + ".zip")
        if os.path.isfile(src_zip):
            with zipfile.ZipFile(src_zip) as z:
                z.extractall(os.path.join(base, name))
            log(f"  {name}.zip extracted -> "
                f"{len(os.listdir(os.path.join(base, name)))} files")
            continue
        raise FileNotFoundError(
            f"no {name} directory or zip under {src_root}")
    return base


def load_views():
    base = resolve_input()
    imgs = {}
    for v in VIEWS:
        p = os.path.join(base, "rgb", f"{v}.png")
        imgs[v] = Image.open(p).convert("RGB")
        log(f"loaded {v} {imgs[v].size}")
    return imgs


def save(model, view, arr):
    d = os.path.join(OUT, model)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"{view}.npy"), arr.astype(np.float16))


def run_depth_anything(imgs, device):
    from transformers import pipeline
    name = "depth-anything/Depth-Anything-V2-Large-hf"
    log(f"== {name}")
    pipe = pipeline("depth-estimation", model=name, device=device)
    for v, im in imgs.items():
        t = time.time()
        out = pipe(im)
        d = np.array(out["predicted_depth"].squeeze().float().cpu())
        if d.shape != (im.height, im.width):
            d = np.array(Image.fromarray(d).resize(im.size, Image.BILINEAR))
        save("depth_anything_v2_large", v, d)
        log(f"   {v} {d.shape} range [{d.min():.3f},{d.max():.3f}] "
            f"{time.time()-t:.1f}s")
    manifest["models"]["depth_anything_v2_large"] = {
        "kind": "depth", "hf": name,
        "note": "affine-invariant inverse depth (larger == nearer)"}
    del pipe


def run_marigold(imgs, device, kind, repo, ensemble, steps, res):
    from diffusers import MarigoldDepthPipeline, MarigoldNormalsPipeline
    import torch
    cls = MarigoldDepthPipeline if kind == "depth" else MarigoldNormalsPipeline
    log(f"== {repo}  ensemble={ensemble} steps={steps} res={res}")
    pipe = cls.from_pretrained(repo, variant="fp16", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    key = repo.split("/")[-1].replace("-", "_")
    for v, im in imgs.items():
        t = time.time()
        out = pipe(im, ensemble_size=ensemble, num_inference_steps=steps,
                   processing_resolution=res, match_input_resolution=True,
                   output_type="np")
        a = out.prediction[0] if kind == "normals" else out.prediction.squeeze()
        a = np.asarray(a, dtype=np.float32)
        save(key, v, a)
        log(f"   {v} {a.shape} range [{a.min():.3f},{a.max():.3f}] "
            f"{time.time()-t:.1f}s")
    manifest["models"][key] = {
        "kind": kind, "hf": repo, "ensemble": ensemble, "steps": steps,
        "processing_resolution": res,
        "note": ("affine-invariant depth (larger == farther)" if kind == "depth"
                 else "unit normals in camera space, OpenGL convention")}
    del pipe


def main():
    import torch
    log("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        log("gpu:", torch.cuda.get_device_name(0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)

    imgs = load_views()

    jobs = [
        ("depth_anything", lambda: run_depth_anything(imgs, 0 if device == "cuda" else -1)),
        ("marigold_depth", lambda: run_marigold(
            imgs, device, "depth", "prs-eth/marigold-depth-v1-1", 5, 4, 1024)),
        ("marigold_normals", lambda: run_marigold(
            imgs, device, "normals", "prs-eth/marigold-normals-v1-1", 5, 4, 1024)),
    ]

    for name, fn in jobs:
        try:
            fn()
            torch.cuda.empty_cache()
        except Exception:
            manifest["errors"][name] = traceback.format_exc()[-2000:]
            log(f"!! {name} FAILED")
            traceback.print_exc()

    with open("/kaggle/working/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log("models done:", list(manifest["models"]))
    log("failures   :", list(manifest["errors"]))


if __name__ == "__main__":
    os.system("pip install -q --upgrade diffusers transformers accelerate 2>&1 | tail -2")
    main()
