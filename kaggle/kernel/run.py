"""M2b: multi-view conditioned estimation.

M2 measured monocular estimators view by view and found the top and bottom
views worse than guessing. That result is about monocular estimation, not
about those views: a model that sees one top-down orthographic image has no
context for it, while a multi-view model interprets that same image alongside
the side views.

This runs a strong monocular model as the control and two multi-view models as
the treatment, so "multi-view helps" is separated from "a better model helps".

VGGT additionally emits a joint world point map -- the integration itself,
done by the network -- which is saved for direct scoring against the truth.

Outputs to /kaggle/working/est/<model>/<view>.npy plus manifest.json.
"""
# Installs run before numpy is imported, and only for the jobs selected.
#
# Installing a package that pulls a different numpy after numpy is already
# loaded leaves the interpreter holding the old module against new headers,
# and the next compiled extension to import -- scipy here -- dies on an ABI
# mismatch that has nothing to do with the model. depth-anything-3 also does
# not declare addict among its dependencies.
import os as _os

_PKGS = {
    "moge2": "git+https://github.com/microsoft/MoGe.git",
    "vggt": "git+https://github.com/facebookresearch/vggt.git",
    "da3": "depth-anything-3 addict",
}
_ONLY = _os.environ.get("ONLY", "da3").split(",")
for _job in _ONLY:
    if _job in _PKGS:
        print(f"[install] {_job}: {_PKGS[_job]}", flush=True)
        _os.system(f"pip install -q {_PKGS[_job]} 2>&1 | tail -2")

import json
import os
import time
import traceback

import numpy as np
from PIL import Image

OUT = "/kaggle/working/est"
VIEWS = ["01_front", "02_right", "03_back", "04_left", "05_top", "06_bottom"]
manifest = {"models": {}, "views": VIEWS, "errors": {}, "notes": {}}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_input_root():
    root = "/kaggle/input"
    if not os.path.isdir(root):
        raise FileNotFoundError("/kaggle/input does not exist; nothing attached")
    seen = []
    for cur, dirs, files in os.walk(root):
        if cur[len(root):].count(os.sep) > 4:
            dirs[:] = []
            continue
        names = set(dirs) | set(files)
        seen.append(os.path.relpath(cur, root))
        if {"rgb", "rgb.zip"} & names:
            log(f"input root: {cur}")
            return cur
    raise FileNotFoundError("no rgb/ or rgb.zip under /kaggle/input; walked: "
                            + ", ".join(seen[:40]))


def resolve_input():
    import zipfile
    src_root = find_input_root()
    base = "/kaggle/working/views"
    os.makedirs(base, exist_ok=True)
    for name in ("rgb", "mask"):
        d = os.path.join(src_root, name)
        if os.path.isdir(d):
            dst = os.path.join(base, name)
            if not os.path.exists(dst):
                os.symlink(d, dst)
            continue
        z = os.path.join(src_root, name + ".zip")
        if os.path.isfile(z):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(os.path.join(base, name))
            continue
        raise FileNotFoundError(f"no {name} under {src_root}")
    return base


def save(model, view, arr):
    d = os.path.join(OUT, model)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"{view}.npy"), np.asarray(arr, dtype=np.float16))


def to_full(a, size):
    """Resample a model-resolution map back to the reference resolution."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        if a.shape == (size, size):
            return a
        return np.asarray(Image.fromarray(a).resize((size, size),
                                                    Image.BILINEAR))
    ch = [np.asarray(Image.fromarray(a[..., i]).resize((size, size),
                                                       Image.BILINEAR))
          for i in range(a.shape[-1])]
    return np.stack(ch, -1)


# --------------------------------------------------------------- monocular
def run_moge2(paths, size, device):
    import torch
    from moge.model.v2 import MoGeModel
    repo = "Ruicheng/moge-2-vitl-normal"
    log(f"== {repo} (monocular control)")
    model = MoGeModel.from_pretrained(repo).to(device).eval()
    got_normal = False
    for v, p in paths.items():
        t = time.time()
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        x = torch.tensor(im).permute(2, 0, 1).to(device)
        with torch.no_grad():
            out = model.infer(x)
        d = out["depth"].float().cpu().numpy()
        save("moge2_vitl_depth", v, to_full(d, size))
        if "normal" in out and out["normal"] is not None:
            n = out["normal"].float().cpu().numpy()
            if n.shape[0] == 3:
                n = np.transpose(n, (1, 2, 0))
            save("moge2_vitl_normal", v, to_full(n, size))
            got_normal = True
        log(f"   {v} depth {d.shape} {time.time()-t:.1f}s keys={list(out)}")
    manifest["models"]["moge2_vitl_depth"] = {
        "kind": "depth", "hf": repo, "conditioning": "monocular",
        "note": "metric-ish depth (larger == farther)"}
    if got_normal:
        manifest["models"]["moge2_vitl_normal"] = {
            "kind": "normals", "hf": repo, "conditioning": "monocular",
            "note": "camera-space normals"}
    del model


# -------------------------------------------------------------- multi-view
def run_vggt(paths, size, device):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    repo = "facebook/VGGT-1B"
    log(f"== {repo} (all six views in one forward pass)")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 \
        else torch.float16
    model = VGGT.from_pretrained(repo).to(device).eval()
    imgs = load_and_preprocess_images([paths[v] for v in VIEWS]).to(device)
    log(f"   batch {tuple(imgs.shape)}")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        pred = model(imgs[None])
    log(f"   outputs: {list(pred)}")
    depth = pred["depth"][0].float().cpu().numpy()          # (S,H,W,1)
    conf = pred.get("depth_conf")
    for i, v in enumerate(VIEWS):
        save("vggt_1b_depth", v, to_full(depth[i].squeeze(), size))
        if conf is not None:
            save("vggt_1b_conf", v, to_full(conf[0][i].float().cpu().numpy(), size))
    if "world_points" in pred:
        wp = pred["world_points"][0].float().cpu().numpy()
        np.save("/kaggle/working/vggt_world_points.npy", wp.astype(np.float16))
        if "world_points_conf" in pred:
            np.save("/kaggle/working/vggt_world_points_conf.npy",
                    pred["world_points_conf"][0].float().cpu().numpy()
                    .astype(np.float16))
        log(f"   world points {wp.shape} saved")
        manifest["notes"]["vggt_world_points"] = list(wp.shape)
    manifest["models"]["vggt_1b_depth"] = {
        "kind": "depth", "hf": repo, "conditioning": "multi-view (6 images)",
        "note": "per-view depth in its own camera frame (larger == farther)"}
    del model


def run_da3(paths, size, device, repo="depth-anything/DA3-LARGE"):
    import torch
    import depth_anything_3 as da3
    log(f"== {repo} (multi-view)  package exposes: "
        f"{[a for a in dir(da3) if not a.startswith('_')][:15]}")
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(repo).to(device).eval()
    imgs = [np.asarray(Image.open(paths[v]).convert("RGB")) for v in VIEWS]
    with torch.no_grad():
        pred = model.inference(imgs)
    log(f"   prediction fields: "
        f"{[a for a in dir(pred) if not a.startswith('_')][:20]}")
    depth = np.asarray(getattr(pred, "depth"))
    for i, v in enumerate(VIEWS):
        save("da3_large_depth", v, to_full(np.squeeze(depth[i]), size))
    manifest["models"]["da3_large_depth"] = {
        "kind": "depth", "hf": repo, "conditioning": "multi-view (6 images)",
        "note": "larger == farther"}
    del model


def main():
    import torch
    log("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        log("gpu:", torch.cuda.get_device_name(0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)

    base = resolve_input()
    paths = {v: os.path.join(base, "rgb", f"{v}.png") for v in VIEWS}
    size = Image.open(paths[VIEWS[0]]).size[0]
    log(f"reference resolution {size}")

    only = os.environ.get("ONLY", "").split(",") if os.environ.get("ONLY") else None
    jobs = (("moge2", lambda: run_moge2(paths, size, device)),
            ("vggt", lambda: run_vggt(paths, size, device)),
            ("da3", lambda: run_da3(paths, size, device)))
    for name, fn in jobs:
        if only and name not in only:
            log(f"-- skipping {name}")
            continue
        try:
            fn()
            torch.cuda.empty_cache()
        except Exception:
            manifest["errors"][name] = traceback.format_exc()[-3000:]
            log(f"!! {name} FAILED")
            traceback.print_exc()

    with open("/kaggle/working/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log("done:", list(manifest["models"]), "| failed:", list(manifest["errors"]))


if __name__ == "__main__":
    os.environ.setdefault("ONLY", "da3")
    main()
