# 3D reconstruction — stage 2

Images in, a 3D mesh out: orthographic views with silhouettes and per-view
normal maps are turned into one closed mesh. No learned model.

Input (`data/<name>/`):

- `views/cameras.json` — orthographic cameras (`ortho_scale`, per-view `matrix_world`)
- `views/mask/<view>.png`, `views/rgb/<view>.png`
- `normals/<view>.npy` — camera-space unit normals (x right, y up, z toward the camera; zero where unknown)

Steps (`stage2.py`): visual hull (`tools/hull_field.py`) → per-view depth from
normals placed by cross-view normal matching (`tools/depth_mv.py`) → robust
fusion into one signed distance field, outer shell only (`tools/fuse_field.py`)
→ optional consensus passes → quad retopology (`tools/retopo.py`, Blender) →
optional scoring against a ground-truth mesh.

Run:

```
python3 stage2.py --name bunny --gpu        # CUDA machine
python3 stage2.py --name bunny              # CPU
python3 tools/kaggle_stage2.py push bunny --gpu    # on a Kaggle GPU
python3 tools/kaggle_stage2.py pull bunny --gpu
```

On Lucy (18 views, 12-light photometric normals) the fused mesh scores F@1 97.7.
