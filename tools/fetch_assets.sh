#!/usr/bin/env bash
# Fetch the renderer and the ground-truth mesh. Both land in assets/, which is
# gitignored — roughly 1.3 GB once extracted.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p assets && cd assets

BLENDER_VER=4.5.9
if [ ! -x blender/blender ]; then
  echo "==> Blender ${BLENDER_VER} LTS"
  curl -L --retry 3 -o blender.tar.xz \
    "https://download.blender.org/release/Blender4.5/blender-${BLENDER_VER}-linux-x64.tar.xz"
  tar xf blender.tar.xz && rm blender.tar.xz
  mv "blender-${BLENDER_VER}-linux-x64" blender
fi
./blender/blender --version | head -1

if [ ! -f lucy_le.ply ]; then
  echo "==> Stanford Lucy (322 MB)"
  curl -L --retry 3 -o lucy.tar.gz \
    "https://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz"
  tar xzf lucy.tar.gz && rm lucy.tar.gz
  # The distributed PLY is big-endian; Blender 4.x only reads little-endian.
  ./blender/4.5/python/bin/python3.11 ../tools/ply_be2le.py lucy.ply lucy_le.ply
  rm -f lucy.ply
fi
ls -lh lucy_le.ply
echo "done"
