#!/bin/bash
set -e

cd /kaggle/working/BundleRecFramework

mkdir -p third_party

if [ ! -d third_party/CrossCBR ]; then
  echo "[Setup] Cloning official CrossCBR..."
  git clone --depth 1 https://github.com/mysbupt/CrossCBR.git third_party/CrossCBR
else
  echo "[Setup] CrossCBR already exists."
fi

if [ ! -d third_party/BGCN ]; then
  echo "[Setup] Cloning official BGCN..."
  git clone --depth 1 https://github.com/cjx0525/BGCN.git third_party/BGCN
else
  echo "[Setup] BGCN already exists."
fi

echo "[Setup] Done."
