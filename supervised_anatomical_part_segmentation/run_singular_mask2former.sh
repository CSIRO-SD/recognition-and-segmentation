#!/usr/bin/env bash
# Mask2Former — singular labels (11 classes). Crop root: export IMG_ROOT=...
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "${DIR}/run_seg_env.sh"
IMG_ROOT="${IMG_ROOT:-/datasets/work/icvg-phd-marsfield/work/all_crops}"
export CUDA_VISIBLE_DEVICES="3"
# Mask2Former: backbone LR scale, warmup+cosine, grad clip in train_seg.py.
# OOM: pass e.g. --m2f-train-batch-cap 4  or  lower --batch-size
python "${DIR}/train_seg.py" --model mask2former \
  --img-root "${IMG_ROOT}" --label-set singular --device cuda:0 \
  --batch-size 4 --lr 1e-4 --epochs "${SEG_EPOCHS}" "$@"
