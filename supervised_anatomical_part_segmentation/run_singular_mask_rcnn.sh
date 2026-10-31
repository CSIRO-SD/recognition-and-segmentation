#!/usr/bin/env bash
# Mask R-CNN — singular labels (11 classes). Crop root: export IMG_ROOT=...
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "${DIR}/run_seg_env.sh"
IMG_ROOT="${IMG_ROOT:-/datasets/work/icvg-phd-marsfield/work/all_crops}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python "${DIR}/train_seg.py" --model mask_rcnn \
  --img-root "${IMG_ROOT}" --label-set singular --device cuda:0 \
  --batch-size 4 --epochs "${SEG_EPOCHS}" "$@"
