#!/usr/bin/env bash
# sam3 label set (31 classes) — semantic U-Net example.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "${DIR}/run_seg_env.sh"
IMG_ROOT="${IMG_ROOT:-/datasets/work/icvg-phd-marsfield/work/all_crops}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python "${DIR}/train_seg.py" --model unet \
  --img-root "${IMG_ROOT}" --label-set sam3 --device cuda:0 \
  --epochs "${SEG_EPOCHS}" "$@"
