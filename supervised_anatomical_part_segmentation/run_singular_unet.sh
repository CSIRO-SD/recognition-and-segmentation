#!/usr/bin/env bash
# Semantic U-Net — singular label set (11 body parts; spatial-aware COCO JSON).
# Override crop root: export IMG_ROOT=/your/path  (or edit default below).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${DIR}/run_seg_env.sh"
IMG_ROOT="${IMG_ROOT:-/datasets/work/icvg-phd-marsfield/work/all_crops}"
export CUDA_VISIBLE_DEVICES="0"
python "${DIR}/train_seg.py" \
  --model unet \
  --img-root "${IMG_ROOT}" \
  --label-set singular \
  --device cuda:0 \
  --epochs "${SEG_EPOCHS}" \
  "$@"
