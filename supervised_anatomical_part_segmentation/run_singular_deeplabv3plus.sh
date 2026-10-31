#!/usr/bin/env bash
# DeepLabV3+ — singular labels (11 classes). Crop root: export IMG_ROOT=...
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "${DIR}/run_seg_env.sh"
IMG_ROOT="${IMG_ROOT:-/datasets/work/icvg-phd-marsfield/work/all_crops}"
export CUDA_VISIBLE_DEVICES="1"
python "${DIR}/train_seg.py" --model deeplabv3plus \
  --img-root "${IMG_ROOT}" --label-set singular --device cuda:0 \
  --epochs "${SEG_EPOCHS}" "$@"
