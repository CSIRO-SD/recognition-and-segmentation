#!/usr/bin/env bash
# ViT-B/16 224 (timm vit_base_patch16_224), full fine-tune.
set -euo pipefail
source "$(dirname "$0")/run_benchmark_env.sh"
ROOT="/datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python "$(dirname "$0")/train_benchmark_cls.py" \
  --csv "${ROOT}/specimen_benchmark_metadata.csv" \
  --images-root "${ROOT}/final_benchmark_data_crops" \
  --recipe vit_b16 \
  --device cuda:0 \
  "$@"
