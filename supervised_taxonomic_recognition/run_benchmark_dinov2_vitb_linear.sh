#!/usr/bin/env bash
# DINOv2 ViT-B: timm uses patch14 (official). Frozen backbone + linear head.
set -euo pipefail
source "$(dirname "$0")/run_benchmark_env.sh"
ROOT="/datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python "$(dirname "$0")/train_benchmark_cls.py" \
  --csv "${ROOT}/specimen_benchmark_metadata.csv" \
  --images-root "${ROOT}/final_benchmark_data_crops" \
  --recipe dinov2_vitb_linear \
  --device cuda:0 \
  "$@"
