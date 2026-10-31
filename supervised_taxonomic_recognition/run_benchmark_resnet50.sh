#!/usr/bin/env bash
# ResNet-50 supervised on specimen benchmark (label_text, train → test_seen).
set -euo pipefail
source "$(dirname "$0")/run_benchmark_env.sh"
ROOT="/datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main"
export CUDA_VISIBLE_DEVICES="1"
python "$(dirname "$0")/train_benchmark_cls.py" \
  --csv "${ROOT}/specimen_benchmark_metadata.csv" \
  --images-root "${ROOT}/final_benchmark_data_crops" \
  --recipe resnet50 \
  --device cuda:0 \
  "$@"
