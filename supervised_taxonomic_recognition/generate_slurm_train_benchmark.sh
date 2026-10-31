#!/usr/bin/env bash
# 按 run_script.sh 的版式，每个 recipe 写一个独立 SLURM 脚本（只改 job-name 和 --recipe）。
#   bash generate_slurm_train_benchmark.sh
#   sbatch slurm_jobs/run_script_resnet50.sh

set -euo pipefail
GEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${GEN_DIR}/slurm_jobs"
mkdir -p "${OUT_DIR}"

RECIPES=(
  resnet50
  convnext_tiny
  vit_b16
  dinov2_vitb_linear
  dinov2_vitb_finetune
  dinov3_vitl_linear
  dinov3_vitl_finetune
)

emit_one() {
  local recipe="$1"
  local out="${OUT_DIR}/run_script_${recipe}.sh"
  cat >"${out}" <<'HDR'
#! /bin/bash
HDR
  cat >>"${out}" <<EOF
#SBATCH --account=OD-239291
#SBATCH --job-name=${recipe}
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL

cd /datasets/work/icvg-phd-marsfield/work/Junhao/neurips_code/supervised_cls

# Set HuggingFace cache to scratch directory to avoid disk quota issues
export HF_HOME="/scratch3/zha426/hf_cache"
export TRANSFORMERS_CACHE="/scratch3/zha426/hf_cache"
export HF_DATASETS_CACHE="/scratch3/zha426/hf_cache"
export XDG_CACHE_HOME="/scratch3/zha426/cache"

# Create cache directory if it doesn't exist
mkdir -p /scratch3/zha426/hf_cache

# Activate conda environment using the correct method
source /apps/miniforge3/24.9.0/etc/profile.d/conda.sh
conda activate dino

mkdir -p /datasets/work/icvg-phd-marsfield/work/Junhao/neurips_code/supervised_cls/log
echo "-----------Begin------------"

ROOT="/datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main"
python train_benchmark_cls.py \\
  --csv "\${ROOT}/specimen_benchmark_metadata.csv" \\
  --images-root "\${ROOT}/final_benchmark_data_crops" \\
  --recipe ${recipe} \\
  --device cuda:0 \\
  --workers 36

echo "----------- End ------------"
EOF
  echo "Wrote ${out}"
}

for recipe in "${RECIPES[@]}"; do
  emit_one "${recipe}"
done

echo "Done. Example: sbatch ${OUT_DIR}/run_script_resnet50.sh"
