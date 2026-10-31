#! /bin/bash
#SBATCH --account=OD-239291
#SBATCH --job-name=vit_b16
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
python train_benchmark_cls.py \
  --csv "${ROOT}/specimen_benchmark_metadata.csv" \
  --images-root "${ROOT}/final_benchmark_data_crops" \
  --recipe vit_b16 \
  --device cuda:0 \
  --workers 36

echo "----------- End ------------"
