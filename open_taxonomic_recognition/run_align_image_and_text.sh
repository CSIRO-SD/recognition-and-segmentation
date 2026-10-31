cd /datasets/work/icvg-phd-marsfield/work/Junhao/neurips_code/clip_siglip

CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_vision.py \
  --csv /datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main/specimen_benchmark_metadata.csv \
  --images-root /datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main/final_benchmark_data_crops \
  --train-split train \
  --text-col label_text \
  --model google/siglip2-base-patch16-512 \
  --output-dir /datasets/work/icvg-phd-marsfield/work/Junhao/neurips_code/clip_siglip/runs/siglip2_lora_vision \
  --batch-size 128 --lr 4e-4
