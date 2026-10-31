source .env.example
CUDA_VISIBLE_DEVICES='3' python scripts/run_zeroshot.py   --csv /datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main/specimen_benchmark_metadata.csv   --images-root /datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main/final_benchmark_data_crops   --eval-split test_seen,test_unseen   --prompt-col label_text   --model google/siglip2-base-patch16-224 --topk 5
