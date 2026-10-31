export PYTHONPATH="./EVA/EVA-CLIP/rei:${PYTHONPATH}

CUDA_VISIBLE_DEVICES='1' python scripts/eval_eva_clip_official.py   --csv /datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main/specimen_benchmark_metadata.csv   --images-root /datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main/final_benchmark_data_crops   --eval-split test_seen,test_unseen   --prompt-col label_text   --model-name EVA02-CLIP-L-14 --pretrained eva_clip --show-model --hierarchy-report-prefix ./dbg_eva_official/L14/zeroshot"
