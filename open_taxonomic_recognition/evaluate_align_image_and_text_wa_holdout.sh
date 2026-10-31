CUDA_VISIBLE_DEVICES=0 python scripts/run_zeroshot.py   \
            --csv /datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main/specimen_benchmark_metadata_wa_holdout.csv  \
            --images-root /datasets/work/icvg-phd-marsfield/work/Junhao/ocr-insect-main/final_benchmark_data_crops   \
            --eval-split test_seen,test_unseen_shared_geo,test_unseen_wa_only   \
            --prompt-col label_text   \
            --model google/siglip2-base-patch16-512   \
            --peft-adapter /datasets/work/icvg-phd-marsfield/work/Junhao/neurips_code/clip_siglip/runs/wa_holdout_siglip2_lora_vision/   \
            --topk 5