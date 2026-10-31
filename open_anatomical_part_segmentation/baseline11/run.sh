/datasets/work/icvg-phd-marsfield/work/Feng/conda_envs/insect/bin/python -m baseline13.eval_dinov3_dinotxt_balanced \
  --dinov3-repo /path/to/dinov3 \
  --backbone-weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinotxt-weights /path/to/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
  --bpe-path /path/to/bpe_simple_vocab_16e6.txt.gz\
  --device cuda