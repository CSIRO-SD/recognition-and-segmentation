## Environment setup

- Install the packages required by https://github.com/idea-research/groundingdino
- Install the packages required by https://github.com/facebookresearch/segment-anything
- Install the packages required by https://github.com/facebookresearch/sam2
- Install the packages required by https://github.com/facebookresearch/sam3/
- Install the packages required by https://github.com/facebookresearch/dinov3
- Install the packages required by https://github.com/timojl/clipseg

After installation, check your environment using `requirements.txt` and install any missing packages.

## About this code

- baseline1 — SAM3: zero-shot, spatial-aware
- baseline2 — SAM3: zero-shot, singular
- baseline3 — SAM3: fine-tune, singular
- baseline4 — SAM3: fine-tune, spatial-aware
- baseline5 — Grounding DINO + SAM: zero-shot, singular
- baseline6 — Grounding DINO + SAM2: zero-shot, singular
- baseline7 — Grounding DINO + SAM: zero-shot, spatial-aware
- baseline8 — Grounding DINO + SAM2: zero-shot, spatial-aware
- baseline9 — ClipSeg: zero-shot, singular
- baseline10 — ClipSeg: zero-shot, spatial-aware
- baseline11 — DINOv3: zero-shot, spatial-aware
- baseline12 — DINOv3: zero-shot, singular

Please read the paper for details.


## Pre-trained weights
Download weights for SAM, SAM2, SAM3, GroundingDINO, ClipSeg, and DINOv3.

## Fine-tuned weights
The `checkpoints` directory contains fine-tuned SAM3 weights; use these for SAM3 inference.
