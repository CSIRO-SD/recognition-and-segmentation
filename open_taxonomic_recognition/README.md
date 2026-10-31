# Open-class / vision–language baselines (`open_cls`)

Thin wrappers around **`scripts/run_zeroshot.py`** (HF CLIP / SigLIP / MetaCLIP / DFN-CLIP), **`scripts/eval_eva_clip_official.py`** (EVA-CLIP), and **LoRA train/eval** under the sibling **`neurips_code/clip_siglip`** tree.

Default data in the shell files point to **`specimen_benchmark_metadata.csv`** + **`final_benchmark_data_crops`** — change `--csv`, `--images-root`, GPU id, and output paths for your machine.

## Setup

- **HF cache / token (optional):** e.g. `source .env.example` where used (`run_siglip2-base-patch16-224.sh`).
- **EVA-CLIP:** scripts prepend `PYTHONPATH=./EVA/EVA-CLIP/rei:...` so `eval_eva_clip_official.py` can import the EVA helper stack; run from repo root **`open_cls/`**.

## Zero-shot evaluation (`run_zeroshot.py`)

All **`run_*.sh`** below (except EVA + LoRA) call **`python scripts/run_zeroshot.py`** with:

- `--eval-split test_seen,test_unseen`
- `--prompt-col label_text`
- `--topk 5` (where present)

| Shell script | HF `--model` (or equivalent) |
|--------------|------------------------------|
| `run_siglip2-base-patch16-224.sh` | `google/siglip2-base-patch16-224` |
| `run_metaclip2-l14.sh` | `facebook/metaclip-2-worldwide-l14` |
| `run_metaclip-l14-400m.sh` | `facebook/metaclip-l14-400m` |
| `run_metaclip-l14-2.5b.sh` | `facebook/metaclip-l14-fullcc2.5b` |
| `run_metaclip-b32-400m.sh` | `facebook/metaclip-b32-400m` |
| `run_metaclip-2-b16-384.sh` | `facebook/metaclip-2-worldwide-b16-384` |
| `run_dfn_2b.sh` | `apple/DFN2B-CLIP-ViT-L-14` |
| `run_clip-vit-base-patch32.sh` | `openai/clip-vit-base-patch32` |

Some runs pass **`--show-model`** and **`--hierarchy-report-prefix`** to write order-level TSVs under `./dbg_*`.

## EVA-CLIP (`eval_eva_clip_official.py`)

| Shell script | `--model-name` |
|--------------|----------------|
| `run_evaclip_l14.sh` | `EVA02-CLIP-L-14` |
| `run_evaclip_bigE.sh` | `EVA02-CLIP-bigE-14` |
| `run_evaclip_b16.sh` | `EVA02-CLIP-B-16` |

## LoRA: align image ↔ caption (and optional metadata)

These **`cd`** into **`.../neurips_code/clip_siglip`** and call **`train_lora_vision.py`** / **`run_zeroshot.py`** there (paths in the `.sh` files are absolute — edit if your tree differs).

| Script | Role |
|--------|------|
| `run_align_image_and_text.sh` | Train vision LoRA on `(image, label_text)` only. |
| `evaluate_align_image_and_text.sh` | Zero-shot eval with **`--peft-adapter`** pointing at the LoRA output dir. |
| `run_align_image_and_text_and_metadata.sh` | Train LoRA with **`--weak-col weak_context_text`** and **`--meta-loss-weight`** (auxiliary SigLIP-style image–metadata term). |
| `evaluate_align_image_and_text_and_metadata.sh` | Eval with PEFT adapter + **`--weak-col` / `--weak-fusion-alpha`** for optional metadata fusion at test time (see `clip_siglip` `run_zeroshot.py` help). |

**Note:** Other eval paths in this repo include `scripts/eval_metaclip_hf_forward.py`, `scripts/eval_siglip_standalone.py`, and `scripts/run_metadata_fusion_eval.py` (trained fusion head) — use those scripts’ `--help` for options.

## Quick edits

1. Replace **`CUDA_VISIBLE_DEVICES`** per job.
2. Point **`--csv`** / **`--images-root`** at your benchmark.
3. For LoRA wrappers, align **`--output-dir`** and **`--peft-adapter`** with the same run.
