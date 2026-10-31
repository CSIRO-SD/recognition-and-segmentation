# InSpect: Insect Specimen Recognition and Anatomical Part Segmentation

[![Project page](https://img.shields.io/badge/Project-InSpect-2c7a7b)](https://csiro-sd.github.io/InSpect/)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-8c1b13)](https://openreview.net/forum?id=51J0UrBZZy)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffd21e)](https://huggingface.co/datasets/anonymous-dataset/InSpect)

InSpect is a curated natural-history collection dataset for visual insect specimen understanding. It contains **48,184** digitized specimen images spanning **653 taxa**, with aligned insect crops, hierarchical taxonomy, label-derived structured metadata, and **1,859** images with fine-grained anatomical part masks.

The benchmark separates two capabilities:

1. **Taxonomic recognition** from cropped specimen images, with closed-set supervised, zero-shot, fine-tuned, and unseen-taxon evaluation.
2. **Fine-grained anatomical part segmentation** with supervised and text-guided open-vocabulary models under singular and spatial-aware label schemes.

![Overview of InSpect and its two benchmarks](https://raw.githubusercontent.com/CSIRO-SD/InSpect/main/assets/paper/inspect-overview-figure2.png)

## Links

- [Project page](https://csiro-sd.github.io/InSpect/)
- [Paper / OpenReview](https://openreview.net/forum?id=51J0UrBZZy)
- [Dataset on Hugging Face](https://huggingface.co/datasets/anonymous-dataset/InSpect)

## Repository structure

```text
.
|-- open_anatomical_part_segmentation/
|-- open_taxonomic_recognition/
|-- supervised_anatomical_part_segmentation/
|-- supervised_taxonomic_recognition/
`-- README.md
```

| Directory | Benchmark | Main entry points |
|---|---|---|
| [`supervised_taxonomic_recognition/`](https://github.com/CSIRO-SD/recognition-and-segmentation/tree/281643016e81057c9bbb958b1bc74216d352e2a8/supervised_taxonomic_recognition) | Closed-set taxonomic recognition | [`train_benchmark_cls.py`](https://github.com/CSIRO-SD/recognition-and-segmentation/blob/281643016e81057c9bbb958b1bc74216d352e2a8/supervised_taxonomic_recognition/train_benchmark_cls.py) and the `run_benchmark_*.sh` recipes |
| [`open_taxonomic_recognition/`](https://github.com/CSIRO-SD/recognition-and-segmentation/tree/281643016e81057c9bbb958b1bc74216d352e2a8/open_taxonomic_recognition) | Zero-shot and LoRA-adapted open taxonomic recognition | [`scripts/run_zeroshot.py`](https://github.com/CSIRO-SD/recognition-and-segmentation/blob/281643016e81057c9bbb958b1bc74216d352e2a8/open_taxonomic_recognition/scripts/run_zeroshot.py), [`scripts/train_lora_vision.py`](https://github.com/CSIRO-SD/recognition-and-segmentation/blob/281643016e81057c9bbb958b1bc74216d352e2a8/open_taxonomic_recognition/scripts/train_lora_vision.py) |
| [`supervised_anatomical_part_segmentation/`](https://github.com/CSIRO-SD/recognition-and-segmentation/tree/281643016e81057c9bbb958b1bc74216d352e2a8/supervised_anatomical_part_segmentation) | Closed-set anatomical part segmentation | [`train_seg.py`](https://github.com/CSIRO-SD/recognition-and-segmentation/blob/281643016e81057c9bbb958b1bc74216d352e2a8/supervised_anatomical_part_segmentation/train_seg.py) and the `run_singular_*.sh` recipes |
| [`open_anatomical_part_segmentation/`](https://github.com/CSIRO-SD/recognition-and-segmentation/tree/281643016e81057c9bbb958b1bc74216d352e2a8/open_anatomical_part_segmentation) | Text-guided open-vocabulary anatomical segmentation | Twelve baseline folders covering CLIPSeg, Grounding DINO + SAM/SAM2, DINOv3, and SAM3 |

## Data

The [Hugging Face dataset repository](https://huggingface.co/datasets/anonymous-dataset/InSpect) provides:

- `specimen_benchmark_metadata.csv` and `specimen_benchmark_metadata.jsonl`
- `final_benchmark_data_crops.zip`
- `segmentation_images.zip`
- `segmentation_annotations.zip`
- `original_images.zip`

Recognition experiments use cropped insect images, not the original full collection images. This removes the physical specimen label from the visual input. For metadata experiments, direct taxonomic leakage fields - `order`, `family`, `genus`, `species`, `authority`, and `year` - are removed before the remaining fields are serialized as weak specimen context.

### Recognition splits

| Split value in the released CSV | Images | Taxa | Genera | Families | Orders |
|---|---:|---:|---:|---:|---:|
| `train` | 27,082 | 517 | 117 | 13 | 5 |
| `val` | 3,867 | 517 | 117 | 13 | 5 |
| `test_seen` | 7,759 | 517 | 117 | 13 | 5 |
| `test_unseen` | 9,476 | 136 | 66 | 10 | 6 |

`test_seen` contains taxa observed during training. `test_unseen` contains 136 held-out genus-species taxon keys and has no taxon-key overlap with the training, validation, or test-seen splits.

### Anatomical annotations

The anatomical subset contains 1,859 images covering 6 orders, 18 order-family combinations, 126 genera, and 632 genus-species taxa. The released COCO-style annotations provide:

- **Singular labels:** 11 anatomical classes; left/right and positional variants are merged.
- **Spatial-aware labels:** 31 classes; side and position are retained, for example `left_antenna`, `right_hindleg`, and `left_forewing`.

After extracting the annotation archive, the relevant files are:

```text
annotations/
|-- singular_sam3_train.json
|-- singular_sam3_val.json
|-- singular_sam3_test.json
|-- sam3_train 1.json
|-- sam3_val 1.json
`-- sam3_test 1.json
```

## Benchmark protocol and metrics

### Recognition

- **Top-1:** the predicted genus-species taxon key exactly matches the ground truth.
- **Top-5:** the ground-truth taxon appears among the five highest-scoring predictions.
- **Order / family / genus / species accuracy:** predictions are mapped to each level of the taxonomic hierarchy before comparison.
- Zero-shot models rank candidate taxon names by image-text similarity. The paper uses the prompt templates `{taxon}`, `a museum specimen described as: {taxon}`, and `a pinned insect specimen. {taxon}`; embeddings are averaged when multiple templates are used.
- The fine-tuned SigLIP 2 experiments use `google/siglip2-base-patch16-512`, update LoRA adapters in the image encoder, and keep the text encoder fixed. The image-only setting uses AdamW, learning rate `4e-4`, weight decay `0.01`, batch size `128`, and `212` steps. The image-plus-metadata run uses the same setup with metadata loss weight `0.25`.

### Segmentation

- **mIoU:** mean class-wise intersection over union across anatomical parts.
- **Dice/F1:** `2 |P intersect G| / (|P| + |G|)` for predicted pixels `P` and ground-truth pixels `G`.
- **Pixel accuracy:** the proportion of correctly labeled pixels.
- **Mask AP / AR:** COCO-style mask average precision and recall, averaged over IoU thresholds `0.50:0.05:0.95`.
- Subscripts `L`, `M`, and `S` denote large, medium, and small regions, respectively.

All result values below are percentages and are transcribed from Tables 3-6 of the paper.

## Results

### 1. Supervised taxonomic recognition

Closed-set classification is trained on `train` and evaluated on `test_seen` over the fixed set of 517 seen taxa.

| Method | Backbone | Strategy | Top-1 | Top-5 |
|---|---|---|---:|---:|
| ResNet50 | ResNet50 | Fine-tune | 90.01 | 98.49 |
| ConvNeXt | convnext_tiny | Fine-tune | 91.29 | 98.52 |
| ViT | B/16 | Fine-tune | 90.42 | 98.47 |
| DINOv2 | B/14 | Linear probe | 84.75 | 96.75 |
| DINOv2 | B/14 | Fine-tune | 92.00 | 98.94 |
| DINOv3 | L/16 | Linear probe | 82.73 | 96.04 |
| DINOv3 | L/16 | Fine-tune | 93.50 | 99.24 |

### 2. Open taxonomic recognition

| Split | Model | ViT | Setting | Resolution | Order | Family | Genus | Species Top-1 | Species Top-5 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Test-seen | CLIP | L/14 | Zero-shot | 224 | 94.92 | 55.50 | 10.57 | 2.04 | 6.51 |
| Test-seen | EVA-CLIP | H/14 | Zero-shot | 224 | 88.97 | 42.57 | 8.98 | 2.67 | 6.91 |
| Test-seen | DFN | L/14 | Zero-shot | 224 | 95.58 | 55.15 | 12.11 | 2.95 | 7.84 |
| Test-seen | MetaCLIP 1 | B/32 | Zero-shot | 224 | 46.18 | 30.18 | 5.27 | 0.40 | 2.63 |
| Test-seen | MetaCLIP 2 | B/16 | Zero-shot | 384 | 91.40 | 59.85 | 13.03 | 3.38 | 7.42 |
| Test-seen | SigLIP | B/16 | Zero-shot | 224 | 90.48 | 57.60 | 9.51 | 2.55 | 8.15 |
| Test-seen | BioCLIP | B/16 | Zero-shot | 224 | 91.70 | 65.99 | 22.40 | 4.03 | 11.61 |
| Test-seen | BioCLIP 2 | L/14 | Zero-shot | 224 | 99.27 | 74.46 | 31.64 | 5.13 | 18.93 |
| Test-seen | SigLIP 2 | B/16 | Zero-shot | 512 | 94.50 | 47.36 | 16.97 | 1.89 | 8.04 |
| Test-seen | SigLIP 2 | B/16 | Image-only fine-tune | 512 | 89.44 | 50.33 | 21.30 | 3.12 | 8.27 |
| Test-seen | SigLIP 2 | B/16 | Image + metadata fine-tune | 512 | 89.43 | 44.70 | 16.77 | 4.21 | 9.25 |
| Test-unseen | CLIP | L/14 | Zero-shot | 224 | 94.66 | 62.77 | 9.24 | 3.51 | 16.53 |
| Test-unseen | EVA-CLIP | H/14 | Zero-shot | 224 | 86.83 | 56.07 | 12.68 | 2.56 | 15.38 |
| Test-unseen | DFN | L/14 | Zero-shot | 224 | 96.96 | 62.73 | 22.28 | 4.05 | 23.32 |
| Test-unseen | MetaCLIP 1 | B/32 | Zero-shot | 224 | 46.71 | 34.04 | 9.97 | 4.91 | 13.34 |
| Test-unseen | MetaCLIP 2 | B/16 | Zero-shot | 384 | 90.50 | 52.76 | 21.95 | 5.24 | 26.93 |
| Test-unseen | SigLIP | B/16 | Zero-shot | 224 | 91.73 | 64.87 | 17.54 | 5.07 | 20.23 |
| Test-unseen | BioCLIP | B/16 | Zero-shot | 224 | 95.98 | 76.67 | 35.86 | 11.77 | 38.22 |
| Test-unseen | BioCLIP 2 | L/14 | Zero-shot | 224 | 99.66 | 80.94 | 45.82 | 11.37 | 37.28 |
| Test-unseen | SigLIP 2 | B/16 | Zero-shot | 512 | 95.97 | 66.09 | 25.05 | 4.93 | 14.86 |
| Test-unseen | SigLIP 2 | B/16 | Image-only fine-tune | 512 | 94.72 | 65.92 | 29.40 | 6.16 | 21.21 |
| Test-unseen | SigLIP 2 | B/16 | Image + metadata fine-tune | 512 | 95.64 | 68.87 | 28.50 | 11.07 | 32.32 |

The seen and unseen rows have different taxonomic compositions and should not be read as a difficulty-controlled pair. For zero-shot models, neither split is used for task-specific training; both evaluate image-text alignment over candidate taxon names.

### 3. Supervised anatomical part segmentation

| Model | Singular mIoU | Singular Dice/F1 | Singular pixel acc. | Spatial-aware mIoU | Spatial-aware Dice/F1 | Spatial-aware pixel acc. |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 43.53 | 53.88 | 94.05 | 29.86 | 40.61 | 90.09 |
| DeepLabV3+ | 49.56 | 60.04 | 95.29 | 35.64 | 46.56 | 91.93 |
| SegFormer | 48.06 | 57.72 | 95.25 | 32.78 | 42.96 | 91.63 |
| Mask R-CNN | 50.00 | 62.08 | 94.80 | 33.68 | 44.21 | 91.48 |
| Mask2Former | 50.25 | 61.50 | 95.19 | 36.36 | 47.35 | 91.72 |

### 4. Text-guided open-vocabulary anatomical part segmentation

| Label scheme | Model | Setting | mIoU | Dice/F1 | AP | AP-L | AP-M | AP-S | AR | AR-L | AR-M | AR-S |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singular | CLIPSeg | Zero-shot | 6.31 | 9.06 | 0.34 | 0.44 | 0.46 | 0.01 | 0.74 | 0.83 | 0.46 | 0.01 |
| Singular | Grounding DINO + SAM | Zero-shot | 13.81 | 15.20 | 6.61 | 7.80 | 1.83 | 0.31 | 7.58 | 9.06 | 1.74 | 0.21 |
| Singular | Grounding DINO + SAM2 | Zero-shot | 13.73 | 15.12 | 6.51 | 7.66 | 1.88 | 0.34 | 7.50 | 8.94 | 1.78 | 0.23 |
| Singular | DINOv3 | Zero-shot | 3.04 | 5.07 | 0.77 | 0.21 | 0.00 | 0.00 | 0.39 | 0.39 | 0.00 | 0.00 |
| Singular | SAM3 | Zero-shot | 44.28 | 47.03 | 21.43 | 23.59 | 18.49 | 2.55 | 23.02 | 25.47 | 19.31 | 2.58 |
| Singular | SAM3 | Fine-tuned | 89.18 | 84.12 | 83.41 | 84.87 | 74.94 | 59.15 | 82.14 | 78.92 | 71.61 | 52.27 |
| Spatial-aware | CLIPSeg | Zero-shot | 6.30 | 9.04 | 0.17 | 0.21 | 0.32 | 0.00 | 0.46 | 0.58 | 0.34 | 0.00 |
| Spatial-aware | Grounding DINO + SAM | Zero-shot | 3.72 | 4.75 | 0.70 | 0.85 | 0.39 | 0.00 | 1.04 | 1.25 | 0.36 | 0.00 |
| Spatial-aware | Grounding DINO + SAM2 | Zero-shot | 3.75 | 4.69 | 0.73 | 0.89 | 0.45 | 0.00 | 1.03 | 1.35 | 0.40 | 0.00 |
| Spatial-aware | DINOv3 | Zero-shot | 3.04 | 4.77 | 0.16 | 0.19 | 0.00 | 0.00 | 0.32 | 0.37 | 0.00 | 0.00 |
| Spatial-aware | SAM3 | Zero-shot | 14.17 | 14.97 | 7.68 | 8.95 | 7.01 | 0.57 | 13.27 | 17.03 | 8.20 | 0.77 |
| Spatial-aware | SAM3 | Fine-tuned | 65.16 | 62.07 | 54.65 | 56.30 | 30.93 | 15.33 | 44.59 | 54.69 | 29.38 | 15.16 |

## Running the released code

The commands below use the entry points that exist in the linked code snapshot. Set the paths for your machine first:

```bash
export CODE_ROOT=/path/to/recognition-and-segmentation
export DATA_ROOT=/path/to/InSpect
export CSV="$DATA_ROOT/specimen_benchmark_metadata.csv"
export CROP_ROOT="$DATA_ROOT/final_benchmark_data_crops"
export SEG_ROOT=/path/to/extracted/segmentation_data
```

`SEG_ROOT` should contain `annotations/` and the image paths referenced by the COCO `file_name` fields. The released shell wrappers contain author-local paths and, in some cases, fixed GPU IDs; either edit those values or call the Python entry points directly as below.

### Supervised taxonomic recognition

```bash
cd "$CODE_ROOT/supervised_taxonomic_recognition"

python train_benchmark_cls.py \
  --csv "$CSV" \
  --images-root "$CROP_ROOT" \
  --recipe dinov3_vitl_finetune \
  --device cuda:0
```

Available recipes are `resnet50`, `convnext_tiny`, `vit_b16`, `dinov2_vitb_linear`, `dinov2_vitb_finetune`, `dinov3_vitl_linear`, and `dinov3_vitl_finetune`. See the [recipe registry and wrappers](https://github.com/CSIRO-SD/recognition-and-segmentation/tree/281643016e81057c9bbb958b1bc74216d352e2a8/supervised_taxonomic_recognition).

### Open taxonomic recognition

```bash
cd "$CODE_ROOT/open_taxonomic_recognition"
python -m pip install -r requirements.txt

python scripts/run_zeroshot.py \
  --csv "$CSV" \
  --images-root "$CROP_ROOT" \
  --eval-split test_seen,test_unseen \
  --prompt-col label_text \
  --model google/siglip2-base-patch16-512 \
  --topk 5
```

Image-only LoRA adaptation:

```bash
python scripts/train_lora_vision.py \
  --csv "$CSV" \
  --images-root "$CROP_ROOT" \
  --train-split train \
  --text-col label_text \
  --model google/siglip2-base-patch16-512 \
  --output-dir runs/siglip2_lora_vision \
  --batch-size 128 \
  --lr 4e-4

python scripts/run_zeroshot.py \
  --csv "$CSV" \
  --images-root "$CROP_ROOT" \
  --eval-split test_seen,test_unseen \
  --prompt-col label_text \
  --model google/siglip2-base-patch16-512 \
  --peft-adapter runs/siglip2_lora_vision \
  --topk 5
```

For the image-plus-metadata variant, add `--weak-col weak_context_text --meta-loss-weight 0.25` during training. The released training and evaluation wrappers are [here](https://github.com/CSIRO-SD/recognition-and-segmentation/tree/281643016e81057c9bbb958b1bc74216d352e2a8/open_taxonomic_recognition).

### Supervised anatomical part segmentation

```bash
cd "$CODE_ROOT/supervised_anatomical_part_segmentation"

python -m pip install -r requirements.txt

python train_seg.py \
  --model mask2former \
  --label-set singular \
  --img-root "$SEG_ROOT" \
  --train-json "$SEG_ROOT/annotations/singular_sam3_train.json" \
  --test-json "$SEG_ROOT/annotations/singular_sam3_test.json" \
  --device cuda:0 \
  --epochs 100
```

Models supported by `--model` are `unet`, `deeplabv3plus`, `segformer`, `mask_rcnn`, and `mask2former`. To use spatial-aware labels, pass `--label-set sam3` with `sam3_train 1.json` and `sam3_test 1.json`. The released wrappers default to 100 epochs through `SEG_EPOCHS`; additional arguments are forwarded to `train_seg.py`.

### Text-guided open-vocabulary anatomical segmentation

Install the model-specific dependencies listed in the [open segmentation README](https://github.com/CSIRO-SD/recognition-and-segmentation/blob/281643016e81057c9bbb958b1bc74216d352e2a8/open_anatomical_part_segmentation/README.md). For a SAM3 zero-shot evaluation on singular labels:

```bash
cd "$CODE_ROOT/open_anatomical_part_segmentation"

python baseline2/eval_pretrained_sam3_zeroshot.py \
  --gt-json "$SEG_ROOT/annotations/singular_sam3_test.json" \
  --dataset-root "$SEG_ROOT" \
  --device cuda
```

For the spatial-aware setting, use the corresponding entry point and annotation file:

```bash
python baseline1/eval_pretrained_sam3_zeroshot.py \
  --gt-json "$SEG_ROOT/annotations/sam3_test 1.json" \
  --dataset-root "$SEG_ROOT" \
  --device cuda
```

The twelve released baseline folders map to:

| Folders | Model | Setting |
|---|---|---|
| `baseline1`, `baseline2` | SAM3 | Zero-shot, spatial-aware / singular |
| `baseline3`, `baseline4` | SAM3 | Fine-tuned, singular / spatial-aware |
| `baseline5`, `baseline7` | Grounding DINO + SAM | Zero-shot, singular / spatial-aware |
| `baseline6`, `baseline8` | Grounding DINO + SAM2 | Zero-shot, singular / spatial-aware |
| `baseline9`, `baseline10` | CLIPSeg | Zero-shot, singular / spatial-aware |
| `baseline11`, `baseline12` | DINOv3 | Zero-shot, spatial-aware / singular |

Grounding DINO, SAM/SAM2, SAM3, and DINOv3 baselines require their upstream packages and checkpoints. Follow the per-baseline scripts and pass local checkpoint/configuration paths rather than relying on the author-local defaults.

## Citation

The project currently supplies the following provisional BibTeX metadata:

```bibtex
@misc{zhang2026inspect,
  title = {InSpect: A Curated Natural History Collection Dataset
           for Insect Specimen Understanding},
  author = {Zhang, Junhao and Xu, Feng and Ravendran, Ahalya
            and Fisher, Nicole and Tidd, Brendan
            and Petersson, Lars and Wang, Dadong and Li, Xun},
  year = {2026},
  note = {NeurIPS 2026; provisional metadata},
  url = {https://openreview.net/forum?id=51J0UrBZZy}
}
```

## License

InSpect is released under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license for non-commercial research and educational use.

