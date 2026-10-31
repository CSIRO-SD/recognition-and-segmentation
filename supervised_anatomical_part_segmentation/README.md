# Supervised segmentation (spatial-aware COCO masks)

Training wrappers around `train_seg.py` for **polygon / instance COCO JSON** annotations and **semantic-mask evaluation** (mIoU, Dice, pixel accuracy, per-class IoU).

## Labels

Annotations are **spatial-aware** COCO masks: each polygon / instance carries a **category id** tied to the crop coordinate system and the **`file_name`** under `--img-root` (not just global image-level tags).

| `--label-set` | Annotation files | Classes |
|---------------|------------------|---------|
| `singular` | `annotations/singular_sam3_train.json`, `annotations/singular_sam3_test.json` | **11** body-part classes (singular scheme), **spatial-aware** |
| `sam3` | `annotations/sam3_train 1.json`, `annotations/sam3_test 1.json` | **31** classes, **spatial-aware** (SAM3-style part labels in these JSONs) |

 **`--label-set sam3`** therefore uses the **31-class spatial-aware** split from the `sam3_*` COCO files above; **`--label-set singular`** uses the **11-class spatial-aware** `singular_sam3_*` files. Same eval protocol (dense semantic mIoU / Dice); only the taxonomy and JSON paths differ.

The `run_singular_*.sh` scripts use **`--label-set singular`** so filenames match the **11-class singular** setup. To train on **spatial-aware sam3** instead, pass `--label-set sam3` (e.g. `./run_singular_unet.sh --label-set sam3`).

## Image root

Images are resolved as `{IMG_ROOT}/{file_name}` from the COCO `images` entries (e.g. `images/foo.jpg`).

**Option A — environment variable (recommended):**

```bash
export IMG_ROOT=/path/to/your/crop_root
./run_singular_unet.sh
```

**Option B — default in each script:**  
Edit the line in the shell script:

```bash
IMG_ROOT="${IMG_ROOT:-/path/to/your/crop_root}"
```

If `IMG_ROOT` is unset, the scripts fall back to  
`/datasets/work/icvg-phd-marsfield/work/all_crops` — change this to your machine.

## Epochs

`run_seg_env.sh` sets `SEG_EPOCHS` (default **100**). Override:

```bash
export SEG_EPOCHS=50
./run_singular_mask2former.sh
```

## Scripts

| Script | Model |
|--------|--------|
| `run_singular_unet.sh` | U-Net (semantic) |
| `run_singular_deeplabv3plus.sh` | DeepLabV3+ (semantic) |
| `run_singular_segformer.sh` | SegFormer (semantic) |
| `run_singular_mask_rcnn.sh` | Mask R-CNN (instance → painted semantic eval) |
| `run_singular_mask2former.sh` | Mask2Former (instance → painted semantic eval) |

Extra CLI flags are forwarded (e.g. `./run_singular_unet.sh --batch-size 8`).

## Direct Python

```bash
python train_seg.py --model unet --label-set singular \
  --img-root "${IMG_ROOT}" --device cuda:0
```

See `train_seg.py --help` for all options (including Mask2Former `--m2f-*`).
