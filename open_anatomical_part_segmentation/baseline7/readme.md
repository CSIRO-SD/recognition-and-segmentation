# Baseline 9: Spatial Aware : Grounding DINO + SAM (ViT‑H)

Open-vocabulary **boxes from Grounding DINO** using the same per-image prompts as SAM3 singular baselines (COCO category `name`s by default). Each box is refined with **SAM** (`segment_anything`, ViT‑H) via a box prompt. Predictions are written as COCO-style segmentation JSON; metrics follow baseline 4 (**mask AP/AR** as `coco_eval_segm_*` plus **mask_miou** / **mask_dice** from `baseline_common`).

## Data (see `task.md`)

| Role | Path |
|------|------|
| Annotations JSON (cleaned) | `/datasets/work/icvg-phd-marsfield/work/all_crops/cleaned_json/` |
| Images | `/datasets/work/icvg-phd-marsfield/work/all_crops/images/` |
| Train split | `_PROJ_/labelingv2/singular_sam3_train.json` (falls back to `labelingv3/` and `singluar_*` spelling if needed) |
| Val / test | same pattern for `singular_sam3_val.json`, `singular_sam3_test.json` |

`--dataset-root` defaults to `/datasets/work/icvg-phd-marsfield/work/all_crops` so COCO `file_name` entries resolve the same way as other baselines.

## Environment

Recommended Python: your Grounding DINO install (e.g. `/datasets/work/icvg-phd-marsfield/work/Feng/conda_envs/insect/bin/python`).

That environment needs:

- `torch`, `torchvision`
- Grounding DINO (`groundingdino` on `PYTHONPATH` or installed)
- `segment_anything` plus the SAM ViT‑H checkpoint (`sam_vit_h_4b8939.pth`)
- `pycocotools`, `numpy`, `tqdm`, Supervision (`supervision`), and other deps from the official Grounding DINO demos

If `groundingdino` was built from this repo clone, prepend it to `PYTHONPATH`, for example:

```bash
export PYTHONPATH=/scratch3/xu097/proj06/third_party/GroundingDINO:${PYTHONPATH}
```

## Checkpoints

- **Grounding DINO**: pass `--gdino-weights /path/to/groundingdino_swint_ogc.pth` or set **`GROUNDING_DINO_WEIGHTS`**.
- **SAM ViT‑H**: pass `--sam-checkpoint /path/to/sam_vit_h_4b8939.pth` or set **`SAM_VIT_H_PATH`**.

Default `--gdino-config` points at `proj06/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`.

## Run

From the **`proj06` root** (`/scratch3/xu097/proj06`):

```bash
YOUR_PYTHON=/datasets/work/icvg-phd-marsfield/work/Feng/conda_envs/insect/bin/python
export PYTHONPATH=/scratch3/xu097/proj06/third_party/GroundingDINO:${PYTHONPATH}
export GROUNDING_DINO_WEIGHTS=/path/to/groundingdino_swint_ogc.pth
export SAM_VIT_H_PATH=/path/to/sam_vit_h_4b8939.pth

cd /scratch3/xu097/proj06

"$YOUR_PYTHON" -m baseline7.eval_gdino_sam_zeroshot --split test --device cuda
```

Common options (mirrors baseline 4):

- `--gt-json /path/to/coco.json` instead of `--split`
- `--out-dir outputs/baseline7_gdino_sam` (timestamped subdirectory by default)
- `--out-flat` to drop timestamp folder and bake `run_id` into filenames
- `--box-threshold`, `--text-threshold` for Grounding DINO
- `--topk-per-category K` (per image, **0** = no cap)
- `--use-noun-phrase-prompts` for noun-phrase prompts instead of category names

Outputs: `baseline7_zs_<split>_predictions.json`, `*_metrics.json`, `*_metrics.txt` under the run directory.

## Notes

- Evaluation uses **pycocotools only** (no `sam3` import) so the same conda env can run inference and metrics.
- GDINO phrase strings are matched to categories with **lowercased, stripped** names; ensure category names in COCO match what the detector returns or tune thresholds.
