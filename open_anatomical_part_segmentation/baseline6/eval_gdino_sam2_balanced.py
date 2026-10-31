#!/usr/bin/env python3
"""Baseline 8: Grounding DINO + SAM 2 (Hugging Face) on balanced CSV split.

Builds COCO GT from ``labelingv3/data_split_balanced.csv`` (default ``split=test`` rows),
runs GDINO captions from GT category names, SAM2 box prompts via ``Sam2Processor`` /
``Sam2Model``, then COCO mask AP + mask mIoU/Dice (no ``sam3`` dependency).

Example::

    insect/bin/python -m baseline8.eval_gdino_sam2_balanced \\
        --csv-split test --device cuda \\
        --gdino-weights ... --sam2-model facebook/sam2.1-hiera-large
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torchvision.ops import box_convert as tv_box_convert
from tqdm import tqdm

_PROJ = Path(__file__).resolve().parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from baseline5.coco_standalone_seg import (  # noqa: E402
    dump_coco_segm_predictions_json,
    evaluate_coco_segm_offline,
)
from baseline6.balanced_coco_gt import build_coco_for_balanced_csv_split  # noqa: E402
from baseline_common.run_outputs import resolve_eval_output_dir  # noqa: E402
from baseline_common.sam3_coco_utils import (  # noqa: E402
    category_name_to_id,
    compute_mask_miou_dice,
    metrics_to_serializable,
    phrases_per_image,
    resolve_dataset_image_path,
)

DEFAULT_DATASET_ROOT = Path("/datasets/work/icvg-phd-marsfield/work/all_crops")
DEFAULT_SPLIT_CSV = _PROJ / "labelingv3" / "data_split_balanced.csv"
DEFAULT_NORM_MAP = _PROJ / "labelingv3" / "label_normalization.json"
DEFAULT_GDINO_CONFIG = (
    _PROJ
    / "third_party"
    / "GroundingDINO"
    / "groundingdino"
    / "config"
    / "GroundingDINO_SwinT_OGC.py"
)

# Task-oriented metric lines (keys match offline JSON; labels for humans).
FOCUS_METRIC_LABELS: list[tuple[str, str]] = [
    ("coco_eval_segm_AP", "coco_eval_segm_Average_precision (mask AP)"),
    ("coco_eval_segm_AP_large", "coco_eval_segm_AP_large"),
    ("coco_eval_segm_AP_medium", "coco_eval_segm_AP_medium"),
    ("coco_eval_segm_AP_small", "coco_eval_segm_AP_small"),
    ("coco_eval_segm_AR_large", "coco_eval_segm_Average_Recall_large"),
    ("coco_eval_segm_AR_medium", "coco_eval_segm_AR_medium"),
    ("coco_eval_segm_AR_small", "coco_eval_segm_AR_small"),
    ("coco_eval_segm_AR_maxDets@100", "coco_eval_segm_Average_Recall (AR @100 dets)"),
    ("mask_dice", "mask_dice"),
    ("mask_miou", "mask_miou"),
]


def build_normalized_name_to_cat(coco_imageset: dict) -> dict[str, int]:
    return {
        str(c["name"]).lower().strip(): int(c["id"])
        for c in coco_imageset["categories"]
    }


def phrase_to_cat_id(phrase: str, norm_map: dict[str, int]) -> int | None:
    key = phrase.lower().strip().replace(".", "").strip()
    return norm_map.get(key)


def apply_topk_per_category(
    detections: list[tuple[float, torch.Tensor, int]],
    topk: int,
) -> list[tuple[float, torch.Tensor, int]]:
    if topk <= 0:
        return detections
    by_cat: dict[int, list[tuple[float, torch.Tensor, int]]] = defaultdict(list)
    for score, box, cid in detections:
        by_cat[cid].append((score, box, cid))
    out: list[tuple[float, torch.Tensor, int]] = []
    for _cid, lst in sorted(by_cat.items()):
        lst.sort(key=lambda t: -t[0])
        out.extend(lst[:topk])
    return out


def sam2_mask_from_box(
    *,
    pil_rgb,
    xyxy: tuple[float, float, float, float],
    model,
    processor,
    device: str,
) -> torch.Tensor:
    """One box [x0,y0,x1,y1] in pixel coords → float mask [H,W] in original resolution."""
    from PIL import Image

    if not isinstance(pil_rgb, Image.Image):
        raise TypeError("pil_rgb must be PIL.Image")

    x0, y0, x1, y1 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
    input_boxes = [[[x0, y0, x1, y1]]]
    inputs = processor(
        images=pil_rgb, input_boxes=input_boxes, return_tensors="pt"
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
    )[0]
    # (num_objects, num_mask_out, H, W) -> take first object, first mask
    m = masks[0, 0].float()
    return m


def run_gdino_sam2_inference(
    *,
    coco_gt: dict,
    dataset_root: Path,
    gdino_model,
    sam2_model,
    sam2_processor,
    device: str,
    box_threshold: float,
    text_threshold: float,
    topk_per_category: int,
    category_names_only: bool,
    empty_cuda_cache_each_image: bool,
) -> dict[int, dict[str, torch.Tensor]]:
    from groundingdino.util.inference import load_image as gdino_load_image
    from groundingdino.util.inference import predict as gdino_predict
    from PIL import Image

    id_to_file = {int(im["id"]): im["file_name"] for im in coco_gt["images"]}
    norm_cat = build_normalized_name_to_cat(coco_gt)
    literal_name_to_id = category_name_to_id(coco_gt)

    img_to_phrases = phrases_per_image(
        coco_gt, category_names_only=category_names_only
    )

    predictions: dict[int, dict[str, torch.Tensor]] = {}
    skipped_missing = 0

    for image_id, phrases in tqdm(
        sorted(img_to_phrases.items()),
        desc="GroundingDINO + SAM2",
    ):
        rel = id_to_file[image_id]
        img_path = resolve_dataset_image_path(dataset_root, rel)
        if img_path is None:
            skipped_missing += 1
            logging.warning("Skipping image_id=%s: missing file %s", image_id, rel)
            continue

        caption = " . ".join(phrases)

        _rgb_np, gdino_tensor = gdino_load_image(str(img_path))
        h, w = int(_rgb_np.shape[0]), int(_rgb_np.shape[1])
        gdino_tensor = gdino_tensor.to(device)

        with torch.no_grad():
            bx, logits, gdino_phrases = gdino_predict(
                gdino_model,
                gdino_tensor,
                caption,
                box_threshold,
                text_threshold,
                device=device,
            )

        if bx.numel() == 0:
            continue

        scal = torch.tensor([w, h, w, h], dtype=bx.dtype, device=bx.device)
        bx_pix = bx * scal
        xyxy = tv_box_convert(bx_pix, in_fmt="cxcywh", out_fmt="xyxy")

        gdino_scores = logits.to(dtype=torch.float32)
        detections_raw: list[tuple[float, torch.Tensor, int]] = []
        for j in range(xyxy.shape[0]):
            phrase = gdino_phrases[j]
            cid = phrase_to_cat_id(phrase, norm_cat)
            if cid is None:
                cid = literal_name_to_id.get(str(phrase).strip())
            if cid is None:
                logging.warning(
                    "GDINO phrase not mapped to category: %r (image_id=%s)",
                    phrase,
                    image_id,
                )
                continue
            scr = float(gdino_scores[j].item())
            detections_raw.append((scr, xyxy[j].detach().cpu(), cid))

        detections_raw = apply_topk_per_category(
            detections_raw, topk_per_category
        )
        if not detections_raw:
            continue

        pil_rgb = Image.open(img_path).convert("RGB")

        box_chunks: list[torch.Tensor] = []
        score_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []

        for scr, bbox_xyxy_t, cat_id in detections_raw:
            x0, y0, x1, y1 = bbox_xyxy_t.tolist()
            x0 = max(0.0, min(float(x0), float(w - 1)))
            y0 = max(0.0, min(float(y0), float(h - 1)))
            x1 = max(x0 + 1.0, min(float(x1), float(w)))
            y1 = max(y0 + 1.0, min(float(y1), float(h)))

            mask_t = sam2_mask_from_box(
                pil_rgb=pil_rgb,
                xyxy=(x0, y0, x1, y1),
                model=sam2_model,
                processor=sam2_processor,
                device=device,
            )

            box_chunks.append(torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32))
            score_chunks.append(torch.tensor([scr], dtype=torch.float32))
            label_chunks.append(torch.tensor([cat_id], dtype=torch.int64))
            mask_chunks.append(mask_t.unsqueeze(0).unsqueeze(0))

        boxes = torch.cat(box_chunks, dim=0)
        scores = torch.cat(score_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        masks = torch.cat(mask_chunks, dim=0)

        predictions[int(image_id)] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "masks": masks,
        }

        if empty_cuda_cache_each_image and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if skipped_missing:
        msg = (
            f"[baseline8 gdino_sam2] skipped {skipped_missing} image(s) missing "
            f"under {dataset_root}"
        )
        logging.warning(msg)
        print(msg, flush=True)

    return predictions


def write_focus_metrics_summary(metrics: dict[str, float], path: Path) -> None:
    lines: list[str] = []
    for key, label in FOCUS_METRIC_LABELS:
        if key in metrics:
            lines.append(f"{label}: {metrics[key]}")
    lines.append("")
    lines.append("(Full keys: see companion *_metrics.json and *_metrics.txt.)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _weights_from_cli_or_env(cli: Path | None, env_name: str) -> Path | None:
    if cli is not None:
        return cli
    e = os.environ.get(env_name, "").strip()
    return Path(e) if e else None


def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split-csv",
        type=Path,
        default=DEFAULT_SPLIT_CSV,
        help="Balanced split CSV with columns json_path, split.",
    )
    ap.add_argument(
        "--norm-map",
        type=Path,
        default=DEFAULT_NORM_MAP,
        help="label_normalization.json (canonical class names).",
    )
    ap.add_argument(
        "--csv-split",
        type=str,
        default="test",
        metavar="NAME",
        help="Eval split column: test, val, training (or train). Default: test.",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="all_crops root; COCO paths are images/<file>.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJ / "outputs" / "baseline8_gdino_sam2_balanced",
        help="Base output directory (timestamp subfolder unless --out-flat).",
    )
    ap.add_argument("--run-id", type=str, default=None, metavar="ID")
    ap.add_argument(
        "--out-flat",
        action="store_true",
        help="Write outputs in --out-dir with run_id in filenames.",
    )
    ap.add_argument(
        "--gdino-config",
        type=Path,
        default=DEFAULT_GDINO_CONFIG,
        help="Grounding DINO SLConfig Python file.",
    )
    ap.add_argument(
        "--gdino-weights",
        type=Path,
        default=None,
        help="GDINO checkpoint; or env GROUNDING_DINO_WEIGHTS.",
    )
    ap.add_argument(
        "--sam2-model",
        type=str,
        default=os.environ.get("SAM2_HF_MODEL", "").strip()
        or "facebook/sam2.1-hiera-large",
        help="Hugging Face id for Sam2Model / Sam2Processor.",
    )
    ap.add_argument("--box-threshold", type=float, default=0.35)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument(
        "--topk-per-category",
        type=int,
        default=0,
        metavar="K",
        help="Keep top-K GDINO boxes per category per image (0 = no limit).",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    ap.add_argument(
        "--use-noun-phrase-prompts",
        action="store_true",
        help="Use noun_phrase when present instead of category name only.",
    )
    args = ap.parse_args()

    if not args.split_csv.is_file():
        print(f"Missing split CSV: {args.split_csv}", file=sys.stderr)
        return 1
    if not args.norm_map.is_file():
        print(f"Missing label normalization map: {args.norm_map}", file=sys.stderr)
        return 1
    if not args.dataset_root.is_dir():
        print(f"Missing dataset root: {args.dataset_root}", file=sys.stderr)
        return 1

    gdino_ckpt = _weights_from_cli_or_env(
        args.gdino_weights, "GROUNDING_DINO_WEIGHTS"
    )
    if gdino_ckpt is None or not gdino_ckpt.is_file():
        print(
            "Set --gdino-weights or GROUNDING_DINO_WEIGHTS",
            file=sys.stderr,
        )
        return 1
    if not args.gdino_config.is_file():
        print(f"Missing GDINO config: {args.gdino_config}", file=sys.stderr)
        return 1

    try:
        coco_gt = build_coco_for_balanced_csv_split(
            split_csv=args.split_csv,
            norm_map=args.norm_map,
            split_name=args.csv_split,
        )
    except (ValueError, FileNotFoundError, KeyError, SystemExit) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not coco_gt["images"]:
        print("COCO export contains no images for this split.", file=sys.stderr)
        return 1

    eff_out, run_id = resolve_eval_output_dir(
        args.out_dir, run_id=args.run_id, flat=args.out_flat
    )
    split_tag = args.csv_split.strip().lower().replace("/", "_").replace(" ", "")
    prefix = f"baseline8_balanced_{split_tag}"
    if args.out_flat:
        pred_path = eff_out / f"{prefix}_predictions_{run_id}.json"
        metrics_path = eff_out / f"{prefix}_metrics_{run_id}.json"
        metrics_txt = eff_out / f"{prefix}_metrics_{run_id}.txt"
        summary_path = eff_out / f"{prefix}_metrics_focus_{run_id}.txt"
        gt_path = eff_out / f"{prefix}_gt_{run_id}.json"
    else:
        pred_path = eff_out / f"{prefix}_predictions.json"
        metrics_path = eff_out / f"{prefix}_metrics.json"
        metrics_txt = eff_out / f"{prefix}_metrics.txt"
        summary_path = eff_out / f"{prefix}_metrics_focus.txt"
        gt_path = eff_out / f"{prefix}_gt.json"

    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(coco_gt, ensure_ascii=False) + "\n", encoding="utf-8")

    from groundingdino.util.inference import load_model as gdino_load_model
    from transformers import Sam2Model, Sam2Processor

    gdino_model = gdino_load_model(
        str(args.gdino_config),
        str(gdino_ckpt),
        device=args.device,
    )

    print(f"Loading SAM2 ({args.sam2_model})…", flush=True)
    sam2_processor = Sam2Processor.from_pretrained(args.sam2_model)
    sam2_model = Sam2Model.from_pretrained(args.sam2_model).to(args.device)
    sam2_model.eval()

    category_names_only = not args.use_noun_phrase_prompts

    preds = run_gdino_sam2_inference(
        coco_gt=coco_gt,
        dataset_root=args.dataset_root,
        gdino_model=gdino_model,
        sam2_model=sam2_model,
        sam2_processor=sam2_processor,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        topk_per_category=args.topk_per_category,
        category_names_only=category_names_only,
        empty_cuda_cache_each_image=True,
    )

    dump_coco_segm_predictions_json(preds, pred_path)

    metrics = evaluate_coco_segm_offline(gt_path, pred_path)
    metrics.update(compute_mask_miou_dice(gt_path, pred_path))
    metrics_path.write_text(
        json.dumps(metrics_to_serializable(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    mt = [f"{k}: {v}" for k, v in sorted(metrics.items())]
    metrics_txt.write_text("\n".join(mt) + "\n", encoding="utf-8")
    write_focus_metrics_summary(metrics, summary_path)

    print(f"Run ID: {run_id}")
    print(f"CSV split: {args.csv_split}  GT images: {len(coco_gt['images'])}")
    print(f"Wrote GT: {gt_path}")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote metrics (txt): {metrics_txt}")
    print(f"Wrote metrics (focus summary): {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
