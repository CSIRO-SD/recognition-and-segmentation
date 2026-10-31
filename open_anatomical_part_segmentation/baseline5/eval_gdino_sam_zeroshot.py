#!/usr/bin/env python3
"""Baseline 7: Grounding DINO (text → boxes) + SAM (ViT‑H box prompts → masks).

Uses the same captioning policy as SAM3 singular baselines: COCO category ``name`` per
GT image via ``phrases_per_image(..., category_names_only=True)``.

Evaluates mask AP/AR (pycocotools) plus ``mask_miou`` / ``mask_dice`` (baseline_common).

Requires the Grounding DINO / ``segment_anything`` stack on ``PYTHONPATH`` (see readme).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
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
from baseline_common.run_outputs import resolve_eval_output_dir  # noqa: E402
from baseline_common.sam3_coco_utils import (  # noqa: E402
    category_name_to_id,
    compute_mask_miou_dice,
    load_coco_gt,
    metrics_to_serializable,
    phrases_per_image,
    resolve_dataset_image_path,
)

DEFAULT_SPLIT_GT: dict[str, Path] = {
    "train": _PROJ / "labelingv2" / "singular_sam3_train.json",
    "val": _PROJ / "labelingv2" / "singular_sam3_val.json",
    "test": _PROJ / "labelingv2" / "singular_sam3_test.json",
}

DEFAULT_DATASET_ROOT = Path("/datasets/work/icvg-phd-marsfield/work/all_crops")
DEFAULT_GDINO_CONFIG = (
    _PROJ
    / "third_party"
    / "GroundingDINO"
    / "groundingdino"
    / "config"
    / "GroundingDINO_SwinT_OGC.py"
)


def resolve_labeling_json(preferred: Path) -> Path:
    if preferred.is_file():
        return preferred
    name = preferred.name
    if name.startswith("singular_"):
        alt = preferred.with_name(name.replace("singular_", "singluar_", 1))
        if alt.is_file():
            return alt
    if name.startswith("singluar_"):
        alt = preferred.with_name(name.replace("singluar_", "singular_", 1))
        if alt.is_file():
            return alt
    return preferred


def resolve_split_json(preferred: Path) -> Path:
    p = resolve_labeling_json(preferred)
    if p.is_file():
        return p
    if preferred.parent.name == "labelingv2":
        alt = preferred.parent.parent / "labelingv3" / preferred.name
        p2 = resolve_labeling_json(alt)
        if p2.is_file():
            return p2
    return p


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


def run_gdino_sam_inference(
    *,
    coco_gt: dict,
    dataset_root: Path,
    gdino_model,
    sam_predictor,
    device: str,
    box_threshold: float,
    text_threshold: float,
    topk_per_category: int,
    category_names_only: bool,
    empty_cuda_cache_each_image: bool,
) -> dict[int, dict[str, torch.Tensor]]:
    from groundingdino.util.inference import load_image as gdino_load_image
    from groundingdino.util.inference import predict as gdino_predict

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
        desc="GroundingDINO + SAM",
    ):
        rel = id_to_file[image_id]
        img_path = resolve_dataset_image_path(dataset_root, rel)
        if img_path is None:
            skipped_missing += 1
            logging.warning("Skipping image_id=%s: missing file %s", image_id, rel)
            continue

        caption = " . ".join(phrases)

        image_np, gdino_tensor = gdino_load_image(str(img_path))
        h, w = image_np.shape[0], image_np.shape[1]
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

        sam_predictor.set_image(image_np)

        box_chunks: list[torch.Tensor] = []
        score_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []

        for scr, bbox_xyxy, cat_id in detections_raw:
            x0, y0, x1, y1 = bbox_xyxy.tolist()
            x0 = max(0.0, min(float(x0), float(w - 1)))
            y0 = max(0.0, min(float(y0), float(h - 1)))
            x1 = max(x0 + 1.0, min(float(x1), float(w)))
            y1 = max(y0 + 1.0, min(float(y1), float(h)))
            sam_box = np.array([x0, y0, x1, y1], dtype=np.float32)

            m, sam_score, _ = sam_predictor.predict(
                box=sam_box,
                multimask_output=False,
            )
            mask_best = torch.from_numpy(m[0]).float()

            box_chunks.append(bbox_xyxy.unsqueeze(0))
            score_chunks.append(torch.tensor([scr], dtype=torch.float32))
            label_chunks.append(torch.tensor([cat_id], dtype=torch.int64))
            mask_chunks.append(mask_best.unsqueeze(0).unsqueeze(0))
            del sam_score  # SAM IoU logits unused; detector score retained

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
            f"[gdino_sam] skipped {skipped_missing} image(s) with missing files "
            f"under {dataset_root}"
        )
        logging.warning(msg)
        print(msg, flush=True)

    return predictions


def resolve_gt_json_and_tag(args: argparse.Namespace) -> tuple[Path, str]:
    if args.gt_json is not None:
        p = args.gt_json.resolve()
        return p, p.stem
    assert args.split is not None
    cand = DEFAULT_SPLIT_GT[args.split]
    resolved = resolve_split_json(cand)
    return resolved.resolve(), args.split


def _weights_from_cli_or_env(cli: Path | None, env_name: str) -> Path | None:
    if cli is not None:
        return cli
    e = os.environ.get(env_name, "").strip()
    return Path(e) if e else None


def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--split",
        choices=sorted(DEFAULT_SPLIT_GT.keys()),
        help="Predefined singular splits (task.md labelingv2, fallback labelingv3).",
    )
    g.add_argument(
        "--gt-json",
        type=Path,
        help="COCO JSON with GT annotations (overrides --split).",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing image paths from COCO ``file_name``.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJ / "outputs" / "baseline7_gdino_sam",
        help="Base output directory (timestamp subfolder by default).",
    )
    ap.add_argument("--run-id", type=str, default=None, metavar="ID")
    ap.add_argument(
        "--out-flat",
        action="store_true",
        help="Write files directly under --out-dir with run_id in names.",
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
        help="Grounding DINO checkpoint (.pth). Fallback: env GROUNDING_DINO_WEIGHTS.",
    )
    ap.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="SAM ViT-H checkpoint. Fallback: env SAM_VIT_H_PATH.",
    )
    ap.add_argument("--sam-model-type", type=str, default="vit_h")
    ap.add_argument("--box-threshold", type=float, default=0.35)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument(
        "--topk-per-category",
        type=int,
        default=0,
        help="Max GDINO boxes per category per image (0 = unlimited).",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    ap.add_argument(
        "--use-noun-phrase-prompts",
        action="store_true",
        help="Use noun_phrase when present (baseline1-style). Default: category name only.",
    )
    args = ap.parse_args()

    gt_json, name_tag = resolve_gt_json_and_tag(args)

    gdino_ckpt = _weights_from_cli_or_env(
        args.gdino_weights, "GROUNDING_DINO_WEIGHTS"
    )
    sam_ckpt = _weights_from_cli_or_env(args.sam_checkpoint, "SAM_VIT_H_PATH")

    if not gt_json.is_file():
        print(f"Missing GT JSON: {gt_json}", file=sys.stderr)
        return 1
    if not args.dataset_root.is_dir():
        print(f"Missing dataset root: {args.dataset_root}", file=sys.stderr)
        return 1
    if gdino_ckpt is None or not gdino_ckpt.is_file():
        print(
            "Set --gdino-weights or GROUNDING_DINO_WEIGHTS to a Grounding DINO .pth",
            file=sys.stderr,
        )
        return 1
    if sam_ckpt is None or not sam_ckpt.is_file():
        print(
            "Set --sam-checkpoint or SAM_VIT_H_PATH to SAM ViT-H weights",
            file=sys.stderr,
        )
        return 1
    if not args.gdino_config.is_file():
        print(f"Missing GDINO config: {args.gdino_config}", file=sys.stderr)
        return 1

    eff_out, run_id = resolve_eval_output_dir(
        args.out_dir, run_id=args.run_id, flat=args.out_flat
    )
    prefix = f"baseline7_zs_{name_tag}"
    if args.out_flat:
        pred_path = eff_out / f"{prefix}_predictions_{run_id}.json"
        metrics_path = eff_out / f"{prefix}_metrics_{run_id}.json"
        metrics_txt = eff_out / f"{prefix}_metrics_{run_id}.txt"
    else:
        pred_path = eff_out / f"{prefix}_predictions.json"
        metrics_path = eff_out / f"{prefix}_metrics.json"
        metrics_txt = eff_out / f"{prefix}_metrics.txt"

    from groundingdino.util.inference import load_model as gdino_load_model
    from segment_anything import SamPredictor
    from segment_anything import sam_model_registry

    gdino_model = gdino_load_model(
        str(args.gdino_config),
        str(gdino_ckpt),
        device=args.device,
    )

    sam_net = sam_model_registry[args.sam_model_type](checkpoint=str(sam_ckpt))
    sam_net.to(device=args.device)
    sam_predictor = SamPredictor(sam_net)

    coco_gt = load_coco_gt(gt_json)
    category_names_only = not args.use_noun_phrase_prompts

    preds = run_gdino_sam_inference(
        coco_gt=coco_gt,
        dataset_root=args.dataset_root,
        gdino_model=gdino_model,
        sam_predictor=sam_predictor,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        topk_per_category=args.topk_per_category,
        category_names_only=category_names_only,
        empty_cuda_cache_each_image=True,
    )

    dump_coco_segm_predictions_json(preds, pred_path)

    metrics = evaluate_coco_segm_offline(gt_json, pred_path)
    metrics.update(compute_mask_miou_dice(gt_json, pred_path))
    metrics_path.write_text(
        json.dumps(metrics_to_serializable(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [f"{k}: {v}" for k, v in sorted(metrics.items())]
    metrics_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Run ID: {run_id}")
    print(f"GT JSON: {gt_json}")
    print(f"Category-name prompts only: {category_names_only}")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote metrics (txt): {metrics_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
