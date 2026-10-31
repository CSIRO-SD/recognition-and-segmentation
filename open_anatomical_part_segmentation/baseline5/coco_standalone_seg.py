"""COCO segmentation prediction dump + mask AP evaluation without sam3 (pycocotools only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_utils
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Match sam3/sam3/eval/coco_eval_offline.py metric key suffix order.
COCO_SEGM_METRICS = [
    "AP",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_maxDets@1",
    "AR_maxDets@10",
    "AR_maxDets@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def _boxes_xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)


def _mask_tensor_to_rle(mask_hw: torch.Tensor) -> dict[str, Any]:
    """Binary mask [H, W] float or bool -> COCO RLE dict."""
    m = (mask_hw > 0.5).detach().cpu().numpy().astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(m))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def dump_coco_segm_predictions_json(
    predictions: dict[int, dict[str, torch.Tensor]],
    dump_path: Path,
) -> None:
    """Write COCO segmentation results list (RLE, bbox xywh, score, category_id).

    ``predictions`` maps ``image_id`` to dict with ``boxes`` (xyxy), ``scores``,
    ``labels`` (category_id), ``masks`` ([N,1,H,W] or [N,H,W] float).
    """
    coco_results: list[dict[str, Any]] = []
    for image_id, pred in sorted(predictions.items()):
        if pred["scores"].numel() == 0:
            continue
        scores = pred["scores"].detach().cpu()
        labels = pred["labels"].detach().cpu().long()
        boxes = pred["boxes"].detach().cpu()
        masks = pred["masks"].detach().cpu()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        boxes_xywh = _boxes_xyxy_to_xywh(boxes).tolist()
        score_list = scores.tolist()
        label_list = [int(x) for x in labels.tolist()]

        h, w = int(masks.shape[-2]), int(masks.shape[-1])
        for k in range(masks.shape[0]):
            rle = _mask_tensor_to_rle(masks[k])
            area = float(mask_utils.area(rle))
            coco_results.append(
                {
                    "image_id": int(image_id),
                    "category_id": label_list[k],
                    "segmentation": rle,
                    "score": float(score_list[k]),
                    "bbox": boxes_xywh[k],
                    "area": area / float(h * w) if h * w else 0.0,
                }
            )

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text(json.dumps(coco_results), encoding="utf-8")


def evaluate_coco_segm_offline(gt_json: Path, pred_json: Path) -> dict[str, float]:
    """Run COCO mask AP/AR; keys ``coco_eval_segm_*`` match SAM3 offline naming."""
    coco_gt = COCO(str(gt_json))
    raw = pred_json.read_text(encoding="utf-8").strip()
    if not raw or raw == "[]":
        return {f"coco_eval_segm_{m}": 0.0 for m in COCO_SEGM_METRICS}

    coco_dt = coco_gt.loadRes(str(pred_json))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    out: dict[str, float] = {}
    for i, name in enumerate(COCO_SEGM_METRICS):
        out[f"coco_eval_segm_{name}"] = float(coco_eval.stats[i])
    return out
