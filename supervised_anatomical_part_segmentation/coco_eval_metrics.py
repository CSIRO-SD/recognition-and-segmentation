"""COCO mask evaluation with SAM3-style **image-normalized** area buckets (see facebookresearch/sam3 `coco_eval.py`).

Area for each instance is ``mask_pixel_count / (image_width * image_height)`` (same as SAM3 with
``use_normalized_areas=True``). Buckets::

  tiny        [0, 0.001)
  small       [0.001, 0.01)
  medium      [0.01, 0.1)
  large       [0.1, 0.5)
  huge        [0.5, 0.95)
  whole_image [0.95, +inf)

``AP_all`` / ``AR_all use the ``all`` bucket [0, 1e5] (same pattern as SAM3 code).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

# Identical numeric ranges to SAM3 CocoEvaluator.reset() when use_normalized_areas=True.
SAM3_NORMALIZED_AREA_RNG = [
    [0, 1e5],
    [0, 0.001],
    [0.001, 0.01],
    [0.01, 0.1],
    [0.1, 0.5],
    [0.5, 0.95],
    [0.95, 1e5],
]
SAM3_NORMALIZED_AREA_LBL = [
    "all",
    "tiny",
    "small",
    "medium",
    "large",
    "huge",
    "whole_image",
]


def build_sam3_style_gt_coco(annotation_path: str | Path) -> COCO:
    """Load COCO JSON and replace each ann's ``area`` with area / (image width * height)."""
    with open(annotation_path, encoding="utf-8") as f:
        d = json.load(f)
    img_area = {int(im["id"]): float(im["width"] * im["height"]) for im in d["images"]}
    d = copy.deepcopy(d)
    for a in d["annotations"]:
        denom = img_area[int(a["image_id"])]
        a["area"] = float(a["area"]) / denom if denom > 0 else 0.0
    coco = COCO()
    coco.dataset = d
    coco.createIndex()
    return coco


def _mask_to_rle(binary_hw: np.ndarray) -> dict:
    """binary_hw: bool or 0/1 uint8 HxW."""
    m = np.asfortranarray(binary_hw.astype(np.uint8))
    rle = mask_utils.encode(m)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _bbox_xywh_from_binary(mask_u8: np.ndarray) -> list[float]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return [0.0, 0.0, 1.0, 1.0]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return [x0, y0, x1 - x0 + 1.0, y1 - y0 + 1.0]


def detection_dict_for_coco(
    *,
    image_id: int,
    coco_category_id: int,
    mask_hw: np.ndarray,
    score: float,
    orig_width: int,
    orig_height: int,
) -> dict[str, Any]:
    """mask_hw: HxW on **original** image resolution; fills COCO detection fields + SAM3-normalized area."""
    if mask_hw.shape != (orig_height, orig_width):
        raise ValueError(f"mask {mask_hw.shape} vs orig ({orig_height},{orig_width})")
    m = (mask_hw > 0).astype(np.uint8)
    denom = float(orig_width * orig_height)
    area_n = float(m.sum()) / denom if denom > 0 else 0.0
    rle = _mask_to_rle(m)
    bbox = _bbox_xywh_from_binary(m)
    return {
        "image_id": int(image_id),
        "category_id": int(coco_category_id),
        "segmentation": rle,
        "score": float(score),
        "bbox": bbox,
        "area": area_n,
    }


def run_coco_segm_eval(
    coco_gt: COCO,
    detections: list[dict[str, Any]],
    *,
    max_dets: list[int] | None = None,
) -> dict[str, float]:
    """Run ``segm`` COCOeval with SAM3 area ranges; return AP/AR per bucket + aliases."""
    if not detections:
        z = {f"AP_{lbl}": 0.0 for lbl in SAM3_NORMALIZED_AREA_LBL}
        z.update({f"AR_{lbl}": 0.0 for lbl in SAM3_NORMALIZED_AREA_LBL})
        z.update(
            {
                "AP": 0.0,
                "AR": 0.0,
                "APsmall": 0.0,
                "APmedium": 0.0,
                "APlarge": 0.0,
                "ARsmall": 0.0,
                "ARmedium": 0.0,
                "ARlarge": 0.0,
            },
        )
        return z

    coco_dt = coco_gt.loadRes(detections)
    # pycocotools overwrites ``area`` with pixel count; re-normalize like GT for SAM3 buckets.
    for ann in coco_dt.dataset["annotations"]:
        img = coco_gt.loadImgs(ann["image_id"])[0]
        denom = float(img["width"] * img["height"])
        ann["area"] = float(mask_utils.area(ann["segmentation"])) / denom if denom > 0 else 0.0
    coco_dt.createIndex()

    ev = COCOeval(coco_gt, coco_dt, iouType="segm")
    ev.params.maxDets = max_dets or [1, 10, 100]
    ev.params.areaRng = SAM3_NORMALIZED_AREA_RNG
    ev.params.areaRngLbl = SAM3_NORMALIZED_AREA_LBL
    ev.params.useCats = 1

    ev.evaluate()
    ev.accumulate()
    metrics = _summarize_segm_to_dict(ev)

    metrics["AP"] = metrics["AP_all"]
    metrics["AR"] = metrics["AR_all"]
    metrics["APsmall"] = metrics["AP_small"]
    metrics["APmedium"] = metrics["AP_medium"]
    metrics["APlarge"] = metrics["AP_large"]
    metrics["ARsmall"] = metrics["AR_small"]
    metrics["ARmedium"] = metrics["AR_medium"]
    metrics["ARlarge"] = metrics["AR_large"]
    return metrics


def _summarize_segm_to_dict(coco_eval: COCOeval) -> dict[str, float]:
    """Mean AP / AR over categories (standard COCO), per area label + maxDets=100."""
    if not coco_eval.eval:
        raise RuntimeError("coco_eval.run accumulate first")
    p = coco_eval.params
    precision = coco_eval.eval["precision"]  # T x R x K x A x M
    recall = coco_eval.eval["recall"]  # T x K x A x M
    out: dict[str, float] = {}

    def pack(ap: int, area_lbl: str, maxdet: int) -> float:
        aind = [i for i, l in enumerate(p.areaRngLbl) if l == area_lbl]
        mind = [i for i, m in enumerate(p.maxDets) if m == maxdet]
        if not aind or not mind:
            return 0.0
        if ap == 1:
            s = precision[:, :, :, aind, mind]
        else:
            s = recall[:, :, aind, mind]
        s = s[s > -1]
        return float(s.mean()) if len(s) else 0.0

    maxdet_eval = p.maxDets[-1]
    for lbl in p.areaRngLbl:
        out[f"AP_{lbl}"] = pack(1, lbl, maxdet_eval)
        out[f"AR_{lbl}"] = pack(0, lbl, maxdet_eval)
    return out


def semantic_region_metrics(
    hist: np.ndarray | torch.Tensor,
    num_classes: int,
    *,
    skip_background: bool = True,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Confusion ``hist[c_true, c_pred]`` → mIoU, macro **Dice**, and **IoU_per_class** (foreground).

    Per class (one-vs-rest): IoU = TP/(TP+FP+FN); Dice = 2·TP/(2·TP+FP+FN).

    ``class_names``: dense index like ``build_cat_maps`` — slot 0 unused; ``class_names[c]`` names label ``c``.
    If union is empty for a class, that class's IoU is reported as ``0.0``; **mIoU** still averages only
    classes with non-empty union (same as before).
    """
    if hasattr(hist, "cpu"):
        h = hist.cpu().numpy()
    else:
        h = np.asarray(hist, dtype=np.float64)
    start = 1 if skip_background else 0
    ious, dices = [], []
    iou_per: dict[str, float] = {}
    for c in range(start, num_classes):
        tp = h[c, c]
        fp = h[:, c].sum() - tp
        fn = h[c, :].sum() - tp
        denom_i = tp + fp + fn
        iou_c = float(tp / denom_i) if denom_i > 0 else 0.0
        if denom_i > 0:
            ious.append(iou_c)
        if class_names is not None and c < len(class_names) and class_names[c]:
            key = class_names[c]
        else:
            key = f"class_{c}"
        iou_per[key] = iou_c
        dice_d = 2 * tp + fp + fn
        if dice_d > 0:
            dices.append(2 * tp / dice_d)
    return {
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mean_Dice": float(np.mean(dices)) if dices else 0.0,
        "IoU_per_class": iou_per,
    }


def collect_mask_rcnn_coco_detections(
    model: Any,
    dataset: Any,
    device: Any,
    lbl2coco: dict[int, int],
    *,
    score_thresh: float = 0.05,
) -> list[dict[str, Any]]:
    """Mask R-CNN on TV instance dataset → COCO segm dets at **original** WH (SAM3-normalized ``area``)."""
    import torch
    import torch.nn.functional as F

    model.eval()
    detections: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            im_info = dataset.images[idx]
            image_id = int(im_info["id"])
            w0, h0 = int(im_info["width"]), int(im_info["height"])
            img, _target = dataset[idx]
            out = model([img.to(device)])[0]
            scores = out["scores"]
            labels = out["labels"]
            masks = out["masks"]
            if len(scores) == 0:
                continue
            for k in range(scores.shape[0]):
                if float(scores[k]) < score_thresh:
                    continue
                lab = int(labels[k].item())
                if lab == 0 or lab not in lbl2coco:
                    continue
                cid = lbl2coco[lab]
                m512 = masks[k, 0] > 0.5
                mo = (
                    F.interpolate(
                        m512.float().unsqueeze(0).unsqueeze(0),
                        size=(h0, w0),
                        mode="nearest",
                    )[0, 0]
                    .cpu()
                    .numpy()
                )
                detections.append(
                    detection_dict_for_coco(
                        image_id=image_id,
                        coco_category_id=cid,
                        mask_hw=mo,
                        score=float(scores[k]),
                        orig_width=w0,
                        orig_height=h0,
                    )
                )
    return detections


def run_mask_rcnn_test_metrics(
    coco_gt: COCO | None,
    model: Any,
    dataset: Any,
    device: Any,
    lbl2coco: dict[int, int],
    id2lbl: dict[int, int],
    *,
    num_classes: int,
    score_thresh: float = 0.05,
    coco_eval: bool = True,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """One pass over ``dataset``: optional SAM3-style COCO segm AP/AR + semantic confusion (mIoU / Dice).

    If ``coco_eval`` is False, skip building detections and pycocotools (only semantic metrics).
    """
    if coco_eval and coco_gt is None:
        raise ValueError("coco_gt is required when coco_eval=True")
    import torch
    import torch.nn.functional as F

    from coco_mask import index_coco_by_image, rasterize_semantic_mask

    by_im = index_coco_by_image(dataset.coco)
    hist = torch.zeros(num_classes, num_classes, dtype=torch.double, device=device)
    correct = 0
    total = 0
    detections: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            im_info = dataset.images[idx]
            image_id = int(im_info["id"])
            w0, h0 = int(im_info["width"]), int(im_info["height"])
            gt_u8 = rasterize_semantic_mask(by_im.get(image_id, []), id2lbl, w0, h0)
            gt_t = torch.from_numpy(gt_u8).long().to(device)
            img, _target = dataset[idx]
            out = model([img.to(device)])[0]
            scores = out["scores"]
            labels = out["labels"]
            masks = out["masks"]
            pred = torch.zeros((h0, w0), dtype=torch.long, device=device)
            if len(scores) > 0:
                sc = scores.cpu().numpy()
                lb = labels.cpu().numpy()
                order = np.argsort(sc)
                for k in order:
                    if float(sc[k]) < score_thresh:
                        continue
                    lab_dense = int(lb[k])
                    if lab_dense == 0 or lab_dense not in lbl2coco:
                        continue
                    cid = lbl2coco[lab_dense]
                    m512 = masks[k, 0] > 0.5
                    mo = (
                        F.interpolate(
                            m512.float().unsqueeze(0).unsqueeze(0),
                            size=(h0, w0),
                            mode="nearest",
                        )[0, 0]
                        > 0.5
                    )
                    pred[mo] = lab_dense
                    if coco_eval:
                        mo_np = mo.cpu().numpy()
                        detections.append(
                            detection_dict_for_coco(
                                image_id=image_id,
                                coco_category_id=cid,
                                mask_hw=mo_np,
                                score=float(scores[k]),
                                orig_width=w0,
                                orig_height=h0,
                            )
                        )
            p = pred.reshape(-1)
            t = gt_t.reshape(-1)
            correct += int((p == t).sum().item())
            total += int(t.numel())
            ind = num_classes * t.long() + p.long()
            hist += torch.bincount(ind, minlength=num_classes * num_classes).double().view(
                num_classes, num_classes
            )

    sem = semantic_region_metrics(
        hist, num_classes, skip_background=True, class_names=class_names
    )
    sem["pix_acc"] = float(correct / max(1, total))
    if not coco_eval:
        return sem
    coco_m = run_coco_segm_eval(coco_gt, detections)
    return {**coco_m, **sem}


def run_mask2former_test_metrics(
    coco_gt: COCO | None,
    model: Any,
    processor: Any,
    dataset: Any,
    device: Any,
    lbl2coco: dict[int, int],
    id2lbl: dict[int, int],
    *,
    num_classes: int,
    score_thresh: float = 0.35,
    coco_eval: bool = True,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """One pass: optional COCO mask AP/AR + painted semantic metrics."""
    if coco_eval and coco_gt is None:
        raise ValueError("coco_gt is required when coco_eval=True")
    import torch

    from coco_mask import index_coco_by_image, rasterize_semantic_mask

    by_im = index_coco_by_image(dataset.coco)
    hist = torch.zeros(num_classes, num_classes, dtype=torch.double, device=device)
    correct = 0
    total = 0
    detections: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            im_info = dataset.images[idx]
            image_id = int(im_info["id"])
            w0, h0 = int(im_info["width"]), int(im_info["height"])
            gt_u8 = rasterize_semantic_mask(by_im.get(image_id, []), id2lbl, w0, h0)
            gt_t = torch.from_numpy(gt_u8).long().to(device)
            pv = sample["pixel_values"].unsqueeze(0).to(device)
            out_m = model(pixel_values=pv)
            processed = processor.post_process_instance_segmentation(
                out_m,
                threshold=score_thresh,
                target_sizes=[(h0, w0)],
                return_binary_maps=True,
            )[0]
            maps = processed.get("segmentation")
            segs = processed.get("segments_info") or []
            pred = torch.zeros((h0, w0), dtype=torch.long, device=device)
            if maps is not None and len(segs) > 0:
                scores_flat = np.array([float(segs[j]["score"]) for j in range(len(segs))], dtype=np.float64)
                order = np.argsort(scores_flat)
                for j in order:
                    if j >= maps.shape[0]:
                        break
                    seg = segs[j]
                    dense_lbl = int(seg["label_id"]) + 1
                    if dense_lbl not in lbl2coco:
                        continue
                    cid = lbl2coco[dense_lbl]
                    mo = (maps[j].float().cpu().numpy() > 0.5).astype(np.uint8)
                    mo_t = torch.from_numpy(mo).bool().to(device)
                    pred[mo_t] = dense_lbl
                    if coco_eval:
                        detections.append(
                            detection_dict_for_coco(
                                image_id=image_id,
                                coco_category_id=cid,
                                mask_hw=mo,
                                score=float(seg["score"]),
                                orig_width=w0,
                                orig_height=h0,
                            )
                        )
            p = pred.reshape(-1)
            t = gt_t.reshape(-1)
            correct += int((p == t).sum().item())
            total += int(t.numel())
            ind = num_classes * t.long() + p.long()
            hist += torch.bincount(ind, minlength=num_classes * num_classes).double().view(
                num_classes, num_classes
            )

    sem = semantic_region_metrics(
        hist, num_classes, skip_background=True, class_names=class_names
    )
    sem["pix_acc"] = float(correct / max(1, total))
    if not coco_eval:
        return sem
    coco_m = run_coco_segm_eval(coco_gt, detections)
    return {**coco_m, **sem}


def collect_mask2former_coco_detections(
    model: Any,
    processor: Any,
    dataset: Any,
    device: Any,
    lbl2coco: dict[int, int],
    *,
    score_thresh: float = 0.35,
) -> list[dict[str, Any]]:
    """Mask2Former: post-process maps ``label_id`` (0..N-1) to dense labels 1..N then COCO ``category_id``."""
    import torch

    model.eval()
    detections: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            im_info = dataset.images[idx]
            image_id = int(im_info["id"])
            w0, h0 = int(im_info["width"]), int(im_info["height"])
            pv = sample["pixel_values"].unsqueeze(0).to(device)
            out = model(pixel_values=pv)
            processed = processor.post_process_instance_segmentation(
                out,
                threshold=score_thresh,
                target_sizes=[(h0, w0)],
                return_binary_maps=True,
            )[0]
            maps = processed.get("segmentation")
            segs = processed.get("segments_info") or []
            if maps is None or len(segs) == 0:
                continue
            for j, seg in enumerate(segs):
                if j >= maps.shape[0]:
                    break
                mo = (maps[j].float().cpu().numpy() > 0.5).astype(np.uint8)
                dense_lbl = int(seg["label_id"]) + 1
                if dense_lbl not in lbl2coco:
                    continue
                cid = lbl2coco[dense_lbl]
                detections.append(
                    detection_dict_for_coco(
                        image_id=image_id,
                        coco_category_id=cid,
                        mask_hw=mo,
                        score=float(seg["score"]),
                        orig_width=w0,
                        orig_height=h0,
                    )
                )
    return detections
