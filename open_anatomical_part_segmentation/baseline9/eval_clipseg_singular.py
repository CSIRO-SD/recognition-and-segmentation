#!/usr/bin/env python3
"""Baseline 11: Hugging Face CLIPSeg zero-shot segmentation on singular (balanced CSV) test split.

Loads ground truth from ``cleaned_singular_json`` LabelMe merged into COCO (same normalization
as ``labelingv3/export_sam3_coco.py``). For each test image and each GT category present on that
image, runs CLIPSeg with a text prompt (default: canonical category name) and writes mask
predictions for COCO mask AP/mIoU/Dice evaluation.

CLIPSeg usage follows the Transformers docs example for ``CLIPSegForImageSegmentation``:
processor with ``text`` + ``images``, forward pass, ``outputs.logits`` upsampled and thresholded.

Example::

    python -m baseline11.eval_clipseg_singular \\
        --device cuda \\
        --out-dir outputs/baseline11_clipseg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_PROJ = Path(__file__).resolve().parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from baseline5.coco_standalone_seg import (  # noqa: E402
    dump_coco_segm_predictions_json,
    evaluate_coco_segm_offline,
)
from baseline9.singular_coco_from_csv import (  # noqa: E402
    DEFAULT_SINGULAR_REL_DIRS,
    write_singular_gt_json,
)
from baseline_common.run_outputs import resolve_eval_output_dir  # noqa: E402
from baseline_common.sam3_coco_utils import (  # noqa: E402
    compute_mask_miou_dice,
    metrics_to_serializable,
    resolve_dataset_image_path,
)

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

DEFAULT_SPLIT_CSV = _PROJ / "labelingv3" / "data_split_balanced.csv"
DEFAULT_NORM_MAP = _PROJ / "labelingv3" / "label_normalization.json"
DEFAULT_DATASET_ROOT = Path("/datasets/work/icvg-phd-marsfield/work/all_crops")
DEFAULT_MODEL = "CIDAS/clipseg-rd64-refined"


def _image_ann_category_ids(coco: dict, image_ids: list[int] | None = None) -> dict[int, set[int]]:
    want = set(image_ids) if image_ids is not None else None
    out: dict[int, set[int]] = {}
    for ann in coco.get("annotations", []):
        iid = int(ann["image_id"])
        if want is not None and iid not in want:
            continue
        out.setdefault(iid, set()).add(int(ann["category_id"]))
    return out


def _mask_probs_to_bbox_xyxy(prob_hw: torch.Tensor, thr: float) -> tuple[float, float, float, float] | None:
    """prob_hw float [H,W] in original resolution; bbox in xyxy pixels."""
    m = prob_hw >= thr
    if m.ndim != 2:
        raise ValueError("expected mask [H,W]")
    ys, xs = torch.where(m)
    if ys.numel() == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return float(x0), float(y0), float(x1), float(y1)


def clipseg_predictions_for_balanced_split(
    *,
    coco_gt: dict,
    dataset_root: Path,
    model_name: str,
    device: str,
    prompt_template: str,
    threshold: float,
    score_threshold_min_area: float = 1e-6,
) -> dict[int, dict[str, torch.Tensor]]:
    """Run CLIPSeg for each unique GT category present on each image."""
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    processor = CLIPSegProcessor.from_pretrained(model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
    model.eval()

    id_to_file = {int(im["id"]): im["file_name"] for im in coco_gt["images"]}
    id_to_wh = {
        int(im["id"]): (int(im["height"]), int(im["width"])) for im in coco_gt["images"]
    }
    cid_to_name = {c["id"]: c["name"] for c in coco_gt["categories"]}
    img_cats = _image_ann_category_ids(coco_gt)

    preds: dict[int, dict[str, torch.Tensor]] = {}

    dtype = torch.float32
    items = sorted(img_cats.items(), key=lambda t: t[0])
    for image_id, cat_ids in tqdm(items, desc="CLIPSeg inference"):
        if not cat_ids:
            continue

        rel = id_to_file[image_id]
        img_path = resolve_dataset_image_path(dataset_root, rel)
        if img_path is None:
            logging.warning("Missing image image_id=%s path=%s", image_id, dataset_root / rel)
            continue

        image = Image.open(img_path).convert("RGB")
        h, w = image.size[1], image.size[0]
        eh, ew = id_to_wh[image_id]
        if (eh, ew) != (h, w):
            logging.warning(
                "Size mismatch image_id=%s coco=(%s,%s) file=(%s,%s); using file size",
                image_id,
                ew,
                eh,
                w,
                h,
            )

        labels_sorted = sorted(cat_ids)
        texts = [prompt_template.format(cid_to_name[c]) for c in labels_sorted]
        inputs = processor(
            text=texts,
            images=[image] * len(texts),
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        logits = outputs.logits  # [N, h', w']
        probs = logits.sigmoid().to(dtype=dtype)
        # Match original image height/width (PIL uses W,H)
        probs_up = F.interpolate(
            probs.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        boxes: list[torch.Tensor] = []
        masks_hw: list[torch.Tensor] = []
        scores: list[float] = []
        label_ids: list[int] = []

        for ti, cid in enumerate(labels_sorted):
            p_hw = probs_up[ti]
            box = _mask_probs_to_bbox_xyxy(p_hw, threshold)
            if box is None:
                continue
            binary = (p_hw >= threshold).float()
            if float(binary.sum()) < score_threshold_min_area:
                continue
            xyxy_t = torch.tensor(
                [[box[0], box[1], box[2], box[3]]], dtype=dtype, device="cpu"
            )
            inside = (
                float((p_hw[binary > 0.5]).mean().cpu()) if (binary > 0.5).any() else 0.0
            )
            boxes.append(xyxy_t)
            masks_hw.append(binary.cpu())
            scores.append(max(inside, float(p_hw.max().cpu())))
            label_ids.append(cid)

        if not scores:
            continue

        preds[image_id] = {
            "boxes": torch.cat(boxes, dim=0),
            "scores": torch.tensor(scores, dtype=dtype),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "masks": torch.stack(masks_hw, dim=0),
        }

    del model
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split-csv",
        type=Path,
        default=DEFAULT_SPLIT_CSV,
        help="CSV with json_path,split (default: labelingv3/data_split_balanced.csv).",
    )
    ap.add_argument(
        "--norm-map",
        type=Path,
        default=DEFAULT_NORM_MAP,
        help="label_normalization.json for canonical singular names.",
    )
    ap.add_argument(
        "--csv-split",
        type=str,
        default="test",
        help='Split bucket to evaluate ("test", "val", "training").',
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing ``images/…`` referenced by COCO ``file_name``.",
    )
    ap.add_argument(
        "--singular-subdirs",
        type=str,
        default=",".join(DEFAULT_SINGULAR_REL_DIRS),
        help=(
            "Comma-separated directory names directly under ``all_crops`` (sibling of "
            "``cleaned_json``) to search for singular LabelMe JSON—first match wins. "
            "Default tries ``cleaned_singular_json`` then ``noun_only_json`` "
            "(latter covers datasets where singular export is stored there)."
        ),
    )
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model id or local dir (CLIPSegForImageSegmentation).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid probability threshold on upsampled logits.",
    )
    ap.add_argument(
        "--prompt-template",
        type=str,
        default="{}",
        help='Python ``str.format`` template; ``{}`` is replaced by category name. '
        'Example: "a photo of a {}"',
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJ / "outputs" / "baseline11_clipseg",
        help="Run output directory (timestamped subfolder unless --flat).",
    )
    ap.add_argument("--run-id", type=str, default=None, help="Override timestamp run id.")
    ap.add_argument(
        "--flat",
        action="store_true",
        help="Write directly under --out-dir (embed run id in filenames).",
    )
    args = ap.parse_args()

    singular_subdirs = tuple(
        x.strip() for x in args.singular_subdirs.split(",") if x.strip()
    )
    if not singular_subdirs:
        print("error: --singular-subdirs yielded no directories", file=sys.stderr)
        return 2

    if "{}" not in args.prompt_template:
        print(
            "Warning: --prompt-template has no '{}'; all prompts will be identical.",
            file=sys.stderr,
        )

    run_dir, rid = resolve_eval_output_dir(args.out_dir, run_id=args.run_id, flat=args.flat)
    gt_path = run_dir / f"baseline11_singular_{args.csv_split}_gt_{rid}.json"
    pred_path = run_dir / f"baseline11_clipseg_{args.csv_split}_pred_{rid}.json"
    metrics_path = run_dir / f"baseline11_clipseg_{args.csv_split}_metrics_{rid}.json"
    summary_path = run_dir / f"baseline11_clipseg_{args.csv_split}_summary_{rid}.txt"

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Singular LabelMe lookup order (under all_crops): {list(singular_subdirs)}", flush=True)

    coco = write_singular_gt_json(
        args.split_csv,
        args.norm_map,
        args.csv_split,
        gt_path,
        singular_relative_dirs=singular_subdirs,
    )
    print(
        f"GT: images={len(coco['images'])} "
        f"annotations={len(coco['annotations'])} "
        f"categories={len(coco['categories'])}",
        flush=True,
    )

    preds = clipseg_predictions_for_balanced_split(
        coco_gt=coco,
        dataset_root=args.dataset_root,
        model_name=args.model,
        device=args.device,
        prompt_template=args.prompt_template,
        threshold=args.threshold,
    )
    dump_coco_segm_predictions_json(preds, pred_path)

    coco_metrics = evaluate_coco_segm_offline(gt_path, pred_path)
    mask_md = compute_mask_miou_dice(gt_path, pred_path)
    metrics = {**coco_metrics, **mask_md}
    serial = metrics_to_serializable(metrics)
    metrics_path.write_text(json.dumps(serial, indent=2) + "\n", encoding="utf-8")

    lines = [f"baseline11 CLIPSeg ({args.model}) split={args.csv_split} run_id={rid}\n"]
    for key, label in FOCUS_METRIC_LABELS:
        if key in serial:
            lines.append(f"  {label}: {serial[key]:.6f}\n")
    summary_path.write_text("".join(lines), encoding="utf-8")
    print("".join(lines), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
