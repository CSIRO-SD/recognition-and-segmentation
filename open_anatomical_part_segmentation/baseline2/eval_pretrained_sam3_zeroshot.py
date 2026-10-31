#!/usr/bin/env python3
"""Baseline 5: SAM 3 zero-shot on singular splits — GT **class names** as text prompts.

Uses the same inference and COCO offline evaluation as ``baseline2/eval_pretrained_sam3_zeroshot.py``,
adapted for ``singular_sam3_{train,val,test}.json`` and **category name** prompts (not ``noun_phrase``).

The public Hugging Face ``facebook/sam3`` checkpoint pairs with a ViTDet-style trunk
(``embed_dim=1024``, ``depth=32``); see ``baseline_common.sam3_vision_backbone`` for how this
relates to the classical ImageNet ViT-B/16 naming.

Metrics: COCO mask AP/AR (``coco_eval_segm_*``), plus ``mask_miou`` and ``mask_dice``.

Inference is single-process on ``--device`` (one GPU when ``device=cuda``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJ = Path(__file__).resolve().parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from baseline_common.run_outputs import resolve_eval_output_dir  # noqa: E402
from baseline_common.sam3_vision_backbone import (  # noqa: E402
    assert_trunk_matches_hf_sam3_image_checkpoint,
    describe_sam3_vision_trunk_vs_vit_b,
    vision_trunk_embed_depth,
)
from baseline_common.sam3_coco_utils import (  # noqa: E402
    compute_mask_miou_dice,
    dump_coco_segmentation_predictions_json,
    evaluate_coco_offline,
    load_coco_gt,
    metrics_to_serializable,
    resolve_sam3_bpe_path,
    run_sam3_image_inference_to_coco_predictions,
)

# Subset emphasized in baseline5/task.md (all are produced by ``evaluate_coco_offline`` for segm).
PRIMARY_COCO_SEGM_KEYS = (
    "coco_eval_segm_AP",
    "coco_eval_segm_AP_50",
    "coco_eval_segm_AP_75",
    "coco_eval_segm_AP_small",
    "coco_eval_segm_AP_medium",
    "coco_eval_segm_AP_large",
    "coco_eval_segm_AR_maxDets@1",
    "coco_eval_segm_AR_maxDets@10",
    "coco_eval_segm_AR_maxDets@100",
    "coco_eval_segm_AR_small",
    "coco_eval_segm_AR_medium",
    "coco_eval_segm_AR_large",
)
PRIMARY_MASK_KEYS = ("mask_dice", "mask_miou")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gt-json",
        type=Path,
        default=_PROJ / "labelingv3" / "singular_sam3_test.json",
        help="COCO JSON (GT). Defaults to singular_sam3_test.json; use singular_sam3_train.json "
        "or singular_sam3_val.json for other splits.",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/datasets/work/icvg-phd-marsfield/work/all_crops"),
        help="Root containing image paths as in COCO ``file_name`` (see task.md: all_crops).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJ / "outputs" / "baseline5_sam3_zeroshot",
        help="Base directory; default creates a timestamped subfolder (see --out-flat).",
    )
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        metavar="ID",
        help="Run folder / filename tag (default: timestamp YYYYMMDD_HHMMSS).",
    )
    ap.add_argument(
        "--out-flat",
        action="store_true",
        help="Write into --out-dir with timestamp in filenames instead of a timestamp subfolder.",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional SAM3 checkpoint (.pt). Default: download from Hugging Face.",
    )
    ap.add_argument(
        "--bpe-path",
        type=Path,
        default=None,
        help="Path to bpe_simple_vocab_16e6.txt.gz (default: resolve from installed sam3 / repo).",
    )
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu (single-process inference on this device only).",
    )
    ap.add_argument(
        "--skip-trunk-check",
        action="store_true",
        help="Do not verify vision trunk dims against the public HF sam3.pt architecture.",
    )
    ap.add_argument(
        "--use-noun-phrase",
        action="store_true",
        help="Use per-annotation ``noun_phrase`` when present (else category name). "
        "Default is category name only, per task.md.",
    )
    args = ap.parse_args()

    if not args.gt_json.is_file():
        print(f"Missing GT JSON: {args.gt_json}", file=sys.stderr)
        return 1
    if not args.dataset_root.is_dir():
        print(f"Missing dataset root: {args.dataset_root}", file=sys.stderr)
        return 1

    category_names_only = not args.use_noun_phrase

    eff_out, run_id = resolve_eval_output_dir(
        args.out_dir, run_id=args.run_id, flat=args.out_flat
    )
    if args.out_flat:
        pred_path = eff_out / f"baseline5_zeroshot_sam3_predictions_{run_id}.json"
        metrics_path = eff_out / f"baseline5_zeroshot_sam3_metrics_{run_id}.json"
        metrics_txt = eff_out / f"baseline5_zeroshot_sam3_metrics_{run_id}.txt"
        metrics_summary_path = eff_out / f"baseline5_zeroshot_sam3_metrics_summary_{run_id}.txt"
    else:
        pred_path = eff_out / "baseline5_zeroshot_sam3_predictions.json"
        metrics_path = eff_out / "baseline5_zeroshot_sam3_metrics.json"
        metrics_txt = eff_out / "baseline5_zeroshot_sam3_metrics.txt"
        metrics_summary_path = eff_out / "baseline5_zeroshot_sam3_metrics_summary.txt"

    bpe_path = resolve_sam3_bpe_path(preferred=args.bpe_path)

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=args.device,
        eval_mode=True,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        load_from_HF=args.checkpoint is None,
        enable_segmentation=True,
    )
    if not args.skip_trunk_check:
        assert_trunk_matches_hf_sam3_image_checkpoint(model)
    ed, dep = vision_trunk_embed_depth(model)
    print(
        f"SAM3 vision trunk: ViTDet embed_dim={ed}, depth={dep}. "
        + describe_sam3_vision_trunk_vs_vit_b(),
        flush=True,
    )
    processor = Sam3Processor(model, device=args.device)

    coco_gt = load_coco_gt(args.gt_json)
    preds = run_sam3_image_inference_to_coco_predictions(
        coco_gt=coco_gt,
        dataset_root=args.dataset_root,
        processor=processor,
        confidence_threshold=args.confidence_threshold,
        empty_cuda_cache_each_image=True,
        category_names_only=category_names_only,
    )
    dump_coco_segmentation_predictions_json(preds, pred_path)

    metrics = evaluate_coco_offline(
        args.gt_json,
        pred_path,
        iou_type="segm",
        tide=False,
    )
    metrics.update(compute_mask_miou_dice(args.gt_json, pred_path))
    metrics_path.write_text(
        json.dumps(metrics_to_serializable(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [f"{k}: {v}" for k, v in sorted(metrics.items())]
    metrics_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines: list[str] = []
    for k in PRIMARY_COCO_SEGM_KEYS + PRIMARY_MASK_KEYS:
        if k in metrics:
            summary_lines.append(f"{k}: {metrics[k]}")
    metrics_summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Run ID: {run_id}")
    print(f"Prompts: {'category names only' if category_names_only else 'noun_phrase when set'}")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote metrics (txt): {metrics_txt}")
    print(f"Wrote metrics (task summary): {metrics_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
