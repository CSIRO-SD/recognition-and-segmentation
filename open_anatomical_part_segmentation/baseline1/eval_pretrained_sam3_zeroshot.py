#!/usr/bin/env python3
"""Baseline 2: zero-shot pretrained SAM 3 — GT class names as text prompts; COCO mask AP, ``mask_miou``, ``mask_dice``.

Inference uses a single process on ``--device`` (one GPU when ``device=cuda``); multi-GPU parallel inference is not supported here.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gt-json",
        type=Path,
        default=_PROJ / "labelingv2" / "sam3_test.json",
        help="COCO JSON (GT annotations); use sam3_train.json / sam3_val.json / sam3_test.json.",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/datasets/work/icvg-phd-marsfield/work/all_crops"),
        help="Root containing ``images/`` as in COCO ``file_name`` paths.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJ / "outputs",
        help="Base directory for predictions and metrics.",
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
    args = ap.parse_args()

    if not args.gt_json.is_file():
        print(f"Missing GT JSON: {args.gt_json}", file=sys.stderr)
        return 1
    if not args.dataset_root.is_dir():
        print(f"Missing dataset root: {args.dataset_root}", file=sys.stderr)
        return 1

    eff_out, run_id = resolve_eval_output_dir(
        args.out_dir, run_id=args.run_id, flat=args.out_flat
    )
    if args.out_flat:
        pred_path = eff_out / f"exp2_zeroshot_sam3_predictions_{run_id}.json"
        metrics_path = eff_out / f"exp2_zeroshot_sam3_metrics_{run_id}.json"
        metrics_txt = eff_out / f"exp2_zeroshot_sam3_metrics_{run_id}.txt"
    else:
        pred_path = eff_out / "exp2_zeroshot_sam3_predictions.json"
        metrics_path = eff_out / "exp2_zeroshot_sam3_metrics.json"
        metrics_txt = eff_out / "exp2_zeroshot_sam3_metrics.txt"

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

    print(f"Run ID: {run_id}")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote metrics (txt): {metrics_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
