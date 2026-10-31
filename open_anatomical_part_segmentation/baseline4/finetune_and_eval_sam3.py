#!/usr/bin/env python3
"""Experiment 2: fine-tune SAM 3 (300 epochs in default config), then evaluate on test JSON.

Copies ``baseline2/configs/proj06_insect_finetune.yaml`` into the SAM 3 repo config
directory, runs ``sam3/train/train.py``, then runs the same COCO mask-AP evaluation as
``baseline1/eval_pretrained_sam3_test.py`` using the saved ``checkpoint.pt``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

_PROJ = Path(__file__).resolve().parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from baseline_common.run_outputs import resolve_eval_output_dir  # noqa: E402
from baseline_common.sam3_coco_utils import (  # noqa: E402
    compute_mask_miou_dice,
    dump_coco_segmentation_predictions_json,
    evaluate_coco_offline,
    load_coco_gt,
    metrics_to_serializable,
    resolve_sam3_bpe_path,
    run_sam3_image_inference_multiprocess,
    run_sam3_image_inference_to_coco_predictions,
)


def default_sam3_root() -> Path:
    return _PROJ / "sam3"


def install_hydra_config(sam3_root: Path) -> Path:
    src = _PROJ / "baseline3" / "configs" / "proj06_insect_finetune.yaml"
    if not src.is_file():
        raise FileNotFoundError(f"Missing Hydra config: {src}")
    dst_dir = sam3_root / "sam3" / "train" / "configs"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "proj06_insect_finetune.yaml"
    shutil.copy2(src, dst)
    return dst


def run_training(
    sam3_root: Path,
    num_gpus: int,
    *,
    torch_nccl_heartbeat_timeout_sec: int = 3600,
    disable_torch_nccl_heartbeat_monitor: bool = False,
) -> None:
    train_py = sam3_root / "sam3" / "train" / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(
            f"Missing {train_py}. Clone facebookresearch/sam3 and pip install -e ."
        )
    cmd = [
        sys.executable,
        str(train_py),
        "-c",
        "configs/proj06_insect_finetune.yaml",
        "--use-cluster",
        "0",
        "--num-gpus",
        str(num_gpus),
    ]
    env = os.environ.copy()
    # Default PyTorch NCCL watchdog is 480s; long steps or slow I/O can trigger false aborts.
    env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = str(torch_nccl_heartbeat_timeout_sec)
    if disable_torch_nccl_heartbeat_monitor:
        env["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    print("Running:", " ".join(cmd), flush=True)
    print(
        f"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC={env['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC']}",
        end="",
        flush=True,
    )
    if disable_torch_nccl_heartbeat_monitor:
        print(", TORCH_NCCL_ENABLE_MONITORING=0", end="", flush=True)
    print(flush=True)
    subprocess.run(cmd, cwd=str(sam3_root), check=True, env=env)


def find_checkpoint(experiment_log_dir: Path) -> Path:
    ckpt = experiment_log_dir / "checkpoints" / "checkpoint.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}. Training may have failed or skip_saving_ckpts was set."
        )
    return ckpt


def run_test_eval(
    *,
    checkpoint: Path,
    gt_json: Path,
    dataset_root: Path,
    out_dir: Path,
    device: str,
    confidence_threshold: float,
    bpe_path_opt: Path | None = None,
    eval_num_gpus: int | None = None,
    file_tag: str | None = None,
) -> None:
    """If ``file_tag`` is set, append ``_<file_tag>`` to output basenames (flat layout)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{file_tag}" if file_tag else ""
    pred_path = out_dir / f"exp2_finetune300_sam3_test_predictions{tag}.json"
    metrics_path = out_dir / f"exp2_finetune300_sam3_test_metrics{tag}.json"
    metrics_txt = out_dir / f"exp2_finetune300_sam3_test_metrics{tag}.txt"

    bpe_path = resolve_sam3_bpe_path(preferred=bpe_path_opt)

    infer_n = 1
    if device == "cuda" and torch.cuda.is_available():
        dc = torch.cuda.device_count()
        n = eval_num_gpus if eval_num_gpus is not None else dc
        infer_n = max(1, min(n, dc))
    else:
        infer_n = 1

    if infer_n <= 1:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            device=device,
            eval_mode=True,
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            enable_segmentation=True,
        )
        processor = Sam3Processor(model, device=device)

        coco_gt = load_coco_gt(gt_json)
        preds = run_sam3_image_inference_to_coco_predictions(
            coco_gt=coco_gt,
            dataset_root=dataset_root,
            processor=processor,
            confidence_threshold=confidence_threshold,
            empty_cuda_cache_each_image=True,
        )
    else:
        print(
            f"Multi-GPU test inference: {infer_n} workers (one model per GPU)",
            flush=True,
        )
        preds = run_sam3_image_inference_multiprocess(
            gt_json=gt_json,
            dataset_root=dataset_root,
            bpe_path=bpe_path,
            checkpoint_path=checkpoint,
            load_from_hf=False,
            confidence_threshold=confidence_threshold,
            num_workers=infer_n,
            empty_cuda_cache_each_image=True,
        )
    dump_coco_segmentation_predictions_json(preds, pred_path)

    metrics = evaluate_coco_offline(gt_json, pred_path, iou_type="segm", tide=False)
    metrics.update(compute_mask_miou_dice(gt_json, pred_path))
    metrics_path.write_text(
        json.dumps(metrics_to_serializable(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [f"{k}: {v}" for k, v in sorted(metrics.items())]
    metrics_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if file_tag:
        print(f"Eval file tag: {file_tag}")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {metrics_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sam3-root",
        type=Path,
        default=default_sam3_root(),
        help="Root of the cloned facebookresearch/sam3 repository.",
    )
    ap.add_argument(
        "--experiment-log-dir",
        type=Path,
        default=_PROJ / "outputs" / "sam3_finetune_300ep",
        help="Must match paths.experiment_log_dir in the Hydra YAML (checkpoint dir).",
    )
    ap.add_argument(
        "--gt-json",
        type=Path,
        default=_PROJ / "labelingv2" / "sam3_test.json",
        help="Held-out test COCO JSON.",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/datasets/work/icvg-phd-marsfield/work/all_crops"),
        help="Image root (contains ``images/``).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJ / "outputs",
        help="Base directory for experiment 2 test predictions and metrics.",
    )
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        metavar="ID",
        help="Eval run folder / filename tag (default: timestamp YYYYMMDD_HHMMSS).",
    )
    ap.add_argument(
        "--out-flat",
        action="store_true",
        help="Write into --out-dir with timestamp in filenames instead of a timestamp subfolder.",
    )
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Only run test evaluation using an existing checkpoint.pt.",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint for --skip-train (default: experiment_log_dir/checkpoints/checkpoint.pt).",
    )
    ap.add_argument(
        "--bpe-path",
        type=Path,
        default=None,
        help="Path to bpe_simple_vocab_16e6.txt.gz (same as baseline1 --bpe-path).",
    )
    ap.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Training: GPUs for sam3/train/train.py (ignored with --skip-train).",
    )
    ap.add_argument(
        "--torch-nccl-heartbeat-timeout",
        type=int,
        default=3600,
        metavar="SEC",
        help=(
            "Sets TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC for the training subprocess "
            "(PyTorch default 480s often aborts multi-GPU jobs on slow steps; ignored with --skip-train)."
        ),
    )
    ap.add_argument(
        "--disable-torch-nccl-heartbeat-monitor",
        action="store_true",
        help=(
            "Sets TORCH_NCCL_ENABLE_MONITORING=0 for training (avoids watchdog aborts; "
            "can hide real deadlocks — prefer raising --torch-nccl-heartbeat-timeout first)."
        ),
    )
    ap.add_argument(
        "--eval-num-gpus",
        type=int,
        default=None,
        metavar="N",
        help="Test inference after training: parallel SAM3 workers (default: all CUDA devices).",
    )
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = ap.parse_args()

    if not args.gt_json.is_file():
        print(f"Missing: {args.gt_json}", file=sys.stderr)
        return 1
    if not args.dataset_root.is_dir():
        print(f"Missing dataset root: {args.dataset_root}", file=sys.stderr)
        return 1

    install_hydra_config(args.sam3_root)

    if not args.skip_train:
        run_training(
            args.sam3_root,
            args.num_gpus,
            torch_nccl_heartbeat_timeout_sec=args.torch_nccl_heartbeat_timeout,
            disable_torch_nccl_heartbeat_monitor=args.disable_torch_nccl_heartbeat_monitor,
        )

    ckpt = args.checkpoint or find_checkpoint(args.experiment_log_dir)
    if not ckpt.is_file():
        print(f"Missing checkpoint: {ckpt}", file=sys.stderr)
        return 1

    eff_out, run_id = resolve_eval_output_dir(
        args.out_dir, run_id=args.run_id, flat=args.out_flat
    )
    print(f"Eval run ID: {run_id}", flush=True)

    run_test_eval(
        checkpoint=ckpt,
        gt_json=args.gt_json,
        dataset_root=args.dataset_root,
        out_dir=eff_out,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        bpe_path_opt=args.bpe_path,
        eval_num_gpus=args.eval_num_gpus,
        file_tag=run_id if args.out_flat else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
