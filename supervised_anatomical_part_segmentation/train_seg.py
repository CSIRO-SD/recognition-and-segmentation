#!/usr/bin/env python3
"""Unified COCO polygon segmentation training with **semantic-mask evaluation only**.

Two label schemes (``--label-set`` picks default JSONs under this package)::

  singular   ``annotations/singular_sam3_{train,test}.json`` — 11 part classes
  sam3       ``annotations/sam3_{train 1,test 1}.json``     — 31 classes

Resolve images as ``{--img-root}/{file_name}`` (e.g. ``all_crops/images/*.jpg``).

**All models** report the same test metrics on a **dense semantic label map**: **mIoU**, macro **Dice**, **pix_acc**, and **IoU_per_class** (COCO category names for the label set).

- **unet / deeplabv3plus / segformer**: trained with pixel-wise CE on rasterized GT; eval compares pred logits vs GT labels.
- **mask_rcnn / mask2former**: trained on instance masks; eval **paints** predicted instances into one map (higher score overwrites overlap) and compares to the same GT semantic raster.

**Mask2Former** (``train_mask2former``): optional **Detectron2-style** recipe — AdamW with **lower LR on the Swin backbone** (``pixel_level_module.encoder``), **linear warmup + cosine decay** (HuggingFace ``get_cosine_schedule_with_warmup``, step-per-batch), and **grad clipping**. See ``--m2f-*`` flags.

Example::

  python train_seg.py --model unet --label-set singular \\
    --img-root /datasets/work/icvg-phd-marsfield/work/all_crops
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    CocoInstanceTV513Dataset,
    CocoMask2FormerDataset,
    CocoSemanticSegDataset,
    collate_mask2former,
    collate_semantic,
)
from coco_eval_metrics import (  # noqa: E402
    run_mask2former_test_metrics,
    run_mask_rcnn_test_metrics,
    semantic_region_metrics,
)
from models_seg import (  # noqa: E402
    build_mask2former,
    build_mask_rcnn,
    build_semantic_model,
    semantic_forward,
)


def load_coco_json_categories(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["categories"]


def _resolve_ann_path(p: Path, repo: Path) -> Path:
    if p.is_absolute():
        return p
    return (repo / p).resolve()


def _amp_autocast(device: torch.device, amp_dtype: torch.dtype):
    """``torch.cuda.amp.autocast`` on CUDA; no-op elsewhere. No ``device_type`` (not in older PyTorch)."""
    if device.type != "cuda":
        return nullcontext()
    return torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype)


@torch.no_grad()
def eval_semantic_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    model_name: str,
    amp_dtype: torch.dtype,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    hist = torch.zeros(num_classes, num_classes, dtype=torch.double, device=device)
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        if "pixel_values" in batch:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
        else:
            x = batch["images"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
        with _amp_autocast(device, amp_dtype):
            if model_name.lower() == "segformer":
                logits = model(pixel_values=x).logits
            else:
                logits = semantic_forward(model, x, model_name)
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1)
        mask = (y >= 0) & (y < num_classes)
        p = pred[mask]
        t = y[mask]
        correct += int((p == t).sum().item())
        total += int(mask.sum().item())
        ind = num_classes * t.long() + p.long()
        hist += torch.bincount(ind, minlength=num_classes * num_classes).double().view(
            num_classes, num_classes
        )
    sem = semantic_region_metrics(
        hist, num_classes, skip_background=True, class_names=class_names
    )
    sem["pix_acc"] = float(correct / max(1, total))
    return sem


def _json_ready_metrics(d: dict) -> dict:
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _json_ready_metrics(v)
        elif v is None:
            out[k] = None
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, (float, int, str, bool)):
            out[k] = v
        else:
            out[k] = v
    return out


def train_semantic(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    repo = ROOT
    proc = None
    if args.model.lower() == "segformer":
        from transformers import SegformerImageProcessor

        proc = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")

    ds_tr = CocoSemanticSegDataset(
        _resolve_ann_path(Path(args.train_json), repo),
        args.img_root,
        args.size,
        segformer_processor=proc,
    )
    ds_te = CocoSemanticSegDataset(
        _resolve_ann_path(Path(args.test_json), repo),
        args.img_root,
        args.size,
        segformer_processor=proc,
    )
    ncls = ds_tr.num_labels_with_bg
    print(f"[semantic] foreground classes: {ds_tr.num_fg_classes}; with background: {ncls}")

    model = build_semantic_model(args.model, ncls, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
    scaler = GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_semantic,
        drop_last=False,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_semantic,
    )

    best_miou = 0.0
    last_sem: dict[str, Any] = {}
    for epoch in range(args.epochs):
        model.train()
        losses = []
        pbar = tqdm(dl_tr, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            if "pixel_values" in batch:
                x = batch["pixel_values"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
            else:
                x = batch["images"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with _amp_autocast(device, amp_dtype):
                if args.model.lower() == "segformer":
                    out = model(pixel_values=x, labels=y)
                    loss = out.loss
                else:
                    if args.model.lower() == "deeplabv3plus":
                        out = model(x)
                        logits = out["out"]
                        loss = ce(logits, y)
                        aux = out.get("aux")
                        if aux is not None:
                            aux = F.interpolate(aux, size=y.shape[-2:], mode="bilinear", align_corners=False)
                            loss = loss + 0.5 * ce(aux, y)
                    else:
                        logits = model(x)
                        loss = ce(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses[-20:])))

        sched.step()
        last_sem = eval_semantic_metrics(
            model, dl_te, device, ncls, args.model, amp_dtype, class_names=ds_te.cat_names
        )
        miou = last_sem["mIoU"]
        best_miou = max(best_miou, miou)
        print(
            f"Epoch {epoch+1}: train_loss={np.mean(losses):.4f}  "
            f"test mIoU={miou*100:.2f}%  Dice={last_sem['mean_Dice']*100:.2f}%  "
            f"pix_acc={last_sem['pix_acc']*100:.2f}%  "
            f"(best mIoU={best_miou*100:.2f}%)",
        )

    out = {
        "eval": "semantic_mask",
        "train_style": "pixel_semantic",
        "model": args.model,
        "label_set": args.label_set,
        "train_json": str(_resolve_ann_path(Path(args.train_json), repo)),
        "test_json": str(_resolve_ann_path(Path(args.test_json), repo)),
        **last_sem,
        "best_mIoU": best_miou,
        "num_classes_with_bg": ncls,
        "num_fg_classes": ds_tr.num_fg_classes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready_metrics(out), f, indent=2)
    torch.save(model.state_dict(), args.output_dir / "checkpoint_last.pth")
    print(f"Saved {args.output_dir / 'results.json'}")


def collate_detection(batch):
    return tuple(zip(*batch))


def train_mask_rcnn(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    repo = ROOT
    n_fg = len(load_coco_json_categories(_resolve_ann_path(Path(args.train_json), repo)))
    num_classes = n_fg + 1
    ds_tr = CocoInstanceTV513Dataset(_resolve_ann_path(Path(args.train_json), repo), args.img_root, args.size)
    ds_te = CocoInstanceTV513Dataset(_resolve_ann_path(Path(args.test_json), repo), args.img_root, args.size)

    model = build_mask_rcnn(num_classes, pretrained=True)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(3, args.epochs // 3), gamma=0.1)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=max(1, args.batch_size // 2),
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_detection,
        pin_memory=device.type == "cuda",
    )

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, targets in tqdm(dl_tr, desc=f"maskrcnn {epoch+1}/{args.epochs}", leave=False):
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        print(f"Epoch {epoch+1}: train_loss={np.mean(losses):.4f}")

    test_json_path = _resolve_ann_path(Path(args.test_json), repo)
    lbl2coco = {v: k for k, v in ds_te.id2lbl.items()}
    print(
        "[mask_rcnn] test: semantic mIoU / Dice / pix_acc "
        "(instances → painted label map)",
    )
    metrics = run_mask_rcnn_test_metrics(
        None,
        model,
        ds_te,
        device,
        lbl2coco,
        ds_te.id2lbl,
        num_classes=num_classes,
        score_thresh=0.05,
        coco_eval=False,
        class_names=ds_te.cat_names,
    )
    print(
        f"  mIoU={metrics['mIoU']*100:.2f}%  Dice={metrics['mean_Dice']*100:.2f}%  "
        f"pix_acc={metrics['pix_acc']*100:.2f}%",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "eval": "semantic_mask",
        "train_style": "instance_heads",
        "model": args.model,
        "label_set": args.label_set,
        "train_json": str(_resolve_ann_path(Path(args.train_json), repo)),
        "test_json": str(test_json_path),
        "eval_note": (
            "Mask R-CNN: trained on instance targets; test map paints instances by score (high overwrites low on overlap)."
        ),
        "num_fg_classes": n_fg,
        "num_classes_with_bg": num_classes,
        **metrics,
        "best_mIoU": metrics["mIoU"],
    }
    with open(args.output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready_metrics(out), f, indent=2)
    torch.save(model.state_dict(), args.output_dir / "checkpoint_last.pth")
    print(f"Saved {args.output_dir / 'results.json'}")


def _mask2former_adamw_groups(
    model: torch.nn.Module, lr: float, weight_decay: float, backbone_lr_scale: float
) -> tuple[list[dict[str, Any]], bool]:
    """Split HF Mask2Former params: Swin encoder vs rest (decoder / queries / heads).

    Official Mask2Former/Detectron2 recipes use a **smaller LR on the backbone**.
    """
    marker = "pixel_level_module.encoder"
    backbone: list[torch.nn.Parameter] = []
    rest: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if marker in name:
            backbone.append(p)
        else:
            rest.append(p)
    if not backbone:
        print(
            "[mask2former] no parameter names contain 'pixel_level_module.encoder'; "
            "using a single LR group (check transformers version).",
            file=sys.stderr,
        )
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr, "weight_decay": weight_decay}], False
    return [
        {"params": backbone, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay},
        {"params": rest, "lr": lr, "weight_decay": weight_decay},
    ], True


def train_mask2former(args: argparse.Namespace) -> None:
    from transformers import Mask2FormerImageProcessor
    from transformers.optimization import get_cosine_schedule_with_warmup

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    repo = ROOT
    n_fg = len(load_coco_json_categories(_resolve_ann_path(Path(args.train_json), repo)))
    num_labels = n_fg + 1
    proc = Mask2FormerImageProcessor.from_pretrained(args.mask2former_pretrained)
    ds_tr = CocoMask2FormerDataset(
        _resolve_ann_path(Path(args.train_json), repo),
        args.img_root,
        proc,
        args.size,
    )
    ds_te = CocoMask2FormerDataset(
        _resolve_ann_path(Path(args.test_json), repo),
        args.img_root,
        proc,
        args.size,
    )

    model = build_mask2former(
        num_labels, mid=args.mask2former_pretrained, class_names=ds_tr.cat_names
    ).to(device)
    param_groups, split_ok = _mask2former_adamw_groups(
        model, args.lr, args.weight_decay, args.m2f_backbone_lr_scale
    )
    opt = torch.optim.AdamW(param_groups)
    scaler = GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)

    eff_bs = max(1, args.batch_size)
    if args.m2f_train_batch_cap and eff_bs > args.m2f_train_batch_cap:
        eff_bs = args.m2f_train_batch_cap
    dl_tr = DataLoader(
        ds_tr,
        batch_size=eff_bs,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_mask2former,
        pin_memory=device.type == "cuda",
    )

    steps_per_epoch = max(1, len(dl_tr))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = 0
    sched = None
    if not args.m2f_no_lr_schedule and total_steps > 1:
        warmup_steps = min(total_steps - 1, max(0, int(total_steps * float(args.m2f_warmup_ratio))))
        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        print(
            f"[mask2former] LR schedule: warmup_steps={warmup_steps}, total_steps={total_steps}, "
            f"backbone_lr_scale={args.m2f_backbone_lr_scale}, max_grad_norm={args.m2f_max_grad_norm}",
            file=sys.stderr,
        )
    elif split_ok:
        print(
            f"[mask2former] param groups: backbone_lr={args.lr * args.m2f_backbone_lr_scale:g}, "
            f"rest_lr={args.lr:g} (no schedule)",
            file=sys.stderr,
        )

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in tqdm(dl_tr, desc=f"m2f {epoch+1}/{args.epochs}", leave=False):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            ml = [m.to(device) for m in batch["mask_labels"]]
            cl = [c.to(device) for c in batch["class_labels"]]
            opt.zero_grad(set_to_none=True)
            with _amp_autocast(device, amp_dtype):
                out = model(pixel_values=pv, mask_labels=ml, class_labels=cl)
                loss = out.loss
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.m2f_max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.m2f_max_grad_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.m2f_max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.m2f_max_grad_norm)
                opt.step()
            if sched is not None:
                sched.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}: train_loss={np.mean(losses):.4f}")

    test_json_path = _resolve_ann_path(Path(args.test_json), repo)
    lbl2coco = {v: k for k, v in ds_te.id2lbl.items()}
    print(
        "[mask2former] test: semantic mIoU / Dice / pix_acc "
        "(instances → painted label map)",
    )
    metrics = run_mask2former_test_metrics(
        None,
        model,
        proc,
        ds_te,
        device,
        lbl2coco,
        ds_te.id2lbl,
        num_classes=num_labels,
        score_thresh=float(args.m2f_score_thresh) if args.m2f_score_thresh is not None else 0.35,
        coco_eval=False,
        class_names=ds_te.cat_names,
    )
    print(
        f"  mIoU={metrics['mIoU']*100:.2f}%  Dice={metrics['mean_Dice']*100:.2f}%  "
        f"pix_acc={metrics['pix_acc']*100:.2f}%",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "checkpoint_last.pth")
    out = {
        "eval": "semantic_mask",
        "train_style": "instance_heads",
        "model": args.model,
        "label_set": args.label_set,
        "train_json": str(_resolve_ann_path(Path(args.train_json), repo)),
        "test_json": str(test_json_path),
        "mask2former_pretrained": args.mask2former_pretrained,
        "m2f_recipe": {
            "backbone_lr_scale": float(args.m2f_backbone_lr_scale),
            "warmup_ratio": float(args.m2f_warmup_ratio),
            "warmup_steps": int(warmup_steps),
            "optimizer_steps_total": int(total_steps),
            "lr_scheduled": not bool(args.m2f_no_lr_schedule),
            "max_grad_norm": float(args.m2f_max_grad_norm),
            "train_batch_effective": int(eff_bs),
            "paint_score_thresh": float(args.m2f_score_thresh)
            if args.m2f_score_thresh is not None
            else 0.35,
        },
        "train_final_loss": float(np.mean(losses)) if losses else 0.0,
        "eval_note": (
            "Mask2Former: trained on instance targets; test map paints instances by score (high overwrites low on overlap)."
        ),
        "num_fg_classes": n_fg,
        "num_classes_with_bg": num_labels,
        **metrics,
        "best_mIoU": metrics["mIoU"],
    }
    with open(args.output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready_metrics(out), f, indent=2)
    print(f"Saved {args.output_dir / 'results.json'}")


def print_plan(args: argparse.Namespace) -> None:
    print("\n" + "=" * 70)
    print("Run configuration (H100: raise batch if memory allows)")
    print("=" * 70)
    for k, v in vars(args).items():
        if k.startswith("_"):
            continue
        print(f"  {k}: {v}")
    print("=" * 70 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        choices=("unet", "deeplabv3plus", "segformer", "mask_rcnn", "mask2former"),
        required=True,
    )
    p.add_argument("--label-set", choices=("singular", "sam3"), default="singular")
    p.add_argument("--img-root", type=Path, required=True)
    p.add_argument("--train-json", type=Path, default=None)
    p.add_argument("--test-json", type=Path, default=None)
    p.add_argument("--val-json", type=Path, default=None, help="Optional; not used yet (test-json for eval).")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mask2former-pretrained", default="facebook/mask2former-swin-small-coco-instance")
    p.add_argument(
        "--m2f-backbone-lr-scale",
        type=float,
        default=0.1,
        help="Mask2Former: LR multiplier for Swin backbone (pixel_level_module.encoder); Detectron2-style ~0.1.",
    )
    p.add_argument(
        "--m2f-warmup-ratio",
        type=float,
        default=0.05,
        help="Mask2Former: linear warmup as a fraction of total optimizer steps, then cosine to 0.",
    )
    p.add_argument(
        "--m2f-max-grad-norm",
        type=float,
        default=0.01,
        help="Mask2Former: clip global grad norm (0 = disabled).",
    )
    p.add_argument(
        "--m2f-no-lr-schedule",
        action="store_true",
        help="Mask2Former: constant LR (still uses backbone vs head LR groups unless backbone scale=1).",
    )
    p.add_argument(
        "--m2f-train-batch-cap",
        type=int,
        default=0,
        metavar="N",
        help="Mask2Former: if >0, cap training batch at N. 0 = use --batch-size as-is.",
    )
    p.add_argument(
        "--m2f-score-thresh",
        type=float,
        default=None,
        help="Mask2Former test painting: instance score cutoff (default 0.35). Try 0.05–0.35 to match Mask R-CNN sensitivity.",
    )
    args = p.parse_args()

    repo = ROOT
    if args.label_set == "singular":
        default_train = "annotations/singular_sam3_train.json"
        default_test = "annotations/singular_sam3_test.json"
    else:
        default_train = "annotations/sam3_train 1.json"
        default_test = "annotations/sam3_test 1.json"
    if args.train_json is None:
        args.train_json = Path(default_train)
    if args.test_json is None:
        args.test_json = Path(default_test)

    if args.output_dir is None:
        args.output_dir = repo / "runs" / f"{args.label_set}_{args.model}"

    wd_def = 0.05
    if args.model in ("mask_rcnn",):
        wd_def = 5e-4
    if args.weight_decay is None:
        args.weight_decay = wd_def
    lr_def = 3e-4
    if args.model == "mask_rcnn":
        lr_def = 0.005
    if args.model == "mask2former":
        lr_def = 1e-4
    if args.model == "segformer":
        lr_def = 6e-5
    if args.lr is None:
        args.lr = lr_def

    print_plan(args)

    if args.model in ("unet", "deeplabv3plus", "segformer"):
        return train_semantic(args)
    if args.model == "mask_rcnn":
        return train_mask_rcnn(args)
    if args.model == "mask2former":
        return train_mask2former(args)
    p.error("Unreachable")


if __name__ == "__main__":
    main()
