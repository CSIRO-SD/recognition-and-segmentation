#!/usr/bin/env python3
"""Supervised classifier on specimen benchmark CSV: label_text as class, train / test_seen splits.

Supports timm backbones: ResNet50, ConvNeXt-T, ViT-B/16, DINOv2 ViT-B (patch14 in timm), DINOv3 ViT-L/16,
with full fine-tune or frozen backbone + linear head (DINO recipes).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Recipe registry: timm id, mode, and recommended hyperparameters for H100 @ ~224–518
# ---------------------------------------------------------------------------

RECIPE_META: dict[str, dict] = {
    "resnet50": {
        "timm_name": "resnet50",
        "freeze_backbone": False,
        "batch_size": 128,
        "epochs": 45,
        "lr": 5e-4,
        "weight_decay": 0.05,
        "warmup_epochs": 4,
        "notes": "Standard ImageNet ResNet-50 classification head.",
        "rough_h100_sec_per_epoch": 18.0,
    },
    "convnext_tiny": {
        "timm_name": "convnext_tiny",
        "freeze_backbone": False,
        "batch_size": 128,
        "epochs": 45,
        "lr": 5e-4,
        "weight_decay": 0.05,
        "warmup_epochs": 4,
        "notes": "ConvNeXt-Tiny.",
        "rough_h100_sec_per_epoch": 22.0,
    },
    "vit_b16": {
        "timm_name": "vit_base_patch16_224",
        "freeze_backbone": False,
        "batch_size": 64,
        "epochs": 50,
        "lr": 4e-4,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "notes": "ViT-B/16 224 (ImageNet-21k/ViT style, not DINO).",
        "rough_h100_sec_per_epoch": 35.0,
    },
    "dinov2_vitb_linear": {
        "timm_name": "vit_base_patch14_dinov2",
        "freeze_backbone": True,
        "batch_size": 128,
        "epochs": 30,
        "lr": 8e-3,
        "weight_decay": 0.01,
        "warmup_epochs": 3,
        "notes": "DINOv2 ViT-B in timm is patch14 (official); backbone frozen, train linear head only.",
        "rough_h100_sec_per_epoch": 28.0,
    },
    "dinov2_vitb_finetune": {
        "timm_name": "vit_base_patch14_dinov2",
        "freeze_backbone": False,
        "batch_size": 64,
        "epochs": 35,
        "lr": 2e-4,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "notes": "DINOv2 ViT-B patch14 full fine-tune (all layers).",
        "rough_h100_sec_per_epoch": 45.0,
    },
    "dinov3_vitl_linear": {
        "timm_name": "vit_large_patch16_dinov3",
        "freeze_backbone": True,
        "batch_size": 64,
        "epochs": 30,
        "lr": 1e-2,
        "weight_decay": 0.01,
        "warmup_epochs": 3,
        "notes": "DINOv3 ViT-L/16 frozen + linear probe.",
        "rough_h100_sec_per_epoch": 55.0,
    },
    "dinov3_vitl_finetune": {
        "timm_name": "vit_large_patch16_dinov3",
        "freeze_backbone": False,
        "batch_size": 32,
        "epochs": 35,
        "lr": 8e-5,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "notes": "DINOv3 ViT-L/16 full fine-tune; lower LR for stability.",
        "rough_h100_sec_per_epoch": 120.0,
    },
}


def print_training_plan(
    recipe: str,
    meta: dict,
    n_train: int,
    n_test: int,
    num_classes: int,
    device_name: str,
) -> None:
    bs = meta["batch_size"]
    ep = meta["epochs"]
    steps_per_epoch = max(1, math.ceil(n_train / bs))
    rough_ep = meta.get("rough_h100_sec_per_epoch", 30.0)
    est_total_min = rough_ep * ep / 60.0
    print("\n" + "=" * 72)
    print(f"Training plan  |  recipe={recipe}  |  device={device_name}")
    print("=" * 72)
    print(f"  timm model      : {meta['timm_name']}")
    print(f"  freeze_backbone : {meta['freeze_backbone']}")
    print(f"  num_classes     : {num_classes}")
    print(f"  train samples   : {n_train}  |  test_seen : {n_test}")
    print(f"  batch_size      : {bs}  |  steps/epoch ≈ {steps_per_epoch}")
    print(f"  epochs          : {ep}")
    print(f"  lr              : {meta['lr']}  |  weight_decay : {meta['weight_decay']}")
    print(f"  warmup_epochs   : {meta['warmup_epochs']}")
    print(f"  notes           : {meta['notes']}")
    print("-" * 72)
    print(
        f"  **Rough H100 wall-clock estimate** (order-of-magnitude): ~{rough_ep:.0f} s/epoch → "
        f"~{est_total_min:.0f} min total ({ep} epochs).",
    )
    print("  (Run --benchmark-iters > 0 after this for a short live timing on your machine.)")
    print("=" * 72 + "\n")


def run_short_benchmark(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    amp_dtype: torch.dtype,
    iters: int,
) -> float | None:
    """Returns average seconds per training step (forward+backward+step)."""
    if iters <= 0:
        return None
    model.train()
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    steps = 0
    t0 = time.perf_counter()
    it = iter(loader)
    for _ in range(iters):
        try:
            imgs, labels = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, labels = next(it)
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(imgs)
                loss = criterion(logits, labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt / max(1, steps)


class CsvSpecimenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_root: Path, label_to_idx: dict[str, int], transform):
        self.paths = [images_root / str(r) for r in df["image_name"].tolist()]
        self.labels = [label_to_idx[str(l).strip()] for l in df["label_text"].tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        im = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, self.labels[i]


class BackboneClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, freeze_backbone: bool):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        feat = self.backbone.num_features
        self.head = nn.Linear(feat, num_classes)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z)


def build_model(recipe: str, num_classes: int, meta: dict) -> nn.Module:
    name = meta["timm_name"]
    freeze = meta["freeze_backbone"]
    if freeze:
        backbone = timm.create_model(name, pretrained=True, num_classes=0)
        return BackboneClassifier(backbone, num_classes, freeze_backbone=True)
    return timm.create_model(name, pretrained=True, num_classes=num_classes)


def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...]) -> list[float]:
    """Top-k accuracy in [0,1] per k."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:k].reshape(-1).float().sum().item() / target.size(0))
    return res


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    model.eval()
    n = 0
    top1 = top5 = 0.0
    use_amp = device.type == "cuda"
    for imgs, labels in tqdm(loader, desc="test_seen", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(imgs)
        else:
            logits = model(imgs)
        b = labels.size(0)
        a1, a5 = accuracy_topk(logits, labels, (1, 5))
        top1 += a1 * b
        top5 += a5 * b
        n += b
    return top1 / max(1, n), top5 / max(1, n)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scheduler: CosineLRScheduler,
    device: torch.device,
    epoch: int,
    amp_dtype: torch.dtype,
) -> float:
    model.train()
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    running = 0.0
    pbar = tqdm(loader, desc=f"train e{epoch+1}", leave=False)
    for step, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(imgs)
                loss = criterion(logits, labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step_update(epoch * len(loader) + step)
        running += loss.item()
        pbar.set_postfix(loss=f"{running / (step + 1):.4f}")
    return running / max(1, len(loader))


def print_all_recipe_cards(n_train: int = 27_082, n_test: int = 7_759, num_classes: int = 517) -> None:
    print("\nAll recipes (rough H100 timing is heuristic; use --benchmark-iters for a live estimate).\n")
    for recipe in sorted(RECIPE_META.keys()):
        print_training_plan(recipe, RECIPE_META[recipe], n_train, n_test, num_classes, "cuda:0 (example)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--images-root", type=Path, default=None)
    p.add_argument(
        "--recipe",
        choices=sorted(RECIPE_META.keys()),
        default=None,
        help="Model recipe (see RECIPE_META in script).",
    )
    p.add_argument(
        "--print-all-recipes",
        action="store_true",
        help="Print training plans for every recipe (uses typical benchmark n_train/n_test/class counts) and exit.",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--benchmark-iters",
        type=int,
        default=40,
        help="Run this many train steps before epoch 1 to estimate sec/step (0 to skip).",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.print_all_recipes:
        print_all_recipe_cards()
        return

    if args.csv is None or args.images_root is None or args.recipe is None:
        p.error("--csv, --images-root, and --recipe are required (unless using --print-all-recipes).")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    meta = dict(RECIPE_META[args.recipe])
    if args.epochs is not None:
        meta["epochs"] = args.epochs
    if args.batch_size is not None:
        meta["batch_size"] = args.batch_size
    if args.lr is not None:
        meta["lr"] = args.lr
    if args.weight_decay is not None:
        meta["weight_decay"] = args.weight_decay

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["label_text", "image_name", "split"])
    df["label_text"] = df["label_text"].astype(str).str.strip()
    df = df[df["label_text"] != ""]

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test_seen"].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        print("Need non-empty train (split=train) and test_seen.", file=sys.stderr)
        sys.exit(1)

    train_labels = sorted(train_df["label_text"].unique())
    label_to_idx = {l: i for i, l in enumerate(train_labels)}
    num_classes = len(train_labels)

    unseen = set(test_df["label_text"].unique()) - set(train_labels)
    if unseen:
        print(f"Warning: {len(unseen)} labels appear in test_seen but not train; dropping those rows.", file=sys.stderr)
        test_df = test_df[test_df["label_text"].isin(train_labels)]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if device.type == "cuda" and amp_dtype == torch.float32:
        amp_dtype = torch.float16

    out_dir = args.output_dir or Path("checkpoints") / f"benchmark_{args.recipe}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "label_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2, ensure_ascii=False)

    print_training_plan(
        args.recipe,
        meta,
        len(train_df),
        len(test_df),
        num_classes,
        str(device),
    )

    # Transforms from timm (per architecture input size & normalization)
    data_stub = timm.create_model(meta["timm_name"], pretrained=False, num_classes=0)
    data_cfg = resolve_model_data_config(data_stub)
    del data_stub
    train_tf = create_transform(**data_cfg, is_training=True)
    eval_tf = create_transform(**data_cfg, is_training=False)

    train_ds = CsvSpecimenDataset(train_df, args.images_root, label_to_idx, train_tf)
    test_ds = CsvSpecimenDataset(test_df, args.images_root, label_to_idx, eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=meta["batch_size"],
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=meta["batch_size"],
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.recipe, num_classes, meta).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=meta["lr"],
        weight_decay=meta["weight_decay"],
    )
    warmup = meta["warmup_epochs"]
    epochs = meta["epochs"]
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=max(1, epochs - warmup),
        lr_min=1e-6,
        warmup_lr_init=1e-7,
        warmup_t=max(1, warmup),
        cycle_limit=1,
        t_in_epochs=True,
    )

    if args.benchmark_iters > 0 and device.type == "cuda":
        sps = run_short_benchmark(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            criterion=criterion,
            amp_dtype=amp_dtype,
            iters=args.benchmark_iters,
        )
        if sps is not None:
            spe = sps * math.ceil(len(train_df) / meta["batch_size"])
            print(
                f"[benchmark] ~{sps*1000:.1f} ms/train-step  |  ~{spe:.1f} s/epoch (train only)  |  "
                f"~{spe * epochs / 60:.1f} min for {epochs} epochs (train only, excl. eval).\n",
            )

    best_top1 = 0.0
    last_top1 = last_top5 = 0.0
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, epoch, amp_dtype
        )
        scheduler.step(epoch + 1)
        t1, t5 = evaluate(model, test_loader, device, amp_dtype)
        last_top1, last_top5 = t1, t5
        best_top1 = max(best_top1, t1)
        print(
            f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
            f"test_top1={t1*100:.2f}%  test_top5={t5*100:.2f}%  (best_top1={best_top1*100:.2f}%)",
        )

    summary = {
        "recipe": args.recipe,
        "timm_name": meta["timm_name"],
        "num_classes": num_classes,
        "n_train": int(len(train_df)),
        "n_test_seen": int(len(test_df)),
        "test_top1": last_top1,
        "test_top5": last_top5,
        "best_test_top1": best_top1,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), out_dir / "checkpoint_last.pth")

    print("\n" + "-" * 72)
    print(f"Final test_seen — top-1: {last_top1*100:.2f}%  |  top-5: {last_top5*100:.2f}%")
    print(f"Best test_seen top-1 this run: {best_top1*100:.2f}%")
    print(f"Saved: {out_dir / 'results.json'}, {out_dir / 'checkpoint_last.pth'}")
    print("-" * 72)


if __name__ == "__main__":
    main()
