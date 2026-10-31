#!/usr/bin/env python3
"""
LoRA **vision tower only** — contrastive fine-tune CLIP / SigLIP / SigLIP2 on (image, label_text) pairs,
optionally plus **the same SigLIP loss** between image and **pooled metadata text** (``weak_context_text``
split into 4 segments, each encoded with the **same** frozen text tower, then mean-pool + L2).

``label_text`` and metadata chunks share one text encoder, so their embeddings live in the **same**
representation space; the extra term only adds another positive text target per image.

Example:

  pip install peft
  python scripts/train_lora_vision.py \\
    --csv /path/to/specimen_benchmark_metadata.csv \\
    --images-root /path/to/final_benchmark_data_crops \\
    --train-split train \\
    --text-col label_text \\
    --weak-col weak_context_text --meta-loss-weight 0.25 \\
    --model google/siglip2-base-patch16-512 \\
    --output-dir ./runs/siglip2_lora_vision

Eval afterward: load ``base_model`` + this folder with PEFT, or merge adapters (see HF PEFT docs).

Requires: torch, transformers, peft, pandas, pillow, tqdm (see requirements.txt).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clip_siglip.data.taxonomy_csv import TaxonomyTable
from clip_siglip.data.weak_context_split import split_weak_context_four
from clip_siglip.train.lora_vision import (
    build_vision_lora_config,
    load_model_for_contrastive_tune,
    save_tune_bundle,
)


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


class _ImageTextPaths(Dataset):
    def __init__(self, paths: list[Path], texts: list[str], weaks: list[str] | None = None):
        assert len(paths) == len(texts)
        self.paths = paths
        self.texts = texts
        self.weaks = weaks

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        if self.weaks is None:
            return self.paths[i], self.texts[i]
        return self.paths[i], self.texts[i], self.weaks[i]


def _collate_paths_texts(batch):
    if len(batch[0]) == 2:
        paths, texts = zip(*batch)
        return list(paths), list(texts), None
    paths, texts, weaks = zip(*batch)
    return list(paths), list(texts), list(weaks)


def _forward_loss(
    model,
    processor,
    image_paths: list[Path],
    texts: list[str],
    device: torch.device,
    *,
    lowercase_text: bool,
):
    from PIL import Image

    if lowercase_text:
        texts = [t.lower() for t in texts]
    images = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        images.append(im)
    inputs = processor(
        text=list(texts),
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    out = model(**inputs, return_loss=True)
    if out.loss is None:
        raise RuntimeError(
            "Model returned loss=None. Check that the checkpoint supports contrastive return_loss "
            "(CLIP / SigLIP / SigLIP2 multimodal forward).",
        )
    return out.loss


def _unwrap_base_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def _siglip_sigmoid_nll(logits: torch.Tensor) -> torch.Tensor:
    """Same reduction as HF ``SiglipModel`` when ``return_loss=True`` (pairwise sigmoid NLL)."""
    b = logits.size(0)
    eye = torch.eye(b, device=logits.device, dtype=logits.dtype)
    m1_diag1 = -torch.ones_like(logits) + 2 * eye
    loglik = F.logsigmoid(m1_diag1 * logits)
    nll = -torch.sum(loglik, dim=-1)
    return nll.mean()


def _pooled_meta_text_embeds(
    model: torch.nn.Module,
    processor,
    weak_batch: list[str],
    device: torch.device,
    *,
    lowercase_text: bool,
) -> torch.Tensor:
    """Four chunks per sample → text tower (frozen) → mean → L2, shape ``[B, D]``."""
    segs: list[tuple[str, str, str, str]] = []
    for s in weak_batch:
        try:
            segs.append(split_weak_context_four(s))
        except ValueError:
            segs.append(("", "", "", ""))
    flat: list[str] = []
    for a, b, c, d in segs:
        for t in (a, b, c, d):
            tt = (t.lower() if lowercase_text else t) or " "
            flat.append(tt)
    tok = processor(text=flat, return_tensors="pt", padding=True, truncation=True)
    tid = tok["input_ids"].to(device, non_blocking=True)
    mask = tok.get("attention_mask")
    if mask is not None:
        mask = mask.to(device, non_blocking=True)
    base = _unwrap_base_model(model)
    bsz = len(weak_batch)
    with torch.no_grad():
        tout = base.get_text_features(input_ids=tid, attention_mask=mask)
        if isinstance(tout, torch.Tensor):
            tfe = tout.float()
        elif hasattr(tout, "pooler_output") and tout.pooler_output is not None:
            tfe = tout.pooler_output.float()
        elif hasattr(tout, "last_hidden_state"):
            tfe = tout.last_hidden_state.float().mean(dim=1)
        else:
            raise RuntimeError(f"Unexpected get_text_features return type: {type(tout)}")
        tfe = tfe.view(bsz, 4, -1).mean(dim=1)
        tfe = F.normalize(tfe, dim=-1)
    return tfe


def _forward_loss_with_optional_meta(
    model,
    processor,
    image_paths: list[Path],
    texts: list[str],
    weak_batch: list[str] | None,
    device: torch.device,
    *,
    lowercase_text: bool,
    meta_loss_weight: float,
) -> torch.Tensor:
    from PIL import Image

    caps = [t.lower() for t in texts] if lowercase_text else list(texts)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(
        text=list(caps),
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    out = model(**inputs, return_loss=True)
    if out.loss is None or out.image_embeds is None:
        raise RuntimeError("Model must return loss and image_embeds (SigLIP / SigLIP-style).")
    loss = out.loss
    if meta_loss_weight > 0 and weak_batch is not None:
        meta_emb = _pooled_meta_text_embeds(model, processor, weak_batch, device, lowercase_text=lowercase_text)
        img_e = F.normalize(out.image_embeds.float(), dim=-1)
        scale = model.logit_scale.exp()
        bias = model.logit_bias
        logits = scale * (img_e @ meta_emb.t()) + bias
        loss = loss + meta_loss_weight * _siglip_sigmoid_nll(logits)
    return loss


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--images-root", required=True, type=Path)
    p.add_argument("--train-split", default="train", help="Value in CSV ``split`` column (default: train).")
    p.add_argument(
        "--species-split",
        default="",
        help="Optional: restrict ``species_split`` (comma-separated, logical AND with --train-split).",
    )
    p.add_argument("--text-col", default="label_text", help="Caption column (default: label_text).")
    p.add_argument("--model", dest="model_id", required=True, help="HF model id or local path.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--bf16",
        action="store_true",
        help="Load and train in bfloat16 (recommended on Ampere+ if stable).",
    )
    p.add_argument(
        "--no-lowercase-text",
        action="store_true",
        help="Keep caption casing (default: lowercase, matching SigLIP2-style pretraining).",
    )
    p.add_argument(
        "--weak-col",
        default=None,
        metavar="COL",
        help="CSV column for weak_context_text (4-segment template). Required if --meta-loss-weight > 0.",
    )
    p.add_argument(
        "--meta-loss-weight",
        type=float,
        default=0.0,
        help="Add SigLIP-style loss between image and pooled metadata text (same text encoder as caption). 0 disables.",
    )
    args = p.parse_args()
    lowercase_captions = not args.no_lowercase_text

    _set_seed(args.seed)

    import pandas as pd

    table = TaxonomyTable.from_csv(args.csv, images_root=args.images_root)
    ss = _parse_csv_list(args.species_split) or None
    df = table.subset(species_splits=ss, split_assignments=[args.train_split])
    if args.text_col not in df.columns:
        print(f"CSV has no column {args.text_col!r}.", file=sys.stderr)
        sys.exit(1)
    if len(df) == 0:
        print("No training rows after filters; check --train-split / --species-split.", file=sys.stderr)
        sys.exit(1)

    if args.meta_loss_weight > 0:
        if not args.weak_col:
            print("When --meta-loss-weight > 0, set --weak-col (e.g. weak_context_text).", file=sys.stderr)
            sys.exit(1)
        if args.weak_col not in df.columns:
            print(f"CSV has no column {args.weak_col!r}.", file=sys.stderr)
            sys.exit(1)

    paths: list[Path] = []
    texts: list[str] = []
    weaks: list[str] = []
    skipped_empty = 0
    skipped_weak = 0
    for _, row in df.iterrows():
        raw = row[args.text_col]
        if pd.isna(raw) or not str(raw).strip():
            skipped_empty += 1
            continue
        if args.meta_loss_weight > 0:
            w = row[args.weak_col]
            if pd.isna(w) or not str(w).strip():
                skipped_weak += 1
                continue
            weaks.append(str(w).strip())
        paths.append(Path(args.images_root) / str(row["image_name"]))
        texts.append(str(raw).strip())

    if skipped_empty:
        print(f"[train] skipped {skipped_empty} rows with empty {args.text_col}", file=sys.stderr)
    if skipped_weak:
        print(f"[train] skipped {skipped_weak} rows with empty {args.weak_col}", file=sys.stderr)

    missing = [x for x in paths if not x.is_file()]
    if missing:
        print(f"[train] warning: {len(missing)} missing files (e.g. {missing[:3]})", file=sys.stderr)

    if not paths:
        print("No usable rows (all captions empty or no images).", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else None
    if args.bf16 and not torch.cuda.is_available():
        print("[train] --bf16 ignored (no CUDA)", file=sys.stderr)

    model, processor, kind = load_model_for_contrastive_tune(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    )
    model = model.to(device)
    if dtype is None:
        model = model.float()

    try:
        from peft import get_peft_model
    except ImportError:
        print("Install peft:  pip install peft", file=sys.stderr)
        sys.exit(1)

    for _, p_ in model.named_parameters():
        p_.requires_grad = False

    lora_cfg = build_vision_lora_config(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    trainable = sum(int(x.numel()) for x in model.parameters() if x.requires_grad)
    total = sum(int(x.numel()) for x in model.parameters())
    print(f"[train] kind={kind} trainable_params={trainable} / total={total}", file=sys.stderr)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    ds = _ImageTextPaths(paths, texts, weaks if args.meta_loss_weight > 0 else None)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_paths_texts,
        pin_memory=torch.cuda.is_available(),
    )

    opt = torch.optim.AdamW(
        (x for x in model.parameters() if x.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch_paths, batch_texts, batch_weaks in pbar:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(dtype == torch.bfloat16)):
                if args.meta_loss_weight > 0:
                    loss = _forward_loss_with_optional_meta(
                        model,
                        processor,
                        batch_paths,
                        batch_texts,
                        batch_weaks,
                        device,
                        lowercase_text=lowercase_captions,
                        meta_loss_weight=args.meta_loss_weight,
                    )
                else:
                    loss = _forward_loss(
                        model,
                        processor,
                        batch_paths,
                        batch_texts,
                        device,
                        lowercase_text=lowercase_captions,
                    )
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [x for x in model.parameters() if x.requires_grad],
                    args.max_grad_norm,
                )
            opt.step()
            steps += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()))

    save_tune_bundle(
        model,
        processor,
        args.output_dir,
        base_model_id=args.model_id,
        train_meta={
            "train_split": args.train_split,
            "species_split_filter": ss,
            "text_col": args.text_col,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "bf16": bool(dtype == torch.bfloat16),
            "lowercase_text": lowercase_captions,
            "steps": steps,
            "n_samples": len(paths),
            "weak_col": args.weak_col,
            "meta_loss_weight": float(args.meta_loss_weight),
        },
    )
    print(f"[train] saved adapter + processor to {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
