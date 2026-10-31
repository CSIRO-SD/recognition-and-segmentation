#!/usr/bin/env python3
"""Train a small late-fusion MLP on top of **frozen** SigLIP/SigLIP2 image+text towers.

Uses per-class logits from the official SigLIP dot-product + (``logit_scale``, ``logit_bias``) when
present, then a shared 2→hidden→1 MLP on ``(s_image, s_meta)`` for each class.

Metadata: ``weak_context_text`` is split into 4 chunks (Latitude / Sampling event / Notes template);
each chunk is encoded with the **same** frozen text encoder; embeddings are mean-pooled and
L2-normalized before matching class text prototypes.

Example::

  python scripts/train_metadata_fusion.py \\
    --csv /path/to/specimen_benchmark_metadata.csv \\
    --images-root /path/to/crops \\
    --train-split train \\
    --model google/siglip2-base-patch16-512 \\
    --output-dir ./runs/siglip2_fusion_head

Eval::

  python scripts/run_metadata_fusion_eval.py --fusion-dir ./runs/siglip2_fusion_head ...
"""

from __future__ import annotations

import argparse
import json
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


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


from clip_siglip.data.taxonomy_csv import TaxonomyTable  # noqa: E402
from clip_siglip.data.weak_context_split import split_weak_context_four  # noqa: E402
from clip_siglip.eval.zeroshot import build_class_prompts, _norm_cell  # noqa: E402
from clip_siglip.models.wrappers import build_vl  # noqa: E402
from clip_siglip.train.fusion_module import FusionHead, fused_class_logits, pooled_meta_embedding  # noqa: E402


class _FusionTrainDataset(Dataset):
    def __init__(self, paths: list[Path], weak_texts: list[str], targets: list[int]):
        assert len(paths) == len(weak_texts) == len(targets)
        self.paths = paths
        self.weak_texts = weak_texts
        self.targets = targets

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        im = Image.open(self.paths[i]).convert("RGB")
        try:
            segs = split_weak_context_four(self.weak_texts[i])
        except ValueError:
            segs = ("", "", "", "")
        return im, int(self.targets[i]), segs


def _collate_fusion(batch):
    ims, ys, segs = zip(*batch)
    return list(ims), torch.tensor(ys, dtype=torch.long), list(segs)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_prototypes(
    backbone,
    labels_sorted: list[str],
    class_prompts: dict[str, list[str]],
    device: torch.device,
) -> torch.Tensor:
    text_embs: list[torch.Tensor] = []
    for lab in labels_sorted:
        prompts = [str(x) for x in class_prompts[lab]]
        txs = backbone.encode_text(prompts)
        proto = txs.mean(dim=0)
        text_embs.append(proto)
    protos = torch.stack(text_embs, dim=0)
    protos = F.normalize(protos.float(), dim=-1).to(device)
    return protos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--images-root", required=True, type=Path)
    p.add_argument("--train-split", default="train")
    p.add_argument("--species-split", default="", help="Optional CSV species_split filter (comma list).")
    p.add_argument("--model", dest="model_id", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--weak-col", default="weak_context_text")
    p.add_argument("--eval-class-col", default="taxon_key")
    p.add_argument("--prompt-col", default="label_text", help="Build text prototypes from this column.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--fusion-hidden", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--peft-adapter", type=Path, default=None, metavar="DIR")
    args = p.parse_args()

    _set_seed(args.seed)

    import pandas as pd

    table = TaxonomyTable.from_csv(args.csv, images_root=args.images_root)
    ss = _parse_csv_list(args.species_split) or None
    df_tr = table.subset(species_splits=ss, split_assignments=[args.train_split])
    if args.weak_col not in df_tr.columns:
        print(f"No column {args.weak_col!r}", file=sys.stderr)
        sys.exit(1)
    if args.eval_class_col not in df_tr.columns:
        print(f"No column {args.eval_class_col!r}", file=sys.stderr)
        sys.exit(1)

    df_tr = df_tr.copy()
    df_tr["__ckey"] = df_tr[args.eval_class_col].map(_norm_cell)
    labels_sorted = sorted(df_tr["__ckey"].unique().tolist())
    if "" in labels_sorted:
        labels_sorted.remove("")
    if not labels_sorted:
        print("No classes in train split after normalization.", file=sys.stderr)
        sys.exit(1)
    lab_to_idx = {lab: i for i, lab in enumerate(labels_sorted)}

    mini_for_prompts = df_tr[df_tr["__ckey"].isin(labels_sorted)].drop_duplicates(subset=["__ckey"])
    class_prompts = build_class_prompts(
        mini_for_prompts,
        class_col="__ckey",
        scientific_name_col="scientific_name" if "scientific_name" in mini_for_prompts.columns else None,
        prompt_col=args.prompt_col if args.prompt_col in mini_for_prompts.columns else None,
    )

    paths: list[Path] = []
    weak_tx: list[str] = []
    ys: list[int] = []
    skipped = 0
    for _, row in df_tr.iterrows():
        ck = _norm_cell(row[args.eval_class_col])
        if ck not in lab_to_idx:
            skipped += 1
            continue
        w = row[args.weak_col]
        if pd.isna(w) or not str(w).strip():
            skipped += 1
            continue
        paths.append(args.images_root / str(row["image_name"]))
        weak_tx.append(str(w).strip())
        ys.append(lab_to_idx[ck])

    if skipped:
        print(f"[train] skipped rows (empty weak / unknown class): {skipped}", file=sys.stderr)
    if not paths:
        print("No training samples.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_vl(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        peft_adapter=args.peft_adapter,
    )
    raw = backbone.model
    raw.eval()
    for p_ in raw.parameters():
        p_.requires_grad = False

    prototypes = _build_prototypes(backbone, labels_sorted, class_prompts, device)

    fusion = FusionHead(hidden_dim=args.fusion_hidden).to(device)
    opt = torch.optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = torch.nn.CrossEntropyLoss()

    ds = _FusionTrainDataset(paths, weak_tx, ys)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate_fusion,
        pin_memory=torch.cuda.is_available(),
    )

    for epoch in range(args.epochs):
        fusion.train()
        losses = []
        for pils, y, segs in tqdm(dl, desc=f"fusion epoch {epoch+1}/{args.epochs}"):
            y = y.to(device)
            with torch.no_grad():
                im = backbone.encode_images_pil(pils)
                meta = pooled_meta_embedding(backbone, segs)
            logits = fused_class_logits(im, meta, prototypes, raw, fusion, mode="both")
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        print(f"[train] epoch {epoch+1} mean_loss={np.mean(losses):.4f}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(fusion.state_dict(), args.output_dir / "fusion_head.pt")
    meta = {
        "schema": "clip_siglip_metadata_fusion_v1",
        "model_id": args.model_id,
        "weak_split": "latitude_sampling_notes_4",
        "weak_col": args.weak_col,
        "class_col": args.eval_class_col,
        "prompt_col": args.prompt_col,
        "labels_sorted": labels_sorted,
        "fusion_hidden": args.fusion_hidden,
        "train_split": args.train_split,
        "species_split_filter": ss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_train_samples": len(paths),
    }
    (args.output_dir / "fusion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] wrote {args.output_dir / 'fusion_head.pt'} and fusion_meta.json", file=sys.stderr)


if __name__ == "__main__":
    main()
