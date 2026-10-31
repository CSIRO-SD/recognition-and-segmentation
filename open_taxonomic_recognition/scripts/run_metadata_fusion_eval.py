#!/usr/bin/env python3
"""Evaluate frozen SigLIP/SigLIP2 + trained metadata fusion head (see ``train_metadata_fusion.py``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


from clip_siglip.data.taxonomy_csv import TaxonomyTable
from clip_siglip.eval.metadata_fusion_eval import fusion_metrics_pil
from clip_siglip.eval.zeroshot import build_class_prompts, build_hierarchy_by_class_col, _norm_cell
from clip_siglip.models.wrappers import build_vl
from clip_siglip.train.fusion_module import FusionHead


def _build_prototypes(backbone, labels_sorted: list[str], class_prompts: dict[str, list[str]], device):
    text_embs: list[torch.Tensor] = []
    for lab in labels_sorted:
        prompts = [str(x) for x in class_prompts[lab]]
        txs = backbone.encode_text(prompts)
        proto = txs.mean(dim=0)
        text_embs.append(proto)
    protos = torch.stack(text_embs, dim=0)
    protos = torch.nn.functional.normalize(protos.float(), dim=-1).to(device)
    return protos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--images-root", required=True, type=Path)
    p.add_argument("--fusion-dir", required=True, type=Path, help="Dir with fusion_head.pt + fusion_meta.json")
    p.add_argument("--eval-split", default="test_seen,test_unseen")
    p.add_argument("--species-split", default="", help="Optional species_split filter (comma).")
    p.add_argument("--model", dest="model_id", default=None, help="Override base model (default: fusion_meta.json).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--fusion-mode", choices=("both", "image_only", "meta_only"), default="both")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--peft-adapter", type=Path, default=None)
    p.add_argument("--hierarchy-report-prefix", type=Path, default=None)
    p.add_argument("--hierarchy-debug", action="store_true")
    args = p.parse_args()

    meta_path = args.fusion_dir / "fusion_meta.json"
    ckpt_path = args.fusion_dir / "fusion_head.pt"
    if not meta_path.is_file() or not ckpt_path.is_file():
        print(f"Need {meta_path} and {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    labels_sorted: list[str] = meta["labels_sorted"]
    class_col = meta["class_col"]
    prompt_col = meta.get("prompt_col")
    weak_col = meta.get("weak_col", "weak_context_text")
    fusion_h = int(meta.get("fusion_hidden", 64))
    model_id = args.model_id or meta["model_id"]

    fusion = FusionHead(hidden_dim=fusion_h)
    fusion.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    fusion.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_vl(
        model_id,
        trust_remote_code=args.trust_remote_code,
        peft_adapter=args.peft_adapter,
    )
    raw = backbone.model
    raw.eval()
    fusion.to(device)

    table = TaxonomyTable.from_csv(args.csv, images_root=args.images_root)
    full_df = table.df
    ss = _parse_csv_list(args.species_split) or None

    key_to_idx = {k: i for i, k in enumerate(labels_sorted)}
    ref = full_df[full_df[class_col].map(_norm_cell).isin(labels_sorted)].copy()
    ref["__ckey"] = ref[class_col].map(_norm_cell)
    class_prompts = build_class_prompts(
        ref,
        class_col="__ckey",
        scientific_name_col="scientific_name" if "scientific_name" in ref.columns else None,
        prompt_col=prompt_col if prompt_col and prompt_col in ref.columns else None,
    )
    for lab in labels_sorted:
        if lab not in class_prompts:
            print(f"Missing prompts for class {lab!r} — check CSV coverage.", file=sys.stderr)
            sys.exit(1)

    prototypes = _build_prototypes(backbone, labels_sorted, class_prompts, device)

    eval_parts = _parse_csv_list(args.eval_split)
    if not eval_parts:
        eval_parts = [None]

    for ep in eval_parts:
        split_assign = [ep] if ep else None
        df = table.subset(species_splits=ss, split_assignments=split_assign)
        if len(df) == 0:
            print(f"No rows for eval_split={ep!r}", file=sys.stderr)
            sys.exit(1)

        paths: list[Path] = []
        weak_list: list[str] = []
        gt_samples: list[dict] = []
        for _, row in df.iterrows():
            paths.append(args.images_root / str(row["image_name"]))
            w = row.get(weak_col, "")
            weak_list.append("" if pd.isna(w) else str(w).strip())
            gt_samples.append(
                {
                    "eval_class": _norm_cell(row[class_col]),
                    "order": row["order"] if "order" in df.columns else "",
                    "family": row["family"] if "family" in df.columns else "",
                    "genus": row["genus"] if "genus" in df.columns else "",
                    "species": row["species"] if "species" in df.columns else "",
                }
            )

        hierarchy = build_hierarchy_by_class_col(ref, group_col="__ckey")
        report_pfx = args.hierarchy_report_prefix
        if report_pfx is not None and ep:
            report_pfx = report_pfx.parent / f"{report_pfx.name}_{ep}"

        metrics, st = fusion_metrics_pil(
            backbone,
            raw,
            fusion,
            prototypes=prototypes,
            labels_sorted=labels_sorted,
            image_paths=paths,
            weak_context_texts=weak_list,
            hierarchy_by_class=hierarchy,
            gt_samples=gt_samples,
            batch_size=args.batch_size,
            topk=args.topk,
            fusion_mode=args.fusion_mode,
            top1_class_label=class_col,
            hierarchy_report_prefix=report_pfx,
            hierarchy_debug_stderr=args.hierarchy_debug,
            key_to_idx=key_to_idx,
        )

        tag = ep or "*"
        print(
            f"Metadata-fusion ({args.fusion_mode}) csv_split=[{tag}] model={model_id} "
            f"evaluated={st['n_evaluated']} skipped_oov={st['n_skipped_oov']}",
        )
        for line in metrics.lines():
            print(line)


if __name__ == "__main__":
    main()
