#!/usr/bin/env python3
"""
Zero-shot image → text (taxonomy) classification: no training, no extra note fields.

Uses CLIP / SigLIP: embed each class with prompts built from ``taxon_key`` / ``scientific_name``,
or from ``--prompt-col`` (e.g. ``label_text``), then match each image embedding by cosine (argmax).

Optional **per-image metadata** (same protocol as fusion): ``--weak-col weak_context_text`` splits each
row into four chunks (Latitude / Sampling event / Notes), encodes with the **same** text tower,
mean-pools to ``meta_emb``, builds class logits ``meta_emb @ prototypes`` (with SigLIP
``logit_scale`` / ``logit_bias`` when present), and fuses with image logits using
``--weak-fusion-alpha`` (weight on the **image** branch; metadata gets ``1-α``). ``α=1`` is plain
image-only (default behavior).

Verifying model vs paper (protocol must match):
  - SigLIP/2 papers report ImageNet-1K zero-shot with **fixed templates** (often different from
    your insect ``label_text``). To sanity-check **your HF checkpoint**, run an **official**
    ILSVRC eval (e.g. Hugging Face ``evaluate`` / OpenCLIP imagenet zeroshot / paper Colab) with the
    **same template set** as the table row you compare to.
  - Your insect CSV eval only checks **pipeline correctness + domain**; numbers will not match
    ImageNet tables.

Metrics (this script):
  - top-1 / top-k on the **eval class column** (default: ``taxon_key``; if ``--prompt-col`` is set,
    defaults to that column e.g. ``label_text`` so labels match text prototypes). Optional
    ``--eval-class-col`` overrides. Hierarchical acc: **order** marginal; **family** requires matching
    order; **genus** requires order+family; **species** (CSV epithet) requires order+family+genus —
    same denominator ``n_samples`` for all.

Example:

  source .env.example  # HF cache (edit paths locally — not modified by repo defaults)
  python scripts/run_zeroshot.py \\
    --csv /path/to/metadata.csv \\
    --images-root /path/to/images \\
    --species-split unseen \\
    --model openai/clip-vit-base-patch32

  # Benchmark-style: eval ``split`` column — test_seen then test_unseen (one process, two metrics)
  python scripts/run_zeroshot.py --csv ... --images-root ... \\
    --eval-split test_seen,test_unseen --prompt-col label_text \\
    --weak-col weak_context_text --weak-fusion-alpha 0.5 \\
    --model google/siglip2-base-patch16-512 --peft-adapter ./runs/siglip2_lora_vision_meta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clip_siglip.data.taxonomy_csv import TaxonomyTable
from clip_siglip.eval.zeroshot import (
    _norm_cell,
    build_class_prompts,
    build_hierarchy_by_class_col,
    zeroshot_metrics_pil,
)
from clip_siglip.models.wrappers import build_vl


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _print_load_summary(backbone, requested_id: str) -> None:
    """Confirm HF resolved the expected architecture (AutoModel reads repo ``config.json``)."""
    m = backbone.model
    cfg = getattr(m, "config", None)
    wrapper = type(backbone).__name__
    print("[clip_siglip] Loaded backbone", file=sys.stderr)
    print(f"  --model argument:   {requested_id!r}", file=sys.stderr)
    print(f"  wrapper chosen:     {wrapper}", file=sys.stderr)
    print(f"  torch module class: {m.__class__.__name__}", file=sys.stderr)
    if cfg is not None:
        mt = getattr(cfg, "model_type", None)
        nop = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
        print(f"  config.model_type:  {mt!r}", file=sys.stderr)
        if nop:
            print(f"  config.name_or_path: {nop!r}", file=sys.stderr)
    print(f"  processor class:    {backbone.processor.__class__.__name__}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--images-root", required=True, type=Path)
    p.add_argument(
        "--species-split",
        default="",
        help="Optional: filter ``species_split`` (seen | unseen). Comma-separated = union.",
    )
    p.add_argument(
        "--eval-split",
        default="",
        help="Optional: filter CSV ``split`` column (e.g. test_seen, test_unseen). "
        "Comma-separated runs multiple evaluations in one invocation.",
    )
    p.add_argument("--model", dest="model_id", default="openai/clip-vit-base-patch32", help="HF model id")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--scientific-name-col",
        default="scientific_name",
        help="Prompts use this column when present; rows still work without it",
    )
    p.add_argument("--taxon-col", default="taxon_key")
    p.add_argument(
        "--prompt-col",
        default=None,
        metavar="COL",
        help="If set (e.g. label_text), build text prototypes from this CSV column instead of "
        "scientific_name/taxon_key templates.",
    )
    p.add_argument(
        "--eval-class-col",
        default=None,
        metavar="COL",
        help="Column used as class id for top-1/top-k and prototype keys (normalized). "
        "Default: same as --prompt-col when --prompt-col is set, else --taxon-col. "
        "Use e.g. --eval-class-col taxon_key --prompt-col label_text for taxon-level classes with label prompts.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k accuracy k (capped at num classes in eval subset).",
    )
    p.add_argument(
        "--show-model",
        action="store_true",
        help="Print loaded PyTorch class, config.model_type, and processor (stderr).",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass to ``AutoModel.from_pretrained`` (SigLIP/Auto path only; EvaCLIP forces this on).",
    )
    p.add_argument(
        "--peft-adapter",
        type=Path,
        default=None,
        metavar="DIR",
        help="PEFT adapter dir from scripts/train_lora_vision.py; merged into base weights before eval.",
    )
    p.add_argument(
        "--hierarchy-report-prefix",
        type=Path,
        default=None,
        metavar="PREFIX",
        help="Write PREFIX_order_per_gt.tsv and PREFIX_order_confusion_wrong.tsv (order diagnostics). "
        "When using multiple --eval-split values, the split tag is appended to the filename stem.",
    )
    p.add_argument(
        "--hierarchy-debug",
        action="store_true",
        help="Print top wrong order pairs and weakest per-gt_order acc to stderr.",
    )
    p.add_argument(
        "--weak-col",
        default=None,
        metavar="COL",
        help="Optional: per-row metadata (e.g. weak_context_text). Four-way split + pooled text embedding "
        "fused with image logits; see --weak-fusion-alpha.",
    )
    p.add_argument(
        "--weak-fusion-alpha",
        type=float,
        default=0.5,
        help="When --weak-col is set: weight on image branch in [0,1]; metadata branch gets 1-α. "
        "α=1.0: image only (ignore metadata). α=0.0: metadata only.",
    )
    args = p.parse_args()

    ss = _parse_csv_list(args.species_split) or None
    eval_parts = _parse_csv_list(args.eval_split)
    if not eval_parts:
        eval_parts = [None]

    if ss is None and eval_parts == [None]:
        print(
            "Specify at least one of --species-split or --eval-split (otherwise the eval subset is ambiguous).",
            file=sys.stderr,
        )
        sys.exit(1)

    table = TaxonomyTable.from_csv(args.csv, images_root=args.images_root)

    import pandas as pd

    if args.weak_col is not None and not (0.0 <= args.weak_fusion_alpha <= 1.0):
        print("--weak-fusion-alpha must be in [0,1].", file=sys.stderr)
        sys.exit(1)

    backbone = build_vl(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        peft_adapter=args.peft_adapter,
    )
    if args.show_model:
        _print_load_summary(backbone, args.model_id)

    for ep in eval_parts:
        split_assign = [ep] if ep else None
        df = table.subset(species_splits=ss, split_assignments=split_assign)
        if len(df) == 0:
            tag = ep or "(empty eval-split)"
            print(f"No rows after filters for eval_split={tag!r}; check CSV.", file=sys.stderr)
            sys.exit(1)

        if args.eval_class_col:
            class_col = args.eval_class_col
        elif args.prompt_col:
            class_col = args.prompt_col
        else:
            class_col = args.taxon_col
        if class_col not in df.columns:
            print(f"Eval class column {class_col!r} not in CSV.", file=sys.stderr)
            sys.exit(1)

        if args.weak_col is not None and args.weak_col not in df.columns:
            print(f"CSV has no column {args.weak_col!r}.", file=sys.stderr)
            sys.exit(1)

        paths: list[Path] = []
        gt_samples: list[dict] = []
        weak_texts: list[str] = []
        for _, row in df.iterrows():
            ip = Path(args.images_root) / str(row["image_name"])
            paths.append(ip)
            gt_samples.append(
                {
                    "eval_class": _norm_cell(row[class_col]),
                    "order": row["order"] if "order" in df.columns else "",
                    "family": row["family"] if "family" in df.columns else "",
                    "genus": row["genus"] if "genus" in df.columns else "",
                    "species": row["species"] if "species" in df.columns else "",
                }
            )
            if args.weak_col is not None:
                w = row[args.weak_col]
                weak_texts.append("" if pd.isna(w) else str(w).strip())

        missing = [x for x in paths if not x.is_file()]
        if missing[:5]:
            print(f"Warning: {len(missing)} missing image paths (showing first 5):", file=sys.stderr)
            for m in missing[:5]:
                print(" ", m, file=sys.stderr)

        sci_col = args.scientific_name_col if args.scientific_name_col in df.columns else None

        class_prompts = build_class_prompts(
            df,
            class_col=class_col,
            scientific_name_col=sci_col,
            prompt_col=args.prompt_col,
        )

        hierarchy = build_hierarchy_by_class_col(df, group_col=class_col)

        report_pfx: Path | None = args.hierarchy_report_prefix
        if report_pfx is not None and ep:
            report_pfx = report_pfx.parent / f"{report_pfx.name}_{ep}"

        metrics = zeroshot_metrics_pil(
            backbone,
            class_prompts,
            paths,
            hierarchy,
            gt_samples,
            batch_size=args.batch_size,
            topk=args.topk,
            top1_class_label=class_col,
            hierarchy_report_prefix=report_pfx,
            hierarchy_debug_stderr=args.hierarchy_debug,
            weak_context_texts=weak_texts if args.weak_col is not None else None,
            weak_fusion_alpha=args.weak_fusion_alpha,
        )

        ss_part = ",".join(ss) if ss else "*"
        ep_part = ep if ep else "*"
        weak_part = ""
        if args.weak_col is not None:
            weak_part = f" weak={args.weak_col!r} α_img={args.weak_fusion_alpha}"
        print(
            f"Zero-shot species_split=[{ss_part}] csv_split=[{ep_part}] model={args.model_id}{weak_part}",
        )
        for line in metrics.lines():
            print(line)


if __name__ == "__main__":
    main()
