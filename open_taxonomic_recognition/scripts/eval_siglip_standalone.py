#!/usr/bin/env python3
"""Standalone zero-shot eval for any HF SigLIP/SigLIP2/CLIP-style model.

Self-contained: does NOT import anything from ``clip_siglip.eval`` or ``clip_siglip.models``.
Use this to debug discrepancies vs ``scripts/run_zeroshot.py`` without touching that pipeline.

Reports two metric variants per split:
  * STRICT (legacy):   per-rank denominator excludes samples with empty GT for that rank;
                       no missing-token unification; family/genus/species are independent matches.
  * LENIENT (current): denominator = all samples; ``unknown/n/a/...`` -> "" so empty matches empty;
                       family/genus/species are CONDITIONAL on higher ranks matching.

Plus: data sanity checks (class count, empty-rank fractions, prediction hierarchy empties).

Example
-------
python scripts/eval_siglip_standalone.py \
    --csv /path/to/specimen_benchmark_metadata.csv \
    --images-root /path/to/final_benchmark_data_crops \
    --eval-split test_seen,test_unseen \
    --prompt-col label_text \
    --model google/siglip2-large-patch16-256 \
    --batch-size 32 --topk 5
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


# ---------------------------------------------------------------------------
# normalization helpers (independent from clip_siglip)
# ---------------------------------------------------------------------------

_MISSING_RANK_TOKENS = frozenset({
    "unknown", "unk", "n/a", "na", "none", "-",
    "unclassified", "not classified", "missing", "null",
    "no data", "nodata",
})


def _norm_cell(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def _norm_rank_strict(x: Any) -> str:
    """Strict: only strip whitespace; ``unknown`` stays as ``unknown``."""
    return _norm_cell(x)


def _norm_rank_lenient(x: Any) -> str:
    """Lenient: missing tokens map to ``""``."""
    s = _norm_cell(x)
    if not s:
        return ""
    if s.lower() in _MISSING_RANK_TOKENS:
        return ""
    return s


# ---------------------------------------------------------------------------
# prompts (mirrors clip_siglip.eval defaults so numbers are comparable)
# ---------------------------------------------------------------------------

def default_prompt_templates(class_key: str, scientific_name: str | None) -> list[str]:
    t = scientific_name.strip() if scientific_name else class_key.replace("__", " ")
    return [
        f"a cropped photo of {t}",
        f"a specimen of taxon {t}",
        f"{t}",
    ]


def label_text_prompt_templates(label_text: str) -> list[str]:
    t = label_text.strip()
    if not t:
        return [""]
    return [
        t,
        f"a museum specimen described as: {t}",
        f"a pinned insect specimen. {t}",
    ]


def build_class_prompts(
    df: pd.DataFrame,
    class_col: str,
    scientific_name_col: str | None,
    prompt_col: str | None,
) -> dict[str, list[str]]:
    work = df.copy()
    work["__class_key"] = work[class_col].map(_norm_cell)
    has_sci = bool(scientific_name_col and scientific_name_col in df.columns)
    use_prompt_col = bool(prompt_col and prompt_col in df.columns)

    out: dict[str, list[str]] = {}
    for _, g in work.groupby("__class_key"):
        key = str(g["__class_key"].iloc[0])

        if use_prompt_col:
            raw = g[prompt_col].dropna().astype(str).map(str.strip)
            raw = raw[raw != ""]
            if len(raw) == 0:
                row0 = g.iloc[0]
                sci = None
                if has_sci:
                    v = row0[scientific_name_col]
                    if pd.notna(v) and str(v).strip():
                        sci = str(v).strip()
                out[key] = default_prompt_templates(key, sci)
                continue
            uniq = raw.unique().tolist()
            if len(uniq) > 1:
                chosen = sorted(uniq, key=lambda s: (-len(s), s))[0]
            else:
                chosen = uniq[0]
            out[key] = label_text_prompt_templates(chosen)
            continue

        row0 = g.iloc[0]
        sci = None
        if has_sci:
            v = row0[scientific_name_col]
            if pd.notna(v) and str(v).strip():
                sci = str(v).strip()
        out[key] = default_prompt_templates(key, sci)
    return out


def build_hierarchy_by_class(df: pd.DataFrame, class_col: str) -> dict[str, dict[str, str]]:
    """One hierarchy dict per distinct normalized ``class_col`` value (first row)."""
    cols = [c for c in ("order", "family", "genus", "species") if c in df.columns]
    out: dict[str, dict[str, str]] = {}
    work = df.copy()
    work["__g"] = work[class_col].map(_norm_cell)
    for _, g in work.groupby("__g"):
        r = g.iloc[0]
        key = str(r["__g"])
        out[key] = {c: _norm_cell(r.get(c)) for c in cols} if cols else {}
    return out


# ---------------------------------------------------------------------------
# model wrapper (auto for SigLIP/SigLIP2; works for any model with get_*_features)
# ---------------------------------------------------------------------------

class StandaloneVL:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        trust_remote_code: bool = False,
        siglip_fixed_padding: bool = False,
    ):
        self.device = device
        self.siglip_fixed_padding = siglip_fixed_padding
        kw: dict[str, Any] = {}
        if trust_remote_code:
            kw["trust_remote_code"] = True
        self.model = AutoModel.from_pretrained(model_id, **kw).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, **kw)
        if not hasattr(self.model, "get_image_features") or not hasattr(self.model, "get_text_features"):
            raise TypeError(
                f"{model_id}: this script only supports models exposing get_image_features and "
                f"get_text_features (CLIP/SigLIP/SigLIP2). Got {type(self.model).__name__}.",
            )
        # Read the fixed text length from text_config (SigLIP/SigLIP2 train at max_position_embeddings,
        # OpenAI CLIP at 77). Used only when --siglip-fixed-padding is on.
        self._text_max_len: int | None = None
        cfg = getattr(self.model, "config", None)
        tc = getattr(cfg, "text_config", None) if cfg is not None else None
        if tc is not None:
            n = getattr(tc, "max_position_embeddings", None)
            if n is not None:
                self._text_max_len = int(n)
        print(
            f"[info] siglip_fixed_padding={self.siglip_fixed_padding} "
            f"text_max_len={self._text_max_len}",
            file=sys.stderr,
        )

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        if self.siglip_fixed_padding and self._text_max_len:
            inp = self.processor(
                text=texts,
                return_tensors="pt",
                truncation=True,
                max_length=self._text_max_len,
                padding="max_length",
            )
        else:
            inp = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inp = {k: v.to(self.device) for k, v in inp.items() if isinstance(v, torch.Tensor)}
        out = self.model.get_text_features(**inp)
        emb = out if isinstance(out, torch.Tensor) else getattr(out, "pooler_output", None)
        if emb is None:
            emb = out.last_hidden_state[:, 0, :]
        return F.normalize(emb.float(), dim=-1)

    @torch.no_grad()
    def encode_images_pil(self, images: list[Image.Image]) -> torch.Tensor:
        images = [im.convert("RGB") for im in images]
        inp = self.processor(images=images, return_tensors="pt", padding=True)
        inp = {k: v.to(self.device) for k, v in inp.items() if isinstance(v, torch.Tensor)}
        # SigLIP2 needs all vision keys (mask + spatial_shapes); CLIP just pixel_values.
        out = self.model.get_image_features(**inp)
        emb = out if isinstance(out, torch.Tensor) else getattr(out, "pooler_output", None)
        if emb is None:
            emb = out.last_hidden_state[:, 0, :]
        return F.normalize(emb.float(), dim=-1)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class _PILDS(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return Image.open(self.paths[i]).convert("RGB"), i


def _collate(batch):
    ims, idxs = zip(*batch)
    return list(ims), list(idxs)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

@dataclass
class RankAccTracker:
    name: str
    n_strict_match: int = 0
    n_strict_den: int = 0  # only counts when GT non-empty for this rank
    n_lenient_match: int = 0
    n_lenient_den: int = 0  # always +1


def _eval_one_split(
    backbone: StandaloneVL,
    df: pd.DataFrame,
    images_root: Path,
    class_col: str,
    prompt_col: str | None,
    scientific_name_col: str | None,
    topk: int,
    batch_size: int,
    num_workers: int,
) -> None:
    if class_col not in df.columns:
        print(f"!! class_col {class_col!r} missing", file=sys.stderr)
        return

    # ---- prepare paths & per-sample GT records ----
    paths: list[Path] = []
    gt_rows: list[dict[str, str]] = []
    has_cols = [c for c in ("order", "family", "genus", "species") if c in df.columns]
    for _, row in df.iterrows():
        paths.append(images_root / str(row["image_name"]))
        rec = {"eval_class": _norm_cell(row[class_col])}
        for c in has_cols:
            rec[c] = _norm_cell(row.get(c))
        gt_rows.append(rec)

    # ---- prototypes ----
    class_prompts = build_class_prompts(df, class_col, scientific_name_col, prompt_col)
    hierarchy = build_hierarchy_by_class(df, class_col)
    labels_sorted = sorted(class_prompts.keys())
    n_cls = len(labels_sorted)
    k_eff = min(max(1, topk), n_cls)

    print(f"\n[info] split rows={len(df)}  class_col={class_col!r}  n_classes={n_cls}", file=sys.stderr)
    print(f"[info] using topk={k_eff} (capped by n_classes)", file=sys.stderr)

    # data sanity
    for c in has_cols:
        n_empty_gt = sum(1 for r in gt_rows if r[c] == "")
        n_unknown_gt = sum(1 for r in gt_rows if r[c].lower() in _MISSING_RANK_TOKENS)
        print(
            f"[sanity-gt] {c}: empty={n_empty_gt}/{len(gt_rows)}  "
            f"missing-token-like={n_unknown_gt}",
            file=sys.stderr,
        )
    for c in has_cols:
        n_empty_h = sum(1 for k in labels_sorted if not hierarchy.get(k, {}).get(c, ""))
        print(f"[sanity-pred-hier] {c}: empty in hierarchy table = {n_empty_h}/{n_cls}", file=sys.stderr)

    # ---- text prototypes ----
    proto_list = []
    for lab in labels_sorted:
        emb = backbone.encode_text([str(p) for p in class_prompts[lab]])
        proto_list.append(emb.mean(dim=0))
    prototypes = F.normalize(torch.stack(proto_list, dim=0).float(), dim=-1)

    # ---- image features ----
    ds = _PILDS(paths)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=_collate)

    n_total = 0
    n_top1 = 0
    n_topk = 0
    trackers = {c: RankAccTracker(c) for c in ("order", "family", "genus", "species") if c in has_cols}

    # cascade trackers (for the "current" lenient definition)
    n_cascade_correct: dict[str, int] = {c: 0 for c in trackers}

    for ims, idxs in tqdm(dl, desc="zero-shot"):
        feat = backbone.encode_images_pil(ims)
        logits = feat @ prototypes.T
        top1 = logits.argmax(dim=-1).cpu().tolist()
        _, topk_idx = logits.topk(k_eff, dim=-1)
        topk_idx = topk_idx.cpu().tolist()

        for j, i_g in enumerate(idxs):
            gt = gt_rows[i_g]
            pred_key = labels_sorted[top1[j]]
            ph = hierarchy.get(pred_key, {})
            pred_set = {labels_sorted[k] for k in topk_idx[j]}

            if pred_key == gt["eval_class"]:
                n_top1 += 1
            if gt["eval_class"] in pred_set:
                n_topk += 1

            # rank trackers (strict + lenient)
            cascade_ok = True
            for c, tr in trackers.items():
                gt_s = gt[c]
                p_s = _norm_cell(ph.get(c, ""))

                # strict
                if _norm_rank_strict(gt_s) != "":
                    tr.n_strict_den += 1
                    if _norm_rank_strict(p_s) == _norm_rank_strict(gt_s):
                        tr.n_strict_match += 1

                # lenient
                tr.n_lenient_den += 1
                gt_l = _norm_rank_lenient(gt_s)
                p_l = _norm_rank_lenient(p_s)
                if p_l == gt_l:
                    tr.n_lenient_match += 1
                else:
                    cascade_ok = False
                if cascade_ok:
                    n_cascade_correct[c] += 1

            n_total += 1

    # ---- print ----
    def pct(num: int, den: int) -> str:
        return "n/a" if den <= 0 else f"{(num / den) * 100:.4f}"

    print()
    print(f"top-1 ({class_col}): {pct(n_top1, n_total)}  (n={n_total})")
    print(f"top-{k_eff}: {pct(n_topk, n_total)}")

    print("\n[STRICT — legacy: skip empty GT, no missing-token unification, independent ranks]")
    for c, tr in trackers.items():
        print(
            f"  {c}: {pct(tr.n_strict_match, tr.n_strict_den)}  (correct={tr.n_strict_match} / den={tr.n_strict_den})"
        )

    print("\n[LENIENT — current: empty matches empty, missing-token unified, INDEPENDENT (no cascade)]")
    for c, tr in trackers.items():
        print(
            f"  {c}: {pct(tr.n_lenient_match, tr.n_lenient_den)}  (correct={tr.n_lenient_match} / den={tr.n_lenient_den})"
        )

    print("\n[LENIENT + CASCADE — current run_zeroshot.py definition]")
    for c, n_cor in n_cascade_correct.items():
        print(f"  {c}: {pct(n_cor, n_total)}  (correct={n_cor} / den={n_total})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--images-root", required=True, type=Path)
    p.add_argument("--eval-split", default="", help="comma-separated CSV split values (e.g. test_seen,test_unseen)")
    p.add_argument("--species-split", default="", help="optional species_split filter (comma = union)")
    p.add_argument("--model", dest="model_id", default="google/siglip2-large-patch16-256")
    p.add_argument("--prompt-col", default=None, help="e.g. label_text")
    p.add_argument(
        "--class-col",
        default=None,
        help="Class id column for prototypes/eval. Default: --prompt-col if set, else taxon_key.",
    )
    p.add_argument("--scientific-name-col", default="scientific_name")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--split-col", default="split", help="CSV column that holds split labels")
    p.add_argument("--species-split-col", default="species_split")
    p.add_argument(
        "--siglip-fixed-padding",
        action="store_true",
        help="Pad text to text_config.max_position_embeddings with padding='max_length' (SigLIP/SigLIP2 "
        "expect fixed-length inputs and pool the last token). Default: padding=True (pad to longest in batch).",
    )
    args = p.parse_args()

    df_full = pd.read_csv(args.csv)

    eval_parts = _parse_csv_list(args.eval_split) or [None]
    ss_filter = _parse_csv_list(args.species_split) or None

    class_col = args.class_col or args.prompt_col or "taxon_key"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] loading {args.model_id} on {device}", file=sys.stderr)
    backbone = StandaloneVL(
        args.model_id,
        device=device,
        trust_remote_code=args.trust_remote_code,
        siglip_fixed_padding=args.siglip_fixed_padding,
    )
    print(
        f"[info] model={type(backbone.model).__name__}  processor={type(backbone.processor).__name__}",
        file=sys.stderr,
    )

    for ep in eval_parts:
        df = df_full
        if ep is not None and args.split_col in df.columns:
            df = df[df[args.split_col].astype(str) == ep]
        if ss_filter is not None and args.species_split_col in df.columns:
            df = df[df[args.species_split_col].astype(str).isin(ss_filter)]
        if len(df) == 0:
            print(f"!! no rows for split={ep!r}", file=sys.stderr)
            continue

        ss_part = ",".join(ss_filter) if ss_filter else "*"
        ep_part = ep if ep else "*"
        print(
            f"\n=== Standalone zero-shot  species_split=[{ss_part}] csv_split=[{ep_part}] "
            f"model={args.model_id} ===",
        )
        _eval_one_split(
            backbone,
            df.reset_index(drop=True),
            images_root=args.images_root,
            class_col=class_col,
            prompt_col=args.prompt_col,
            scientific_name_col=args.scientific_name_col,
            topk=args.topk,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
