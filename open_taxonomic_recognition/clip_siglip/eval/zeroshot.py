"""Zero-shot: cosine image–text prototypes per class (taxon_key and/or label_text)."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clip_siglip.models.wrappers import VLBackbone
from clip_siglip.data.weak_context_split import split_weak_context_four
from clip_siglip.train.fusion_module import adjust_logits_if_siglip, pooled_meta_embedding


def default_prompt_templates(taxon_key: str, scientific_name: str | None) -> list[str]:
    t = scientific_name.strip() if scientific_name else taxon_key.replace("__", " ")
    return [
        f"a cropped photo of {t}",
        f"a specimen of taxon {t}",
        f"{t}",
    ]


def label_text_prompt_templates(label_text: str) -> list[str]:
    """Prompts built from CSV ``label_text`` (Order/Family/Genus/Species sentence)."""
    t = label_text.strip()
    if not t:
        return [""]
    return [
        t,
        f"a museum specimen described as: {t}",
        f"a pinned insect specimen. {t}",
    ]


def _norm_cell(x: Any) -> str:
    if x is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(x):
            return ""
    except Exception:
        pass
    if isinstance(x, float):
        import math

        if math.isnan(x):
            return ""
    return str(x).strip()


_MISSING_RANK_TOKENS = frozenset(
    {
        "unknown",
        "unk",
        "n/a",
        "na",
        "none",
        "-",
        "unclassified",
        "not classified",
        "missing",
        "null",
        "no data",
        "nodata",
    }
)


def _norm_rank_field(x: Any) -> str:
    """Normalize a single taxonomic rank cell (CSV or hierarchy dict).

    Empty, NaN, and common placeholders (e.g. ``unknown``, ``n/a``) map to ``""`` so they match
    across ``label_text``-style wording vs sparse CSV columns.
    """
    s = _norm_cell(x)
    if not s:
        return ""
    if s.lower() in _MISSING_RANK_TOKENS:
        return ""
    return s


def build_class_prompts(
    df,
    class_col: str = "taxon_key",
    scientific_name_col: str | None = "scientific_name",
    prompt_col: str | None = None,
) -> dict[str, list[str]]:
    """One entry per distinct **class** in ``df`` (``class_col``), list of prompt variants (averaged later).

    Keys are ``_norm_cell`` values of ``class_col`` so they match per-row GT strings.

    If ``prompt_col`` is set (e.g. ``label_text``), prompts come from that column within each class
    group. When ``class_col == prompt_col`` (typical: eval and prompts both use ``label_text``),
    each group has a single caption; empty captions map to ``label_text_prompt_templates`` (``['']``).
    When ``class_col`` is ``taxon_key`` but ``prompt_col`` is ``label_text``, multiple captions per
    taxon trigger a warning and one representative is kept.
    """
    import pandas as pd

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
                if class_col == (prompt_col or ""):
                    out[key] = label_text_prompt_templates(key)
                else:
                    row0 = g.iloc[0]
                    sci = None
                    if has_sci:
                        v = row0[scientific_name_col]
                        if pd.notna(v) and str(v).strip():
                            sci = str(v).strip()
                    out[key] = default_prompt_templates(key, sci)
                continue

            uniq = raw.unique()
            if len(uniq) > 1:
                chosen = sorted(uniq, key=lambda s: (-len(s), s))[0]
                print(
                    f"[WARN] class_col={class_col!r} key={key!r}: multiple {prompt_col} variants ({len(uniq)}); "
                    f"using one representative (len={len(chosen)}).",
                )
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


def build_hierarchy_by_class_col(df, group_col: str = "taxon_key") -> dict[str, dict[str, str]]:
    """One hierarchy dict per distinct normalized ``group_col`` value (first row of each group)."""
    cols = ["order", "family", "genus", "species"]
    present = [c for c in cols if c in df.columns]
    out: dict[str, dict[str, str]] = {}
    work = df.copy()
    work["__g"] = work[group_col].map(_norm_cell)
    for _, g in work.groupby("__g"):
        r = g.iloc[0]
        key = str(r["__g"])
        out[key] = {c: _norm_rank_field(r.get(c)) for c in present} if present else {}
    return out


def build_hierarchy_by_taxon(df, taxon_col: str = "taxon_key") -> dict[str, dict[str, str]]:
    """Backward-compatible alias for ``build_hierarchy_by_class_col(..., group_col=taxon_col)``."""
    return build_hierarchy_by_class_col(df, group_col=taxon_col)


class _PilIndexDataset(Dataset):
    """Yield ``(image, index)`` so GT / hierarchy come from a single per-sample record list."""

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        im = Image.open(self.paths[i]).convert("RGB")
        return im, i


class _PilDataset(Dataset):
    """Backward-compatible: ``(image, class_label)`` for callers that only need top-1 on labels."""

    def __init__(self, paths: list[Path], y: list[str]):
        assert len(paths) == len(y)
        self.paths = paths
        self.y = y

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        im = Image.open(self.paths[i]).convert("RGB")
        return im, self.y[i]


@dataclass
class ZeroShotMetrics:
    """Hierarchical metrics share denominator ``n_samples``. Family+ genus+ species are **conditional**:
    a lower rank counts as correct only if **all higher ranks** (pred vs GT) already match, to avoid
    rewarding homonyms under the wrong order/family/genus.
    """

    n_samples: int
    top1_acc: float
    topk_acc: float
    topk: int
    species_taxon_acc: float  # same as top-1 on class label (e.g. label_text or taxon_key)
    order_acc: float | None
    family_acc: float | None
    genus_acc: float | None
    species_name_acc: float | None  # CSV species column; same denom as other ranks
    n_order: int
    n_family: int
    n_genus: int
    n_species_col: int
    top1_class_label: str = "taxon_key"  # printed in metrics header (e.g. label_text)

    def lines(self) -> list[str]:
        lines = [
            f"  top-1 ({self.top1_class_label}):     {self.top1_acc:.6f}  (n={self.n_samples})",
            f"  top-{self.topk}:              {self.topk_acc:.6f}",
            f"  order acc:            {self._fmt(self.order_acc)}  (denom={self.n_order})",
            f"  family acc:           {self._fmt(self.family_acc)}  (denom={self.n_family}; requires order match)",
            f"  genus acc:            {self._fmt(self.genus_acc)}  (denom={self.n_genus}; requires order+family match)",
            f"  species (CSV column): {self._fmt(self.species_name_acc)}  (denom={self.n_species_col}; requires order+family+genus match; unknown/empty unified)",
        ]
        return lines

    @staticmethod
    def _fmt(x: float | None) -> str:
        if x is None:
            return "n/a"
        return f"{x:.6f}"


def _write_order_hierarchy_tsv(
    prefix: Path,
    o_tot: Counter[str],
    o_cor: Counter[str],
    o_conf: Counter[tuple[str, str]],
) -> tuple[Path, Path]:
    """Write ``{stem}_order_per_gt.tsv`` and ``{stem}_order_confusion_wrong.tsv`` under ``prefix.parent``."""
    prefix = Path(prefix)
    parent = prefix.parent
    stem = prefix.name
    parent.mkdir(parents=True, exist_ok=True)
    p_per = parent / f"{stem}_order_per_gt.tsv"
    p_conf = parent / f"{stem}_order_confusion_wrong.tsv"

    with p_per.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["gt_order", "n_samples", "n_order_correct", "order_acc"])
        for gt in sorted(o_tot.keys(), key=lambda k: (-o_tot[k], k)):
            nt = o_tot[gt]
            nc = int(o_cor[gt])
            w.writerow([gt, nt, nc, f"{nc / max(nt, 1):.8f}"])

    wrong_pairs = [(gt, pr, n) for (gt, pr), n in o_conf.items() if gt != pr]
    wrong_pairs.sort(key=lambda t: -t[2])
    with p_conf.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["gt_order", "pred_order", "n"])
        for gt, pr, n in wrong_pairs:
            w.writerow([gt, pr, n])

    return p_per, p_conf


def _print_order_hierarchy_debug(
    o_tot: Counter[str],
    o_cor: Counter[str],
    o_conf: Counter[tuple[str, str]],
    *,
    min_n: int = 40,
    worst_k: int = 15,
    top_wrong_pairs: int = 25,
) -> None:
    wrong = [(gt, pr, n) for (gt, pr), n in o_conf.items() if gt != pr]
    wrong.sort(key=lambda t: -t[2])
    print("[hierarchy-debug] top wrong order pairs (gt_order -> pred_order, count)", file=sys.stderr)
    for gt, pr, n in wrong[:top_wrong_pairs]:
        print(f"  {gt!r} -> {pr!r}\t{n}", file=sys.stderr)

    stats: list[tuple[float, int, int, str]] = []
    for gt, nt in o_tot.items():
        if nt < min_n:
            continue
        nc = int(o_cor[gt])
        stats.append((nc / max(nt, 1), nc, nt, gt))
    stats.sort(key=lambda x: (x[0], -x[2]))
    print(
        f"[hierarchy-debug] lowest per-gt_order accuracy (need at least min_n={min_n} samples)",
        file=sys.stderr,
    )
    for acc, nc, nt, gt in stats[:worst_k]:
        print(f"  acc={acc:.4f}\t{nc}/{nt}\tgt_order={gt!r}", file=sys.stderr)

    n_empty_gt = o_tot.get("", 0)
    if n_empty_gt > 0:
        print(
            f"[hierarchy-debug] note: {n_empty_gt} rows have empty GT order after rank normalization; "
            "order match then requires pred order empty too.",
            file=sys.stderr,
        )


@torch.no_grad()
def zeroshot_metrics_pil(
    backbone: VLBackbone,
    class_prompts_by_taxon: dict[str, Sequence[str]],
    image_paths: list[Path],
    hierarchy_by_class: Mapping[str, Mapping[str, str]],
    gt_samples: Sequence[Mapping[str, Any]],
    batch_size: int = 32,
    num_workers: int = 4,
    topk: int = 5,
    *,
    top1_class_label: str = "taxon_key",
    hierarchy_report_prefix: Path | None = None,
    hierarchy_debug_stderr: bool = False,
    weak_context_texts: list[str] | None = None,
    weak_fusion_alpha: float = 0.5,
) -> ZeroShotMetrics:
    """Full metrics from one list ``gt_samples`` (each row needs ``eval_class`` + rank columns).

    Top-1 / top-k compare ``eval_class`` to predicted prototype key (must match ``class_prompts_by_taxon``).
    Ranks use ``_norm_rank_field`` (unknown/empty unified). **Family/genus/species** acc are conditional:
    family must match only if order matches; genus only if order and family match; species only if
    order, family, and genus match (same denominator ``n_samples`` each).

    ``hierarchy_report_prefix``: write ``{stem}_order_per_gt.tsv`` and ``{stem}_order_confusion_wrong.tsv``.
    ``hierarchy_debug_stderr``: print top wrong (gt,pred) pairs and weakest per-gt_order acc to stderr.

    If ``weak_context_texts`` is set (same length as ``image_paths``), each sample's metadata string is split
    into four chunks (Latitude / Sampling event / Notes), encoded with the same text tower, mean-pooled,
    then fused with image logits: ``(1-α)*meta_branch + α*image_branch`` after SigLIP ``logit_scale`` /
    ``logit_bias`` when present (``weak_fusion_alpha`` ∈ [0,1], weight on **image**). ``α=1`` skips
    metadata encoding (identical to no weak column).
    """
    assert len(image_paths) == len(gt_samples)
    if weak_context_texts is not None:
        if len(weak_context_texts) != len(image_paths):
            raise ValueError("weak_context_texts must match image_paths length")
        a = float(weak_fusion_alpha)
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"weak_fusion_alpha must be in [0,1], got {a}")

    labels_sorted = sorted(class_prompts_by_taxon.keys())
    n_cls = len(labels_sorted)
    k_eff = min(max(1, topk), n_cls)

    text_embs: list[torch.Tensor] = []
    for lab in labels_sorted:
        prompts = [str(x) for x in class_prompts_by_taxon[lab]]
        txs = backbone.encode_text(prompts)
        proto = txs.mean(dim=0)
        text_embs.append(proto)
    prototypes = torch.stack(text_embs, dim=0)
    prototypes = torch.nn.functional.normalize(prototypes.float(), dim=-1)

    ds = _PilIndexDataset(image_paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_indices)

    top1_correct = 0
    topk_correct = 0
    o_cor, o_den = 0, 0
    f_cor, f_den = 0, 0
    g_cor, g_den = 0, 0
    sp_cor, sp_den = 0, 0
    n_total = 0
    o_tot_by_gt: Counter[str] = Counter()
    o_cor_by_gt: Counter[str] = Counter()
    o_ord_confusion: Counter[tuple[str, str]] = Counter()

    raw_model = backbone.model
    use_weak = weak_context_texts is not None and float(weak_fusion_alpha) < 1.0 - 1e-6
    alpha_img = float(weak_fusion_alpha) if weak_context_texts is not None else 1.0
    meta_only = weak_context_texts is not None and float(weak_fusion_alpha) <= 1e-6

    if meta_only:
        desc = "zero-shot+weak(meta_only)"
    elif use_weak:
        desc = "zero-shot+weak"
    else:
        desc = "zero-shot"

    for pil_batch, idxs in tqdm(dl, desc=desc):
        build_weak = weak_context_texts is not None and (use_weak or meta_only)
        segs: list[tuple[str, str, str, str]] | None = None
        if build_weak:
            segs = []
            for j in idxs:
                w = weak_context_texts[j]
                try:
                    segs.append(split_weak_context_four(w))
                except ValueError:
                    segs.append(("", "", "", ""))

        if meta_only:
            assert segs is not None
            meta = pooled_meta_embedding(backbone, segs)
            logits = adjust_logits_if_siglip(meta @ prototypes.T, raw_model)
        elif use_weak:
            assert segs is not None
            im = backbone.encode_images_pil(pil_batch)
            meta = pooled_meta_embedding(backbone, segs)
            li = adjust_logits_if_siglip(im @ prototypes.T, raw_model)
            lm = adjust_logits_if_siglip(meta @ prototypes.T, raw_model)
            logits = alpha_img * li + (1.0 - alpha_img) * lm
        else:
            im = backbone.encode_images_pil(pil_batch)
            logits = im @ prototypes.T
        pred_idx = logits.argmax(dim=-1).cpu().tolist()
        vals, idx = logits.topk(k_eff, dim=-1)
        _ = vals
        topk_idx = idx.cpu().tolist()

        for j, i_samp in enumerate(idxs):
            # ``i_samp`` is already the global index in ``image_paths`` / ``gt_samples`` (from dataset).
            gt_row = gt_samples[i_samp]
            g = _norm_cell(gt_row.get("eval_class", ""))
            gt_ord = _norm_rank_field(gt_row.get("order"))
            gt_fam = _norm_rank_field(gt_row.get("family"))
            gt_gen = _norm_rank_field(gt_row.get("genus"))
            gt_spc = _norm_rank_field(gt_row.get("species"))

            p_tk = labels_sorted[pred_idx[j]]
            ph = hierarchy_by_class.get(p_tk, {})
            p_ord = _norm_rank_field(ph.get("order", ""))
            p_fam = _norm_rank_field(ph.get("family", ""))
            p_gen = _norm_rank_field(ph.get("genus", ""))
            p_spc = _norm_rank_field(ph.get("species", ""))

            if p_tk == g:
                top1_correct += 1
            top_set = {labels_sorted[topk_idx[j][kk]] for kk in range(k_eff)}
            if g in top_set:
                topk_correct += 1

            o_den += 1
            o_tot_by_gt[gt_ord] += 1
            if p_ord == gt_ord:
                o_cor += 1
                o_cor_by_gt[gt_ord] += 1
            o_ord_confusion[(gt_ord, p_ord)] += 1

            f_den += 1
            if p_ord == gt_ord and p_fam == gt_fam:
                f_cor += 1

            g_den += 1
            if p_ord == gt_ord and p_fam == gt_fam and p_gen == gt_gen:
                g_cor += 1

            sp_den += 1
            if p_ord == gt_ord and p_fam == gt_fam and p_gen == gt_gen and p_spc == gt_spc:
                sp_cor += 1

            n_total += 1

    if hierarchy_report_prefix is not None:
        p_gt, p_wrong = _write_order_hierarchy_tsv(
            hierarchy_report_prefix,
            o_tot_by_gt,
            o_cor_by_gt,
            o_ord_confusion,
        )
        print(f"[hierarchy] wrote {p_gt}\n[hierarchy] wrote {p_wrong}", file=sys.stderr)
    if hierarchy_debug_stderr:
        _print_order_hierarchy_debug(o_tot_by_gt, o_cor_by_gt, o_ord_confusion)

    def rate(num: int, den: int) -> float | None:
        if den <= 0:
            return None
        return num / den
    # import pdb; pdb.set_trace()
    return ZeroShotMetrics(
        n_samples=n_total,
        top1_acc=top1_correct / max(n_total, 1),
        topk_acc=topk_correct / max(n_total, 1),
        topk=k_eff,
        species_taxon_acc=top1_correct / max(n_total, 1),
        order_acc=rate(o_cor, o_den),
        family_acc=rate(f_cor, f_den),
        genus_acc=rate(g_cor, g_den),
        species_name_acc=rate(sp_cor, sp_den),
        n_order=o_den,
        n_family=f_den,
        n_genus=g_den,
        n_species_col=sp_den,
        top1_class_label=top1_class_label,
    )


@torch.no_grad()
def zeroshot_top1_pil(
    backbone: VLBackbone,
    class_prompts_by_class: dict[str, Sequence[str]],
    image_paths: list[Path],
    ground_truth_class: list[str],
    batch_size: int = 32,
    num_workers: int = 4,
) -> float:
    """Backward-compatible: micro average top-1 vs ``ground_truth_class`` keys (must match prompt dict)."""
    labels_sorted = sorted(class_prompts_by_class.keys())

    text_embs: list[torch.Tensor] = []
    for lab in labels_sorted:
        prompts = [str(x) for x in class_prompts_by_class[lab]]
        txs = backbone.encode_text(prompts)
        proto = txs.mean(dim=0)
        text_embs.append(proto)
    prototypes = torch.stack(text_embs, dim=0)
    prototypes = torch.nn.functional.normalize(prototypes.float(), dim=-1)

    ds = _PilDataset(image_paths, ground_truth_class)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)

    correct = 0
    n = 0
    for pil_batch, ys in tqdm(dl, desc="zero-shot"):
        im = backbone.encode_images_pil(pil_batch)
        logits = im @ prototypes.T
        pred_idx = logits.argmax(dim=-1).cpu().tolist()
        for j, g in enumerate(ys):
            if labels_sorted[pred_idx[j]] == g:
                correct += 1
            n += 1

    return correct / max(n, 1)


def _collate(batch):
    ims, ys = zip(*batch)
    return list(ims), list(ys)


def _collate_indices(batch):
    ims, idxs = zip(*batch)
    return list(ims), list(idxs)
