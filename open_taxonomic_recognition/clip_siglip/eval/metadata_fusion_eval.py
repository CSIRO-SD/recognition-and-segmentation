"""Eval with frozen VL backbone + trained metadata/image late-fusion head."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_siglip.eval.zeroshot import (  # noqa: F401
    ZeroShotMetrics,
    _collate_indices,
    _norm_cell,
    _norm_rank_field,
    _print_order_hierarchy_debug,
    _PilIndexDataset,
    _write_order_hierarchy_tsv,
)
from clip_siglip.models.wrappers import VLBackbone
from clip_siglip.train.fusion_module import FusionHead, fused_class_logits, pooled_meta_embedding


@torch.no_grad()
def fusion_metrics_pil(
    backbone: VLBackbone,
    raw_model: nn.Module,
    fusion: FusionHead,
    *,
    prototypes: torch.Tensor,
    labels_sorted: list[str],
    image_paths: list[Path],
    weak_context_texts: list[str],
    hierarchy_by_class: Mapping[str, Mapping[str, str]],
    gt_samples: Sequence[Mapping[str, Any]],
    batch_size: int = 32,
    num_workers: int = 4,
    topk: int = 5,
    fusion_mode: str = "both",
    top1_class_label: str = "taxon_key",
    hierarchy_report_prefix: Path | None = None,
    hierarchy_debug_stderr: bool = False,
    key_to_idx: dict[str, int] | None = None,
) -> tuple[ZeroShotMetrics, dict[str, int]]:
    """``prototypes``: ``[K, D]`` on ``raw_model.device``, rows aligned with ``labels_sorted``.

    If ``key_to_idx`` is set, rows whose ``eval_class`` is missing from it are **skipped** (OOD w.r.t.
    train-closed vocabulary). Returns ``(metrics, stats)`` where ``stats`` includes ``n_skipped_oov``.
    """
    assert len(image_paths) == len(gt_samples) == len(weak_context_texts)
    from clip_siglip.data.weak_context_split import split_weak_context_four

    fusion.eval()
    device = next(raw_model.parameters()).device
    protos = prototypes.float().to(device)
    K = len(labels_sorted)
    k_eff = min(max(1, topk), K)

    # Resolve which samples have in-vocab GT
    k2i = key_to_idx
    if k2i is None:
        k2i = {lab: i for i, lab in enumerate(labels_sorted)}

    active_indices: list[int] = []
    for i, row in enumerate(gt_samples):
        g = _norm_cell(row.get("eval_class", ""))
        if g in k2i:
            active_indices.append(i)

    n_skipped = len(image_paths) - len(active_indices)
    stats = {"n_skipped_oov": n_skipped, "n_evaluated": len(active_indices)}

    ds_paths = [image_paths[i] for i in active_indices]
    ds_weak = [weak_context_texts[i] for i in active_indices]
    ds_gt = [gt_samples[i] for i in active_indices]

    ds = _PilIndexDataset(ds_paths)
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
    n_parse_err = 0

    for pil_batch, idxs in tqdm(dl, desc=f"fusion-{fusion_mode}"):
        segs: list[tuple[str, str, str, str]] = []
        for j in idxs:
            try:
                segs.append(split_weak_context_four(ds_weak[j]))
            except ValueError:
                segs.append(("", "", "", ""))
                n_parse_err += 1

        im = backbone.encode_images_pil(pil_batch)
        meta = pooled_meta_embedding(backbone, segs)
        logits = fused_class_logits(im, meta, protos, raw_model, fusion, mode=fusion_mode)
        pred_idx = logits.argmax(dim=-1).cpu().tolist()
        _, idx = logits.topk(k_eff, dim=-1)
        topk_idx = idx.cpu().tolist()

        for j, jg in enumerate(idxs):
            gt_row = ds_gt[jg]
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

    if n_parse_err:
        print(f"[fusion-eval] weak_context parse fallbacks used: {n_parse_err}", file=sys.stderr)
    if n_skipped:
        print(f"[fusion-eval] skipped {n_skipped} samples with OOV eval_class (w.r.t. train vocab)", file=sys.stderr)

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

    stats["n_parse_err"] = n_parse_err
    return (
        ZeroShotMetrics(
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
        ),
        stats,
    )
