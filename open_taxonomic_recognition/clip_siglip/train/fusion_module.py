"""Trainable late-fusion head: two scalar logits per class (image / metadata) → one logit (shared MLP)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """``MLP: ℝ² → ℝ`` applied identically to each class (batched as ``[B*K, 2]``)."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        h = max(1, int(hidden_dim))
        self.net = nn.Sequential(nn.Linear(2, h), nn.ReLU(inplace=True), nn.Linear(h, 1))

    def forward(self, pair_logits: torch.Tensor) -> torch.Tensor:
        """``pair_logits`` shape ``[N, 2]`` → ``[N]``."""
        return self.net(pair_logits).squeeze(-1)


def fuse_branches(
    logits_image: torch.Tensor,
    logits_meta: torch.Tensor,
    fusion: FusionHead,
    *,
    mode: str = "both",
) -> torch.Tensor:
    """``logits_*``: ``[B, K]``. Returns fused ``[B, K]``.

    ``mode``: ``both`` | ``image_only`` | ``meta_only`` (ablations; zero the other branch *before* MLP).
    """
    if mode not in ("both", "image_only", "meta_only"):
        raise ValueError(mode)
    li = logits_image
    lm = logits_meta
    if mode == "image_only":
        lm = torch.zeros_like(lm)
    elif mode == "meta_only":
        li = torch.zeros_like(li)
    stacked = torch.stack([li, lm], dim=-1)
    b, k, _ = stacked.shape
    flat = stacked.reshape(b * k, 2)
    out = fusion(flat).view(b, k)
    return out


def adjust_logits_if_siglip(logits: torch.Tensor, raw_model: torch.nn.Module) -> torch.Tensor:
    """Apply SigLIP / SigLIP2 ``logit_scale`` and ``logit_bias`` when present (HF ``SiglipModel``)."""
    ls = getattr(raw_model, "logit_scale", None)
    lb = getattr(raw_model, "logit_bias", None)
    if ls is not None and lb is not None:
        return logits * ls.exp() + lb
    return logits


def pooled_meta_embedding(
    backbone,
    segments_batch: list[tuple[str, str, str, str]],
) -> torch.Tensor:
    """Four metadata chunks per sample → mean of text embeddings, L2-normalized (``[B, D]``)."""
    flat: list[str] = []
    for a, b, c, d in segments_batch:
        flat.extend([a, b, c, d])
    t = backbone.encode_text(flat)
    bsz = len(segments_batch)
    t = t.view(bsz, 4, -1).mean(dim=1)
    return torch.nn.functional.normalize(t.float(), dim=-1)


def fused_class_logits(
    image_emb: torch.Tensor,
    meta_emb: torch.Tensor,
    prototypes: torch.Tensor,
    raw_model: torch.nn.Module,
    fusion: FusionHead,
    *,
    mode: str = "both",
) -> torch.Tensor:
    """``image_emb``, ``meta_emb``: ``[B, D]``; ``prototypes``: ``[K, D]`` (L2-normalized)."""
    li = adjust_logits_if_siglip(image_emb @ prototypes.T, raw_model)
    lm = adjust_logits_if_siglip(meta_emb @ prototypes.T, raw_model)
    return fuse_branches(li, lm, fusion, mode=mode)
