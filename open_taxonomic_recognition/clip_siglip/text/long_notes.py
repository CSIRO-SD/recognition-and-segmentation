"""
Three practical ways to feed long ``notes`` (+ metadata) into CLIP / SigLIP-style encoders:

1. **priority** — concat segments in fixed order (label → geo → notes); fill tokens left-to-right; truncate the tail segment only when needed.
2. **chunk_mean** — split the long part into chunks, encode each, L2-mean pool (more compute).
3. **summarize** — optional hook: you supply an LLM call; we only reserve the interface.

All respect a **max token** budget (CLIP ~77, SigLIP often ~64 — pass your processor's value).
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Sequence

import torch

try:
    from transformers import PreTrainedTokenizerBase
except ImportError:
    PreTrainedTokenizerBase = None  # type: ignore


class LongNoteStrategy(str, Enum):
    priority = "priority"
    chunk_mean = "chunk_mean"
    summarize = "summarize"


def _encode_len(tokenizer: "PreTrainedTokenizerBase", text: str, add_special_tokens: bool = True) -> int:
    t = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return len(t)


def fit_priority_text(
    parts: Sequence[str],
    tokenizer: "PreTrainedTokenizerBase",
    max_length: int,
    sep: str = "\n",
) -> str:
    """
    Keep ``parts[0]`` highest priority — typical order:

    ``[short_label_prompt, locality_line, notes_clean]``

    Greedy: append whole segments while they fit; on first overflow, **truncate only the current segment**
    (by character binary search) to use the remaining token budget, then stop.

    ``max_length`` is the **subword count you allow for this string** (exclude BOS/EOS if you count them
    in the model forward — for CLIP 77 total, pass e.g. **75** here when building raw text, or pass 77
    if your tokenizer already matches the processor's ``max_length``).
    """
    accum: list[str] = []
    for p in parts:
        s = (p or "").strip()
        if not s:
            continue
        candidate = sep.join(accum + [s]) if accum else s
        n = len(tokenizer.encode(candidate, add_special_tokens=False))
        if n <= max_length:
            accum.append(s)
            continue
        prefix = (sep.join(accum) + sep) if accum else ""
        base_len = len(tokenizer.encode(prefix, add_special_tokens=False)) if prefix else 0
        remain = max_length - base_len
        if remain <= 0:
            break
        lo, hi = 0, len(s)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            sub = s[:mid]
            full = prefix + sub
            if len(tokenizer.encode(full, add_special_tokens=False)) <= max_length:
                best = sub
                lo = mid + 1
            else:
                hi = mid - 1
        if best:
            accum.append(best)
        break
    return sep.join(accum).strip()


def split_into_chunks(text: str, max_chars: int = 400, overlap: int = 0) -> list[str]:
    """Slice by character count; optional overlap for smoother mean (small overlap often enough)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    step = max(1, max_chars - overlap)
    for i in range(0, len(text), step):
        chunks.append(text[i : i + max_chars])
    return chunks


@torch.no_grad()
def chunked_mean_embed(
    texts: list[str],
    encode_fn: Callable[[list[str]], torch.Tensor],
    max_chars: int = 400,
    overlap: int = 0,
) -> torch.Tensor:
    """
    Encode each long field by chunking, then **mean of L2-normalized** chunk embeddings (per row).

    ``encode_fn`` maps a batch of strings to a ``[B, D]`` tensor (your ``get_text_features`` wrapper).
    For a single logical row with K chunks we average in embedding space after normalizing chunks.

    Parameters
    ----------
    texts
        One row per sample; internally expanded to multiple chunks → multiple forwards.
    encode_fn
        Must accept list[str], return FloatTensor [len(batch), D].
    """
    device = None
    out_rows: list[torch.Tensor] = []
    for t in texts:
        ch = split_into_chunks(t, max_chars=max_chars, overlap=overlap)
        if not ch:
            emb = encode_fn([" "])
            out_rows.append(emb[0])
            continue
        e = encode_fn(ch)
        if device is None:
            device = e.device
        e = torch.nn.functional.normalize(e.float(), dim=-1)
        out_rows.append(e.mean(dim=0))
    stacked = torch.stack(out_rows, dim=0)
    return torch.nn.functional.normalize(stacked, dim=-1)


def summarize_placeholder(raw: str, summarizer: Callable[[str], str] | None = None, max_chars: int = 500) -> str:
    """
    Strategy **summarize**: plug in ``summarizer`` (e.g. API to an LLM) or fall back to head+tail truncation.

    For a real baseline without an LLM, we keep first/last chars (cheap heuristic — replace with LLM later).
    """
    if summarizer is not None:
        return summarizer(raw)
    if len(raw) <= max_chars:
        return raw
    half = max_chars // 2
    return raw[:half] + "\n[...]\n" + raw[-half:]
