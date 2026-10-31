"""Load pretrained vision–language twins with a single inference API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import inspect
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer


# Keys SigLIP2 vision expects on ``get_image_features`` (see HF model card / modeling_siglip2).
_SIGLIP2_VISION_KEYS = frozenset({"pixel_values", "pixel_attention_mask", "spatial_shapes"})


def _needs_standalone_clip_image_processor(processor: Any) -> bool:
    """Hub EvaCLIP ports often return ``CLIPTokenizer`` from ``AutoProcessor`` — images need ``CLIPImageProcessor``."""
    if isinstance(processor, CLIPProcessor):
        return False
    name = type(processor).__name__
    return name == "CLIPTokenizer" or name.endswith("CLIPTokenizerFast")


def _clip_image_processor_from_hub_or_config(model_name: str, model: torch.nn.Module, load_kw: dict[str, Any]) -> CLIPImageProcessor:
    """``BAAI/EVA-CLIP-8B`` ships tokenizer + weights but no ``preprocessor_config.json``; fall back to CLIP ViT sizing."""
    try:
        return CLIPImageProcessor.from_pretrained(model_name, **load_kw)
    except OSError:
        vc = getattr(getattr(model, "config", None), "vision_config", None)
        sz = int(getattr(vc, "image_size", 224) or 224)
        return CLIPImageProcessor(
            size={"shortest_edge": sz},
            crop_size={"height": sz, "width": sz},
        )


def _filter_inputs_for_get_image_features(model: torch.nn.Module, inputs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Map processor outputs to ``get_image_features`` kwargs (SigLIP2 needs mask + ``spatial_shapes``)."""
    sig = inspect.signature(model.get_image_features)
    explicit = {
        n
        for n, p in sig.parameters.items()
        if n != "self" and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    out: dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k in explicit:
            out[k] = v.to(device)
        elif var_kw and k in _SIGLIP2_VISION_KEYS:
            out[k] = v.to(device)
    if "pixel_values" not in out:
        raise ValueError("Processor output missing pixel_values for get_image_features")
    return out


def _filter_inputs_for_encode_text(model: torch.nn.Module, inputs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Keep only kwargs accepted by ``encode_text`` (EvaCLIP).

    Never forwards ``position_ids`` from the processor: Hub checkpoints (e.g. BAAI/EVA-CLIP-8B) often
    set tokenizer ``model_max_length`` to an enormous value, so sequences can exceed
    ``text_config.max_position_embeddings``; external ``position_ids`` would then index past the text
    position embedding table. The model builds safe positions when ``position_ids`` is omitted.
    """
    if not hasattr(model, "encode_text"):
        return {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    sig = inspect.signature(model.encode_text)
    explicit = {
        n
        for n, p in sig.parameters.items()
        if n != "self" and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    out: dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if k == "position_ids":
            continue
        if not isinstance(v, torch.Tensor):
            continue
        if k in explicit:
            out[k] = v.to(device)
        elif var_kw:
            out[k] = v.to(device)
    return out


def _text_max_length_from_model(model: torch.nn.Module) -> int | None:
    """Context length for CLIP-style text towers (processor/tokenizer ``model_max_length`` is often wrong on Hub)."""
    seen: list[Any] = []
    cfg = getattr(model, "config", None)
    if cfg is not None:
        seen.append(getattr(cfg, "text_config", None))
    tm = getattr(model, "text_model", None)
    if tm is not None:
        seen.append(getattr(tm, "config", None))

    for tc in seen:
        if tc is None:
            continue
        if isinstance(tc, dict):
            m = tc.get("max_position_embeddings")
        else:
            m = getattr(tc, "max_position_embeddings", None)
        if m is not None:
            return int(m)
    # BAAI EvaCLIP on HF uses the same 77-token text context as OpenAI CLIP.
    if type(model).__name__ == "EvaCLIPModel":
        return 77
    return None


def _truncate_text_batch(enc: dict[str, Any], max_len: int) -> None:
    """Hard cap sequence length after tokenization (some tokenizers ignore ``max_length`` when ``model_max_length`` is broken)."""
    ids = enc.get("input_ids")
    if not isinstance(ids, torch.Tensor) or ids.dim() != 2:
        return
    if max_len <= 0:
        return
    if ids.size(-1) <= max_len:
        return
    mask = enc.get("attention_mask")
    enc["input_ids"] = ids[:, :max_len].contiguous()
    if isinstance(mask, torch.Tensor) and mask.dim() == 2 and mask.size(0) == ids.size(0) and mask.size(-1) == ids.size(-1):
        enc["attention_mask"] = mask[:, :max_len].contiguous()


def _clip_loaded_text_limits(model: torch.nn.Module) -> tuple[int | None, int | None, int]:
    """``(max_seq, token_rows, pad_id)`` from **loaded** ``text_model`` weights (config can disagree with checkpoints)."""
    tm = getattr(model, "text_model", None)
    if tm is None:
        return None, None, 0
    seq_emb = getattr(getattr(tm, "embeddings", None), "position_embedding", None)
    tok_emb = getattr(getattr(tm, "embeddings", None), "token_embedding", None)
    max_seq = int(seq_emb.num_embeddings) if isinstance(seq_emb, torch.nn.Embedding) else None
    n_tok = int(tok_emb.num_embeddings) if isinstance(tok_emb, torch.nn.Embedding) else None
    cfg = getattr(tm, "config", None)
    pad = int(getattr(cfg, "pad_token_id", 0) or 0) if cfg is not None else 0
    if n_tok is not None and (pad >= n_tok or pad < 0):
        pad = 0
    return max_seq, n_tok, pad


def _clip_sanitize_token_ids(
    enc: dict[str, Any],
    *,
    num_token_rows: int | None,
    pad_token_id: int,
) -> None:
    """Replace OOB token ids using **runtime** embedding size (index must be in ``[0, num_token_rows)``)."""
    ids = enc.get("input_ids")
    if not isinstance(ids, torch.Tensor) or num_token_rows is None or num_token_rows <= 0:
        return
    ids = ids.long()
    enc["input_ids"] = ids
    bad = (ids >= num_token_rows) | (ids < 0)
    if bad.any():
        p = int(pad_token_id)
        if p >= num_token_rows or p < 0:
            p = 0
        enc["input_ids"] = torch.where(bad, torch.full_like(ids, p), ids)


def _repair_evaclip_style_position_id_buffers(model: torch.nn.Module) -> None:
    """Rebuild ``position_ids`` on towers whose buffers look uninitialized.

    BAAI EvaCLIP Hub ports use ``register_buffer(..., persistent=False)`` for text/vision positional ids.
    After ``from_pretrained`` / weight load, those tensors can be **garbled** on some Transformers+PyTorch
    combinations, so ``position_embedding(position_ids)`` hits CUDA indexSelect asserts.

    **Only** ``EvaCLIPModel``: other ``AutoModel`` checkpoints (SigLIP2, MetaCLIP, …) may use non-(1, n)
    buffer layouts that are still valid — running this repair there could silently change embeddings and eval.
    """
    if type(model).__name__ != "EvaCLIPModel":
        return
    for tower_name in ("text_model", "vision_model"):
        tower = getattr(model, tower_name, None)
        if tower is None:
            continue
        em = getattr(tower, "embeddings", None)
        if em is None:
            continue
        pe = getattr(em, "position_embedding", None)
        buf = getattr(em, "position_ids", None)
        if not isinstance(pe, torch.nn.Embedding) or not isinstance(buf, torch.Tensor):
            continue
        n = int(pe.num_embeddings)
        if n <= 0 or buf.numel() == 0:
            continue
        bad_shape = tuple(buf.shape) != (1, n)
        bad_idx = bool((buf < 0).any()) or bool((buf >= n).any())
        if not (bad_shape or bad_idx):
            continue
        dev = pe.weight.device
        fixed = torch.arange(n, device=dev, dtype=torch.long).unsqueeze(0)
        em.register_buffer("position_ids", fixed, persistent=False)


def _auto_image_features(
    model: torch.nn.Module,
    inputs: dict,
    device: torch.device,
    *,
    pixel_values_only: torch.Tensor | None = None,
) -> torch.Tensor:
    """``get_image_features`` (SigLIP/CLIP) or ``encode_image`` (EvaCLIP)."""
    if pixel_values_only is not None:
        pv = pixel_values_only.to(device)
        if hasattr(model, "get_image_features"):
            return model.get_image_features(pixel_values=pv)
        if hasattr(model, "encode_image"):
            return model.encode_image(pixel_values=pv)
        raise TypeError(f"{type(model).__name__}: no get_image_features or encode_image")

    if hasattr(model, "get_image_features"):
        batch = _filter_inputs_for_get_image_features(model, inputs, device)
        return model.get_image_features(**batch)
    if hasattr(model, "encode_image"):
        if "pixel_values" not in inputs:
            raise ValueError("Processor output missing pixel_values for encode_image")
        return model.encode_image(pixel_values=inputs["pixel_values"].to(device))
    raise TypeError(f"{type(model).__name__}: no get_image_features or encode_image")


def _auto_text_features(model: torch.nn.Module, inputs: dict, device: torch.device) -> torch.Tensor:
    if hasattr(model, "get_text_features"):
        tens = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        return model.get_text_features(**tens)
    if hasattr(model, "encode_text"):
        enc = _filter_inputs_for_encode_text(model, inputs, device)
        # Remote EvaCLIP text tower: avoid AMP on embedding gathers (spurious device asserts on some setups).
        if device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=False):
                return model.encode_text(**enc)
        return model.encode_text(**enc)
    raise TypeError(f"{type(model).__name__}: no get_text_features or encode_text")


def _features_to_tensor(out: Any) -> torch.Tensor:
    """``get_{image,text}_features`` may return a tensor (older CLIP) or ``ModelOutput`` (SigLIP/SigLIP2)."""
    if isinstance(out, torch.Tensor):
        return out.float()
    pooler = getattr(out, "pooler_output", None)
    if pooler is not None:
        return pooler.float()
    last = getattr(out, "last_hidden_state", None)
    if last is not None:
        return last[:, 0, :].float()
    raise TypeError(f"Unexpected vision/text features type: {type(out)!r}")


class VLBackbone(ABC):
    """image -> emb [B,D]; text(list[str]) -> emb [T,D]; both L2-normalized."""

    device: torch.device

    @abstractmethod
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ...

    def encode_images_pil(self, images: list) -> torch.Tensor:
        """Prefer this for inference: PIL images batched via the model processor (correct resize/norm)."""
        images = [im.convert("RGB") if hasattr(im, "convert") else im for im in images]
        inputs = self.processor(images=list(images), return_tensors="pt", padding=True)  # type: ignore[attr-defined]
        pv = inputs["pixel_values"].to(self.device)
        return self.encode_image(pv)

    @abstractmethod
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        ...

    @staticmethod
    def l2norm(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)


class AutoHFVLWrapper(VLBackbone):
    """HF checkpoints via ``AutoModel`` / ``AutoProcessor`` (SigLIP, MetaCLIP, EvaCLIP, …).

    Vision/text encoding uses ``get_*_features`` when present (SigLIP/CLIP-style), else ``encode_image`` /
    ``encode_text`` (BAAI EvaCLIP remote code).
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device | None = None,
        trust_remote_code: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs: dict[str, Any] = {}
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        self.model = AutoModel.from_pretrained(model_name, **kwargs).to(self.device).eval()
        _repair_evaclip_style_position_id_buffers(self.model)
        self.processor = AutoProcessor.from_pretrained(model_name, **kwargs)
        self._clip_image_processor: CLIPImageProcessor | None = None
        if _needs_standalone_clip_image_processor(self.processor):
            self._clip_image_processor = _clip_image_processor_from_hub_or_config(model_name, self.model, kwargs)
        _has_img = hasattr(self.model, "get_image_features") or hasattr(self.model, "encode_image")
        _has_txt = hasattr(self.model, "get_text_features") or hasattr(self.model, "encode_text")
        if not _has_img or not _has_txt:
            raise TypeError(
                f"{model_name} ({type(self.model).__name__}): need get_image_features/encode_image and "
                f"get_text_features/encode_text for this repo.",
            )

    @torch.no_grad()
    def encode_images_pil(self, images: list) -> torch.Tensor:
        """Match HF examples: pass all vision fields from the processor (SigLIP2: mask + spatial_shapes)."""
        from PIL import Image

        images = [im.convert("RGB") if hasattr(im, "convert") else im for im in images]
        if self._clip_image_processor is not None:
            inputs = self._clip_image_processor(images=list(images), return_tensors="pt")
        else:
            inputs = self.processor(images=list(images), return_tensors="pt", padding=True)  # type: ignore[attr-defined]
        out = _auto_image_features(self.model, inputs, self.device)
        return self.l2norm(_features_to_tensor(out))

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Pixels only; omits SigLIP2 mask/shapes — prefer ``encode_images_pil``."""
        pixel_values = pixel_values.to(self.device)
        out = _auto_image_features(self.model, {}, self.device, pixel_values_only=pixel_values)
        return self.l2norm(_features_to_tensor(out))

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        # SigLIP/SigLIP2: official big_vision preprocessing always lowercases (`lower(key="text")`).
        # SigLIP1's SentencePiece tokenizer already lowercases internally → no-op for v1.
        # SigLIP2's Gemma tokenizer is case-sensitive; **without** this, prototypes for structured/template
        # text (e.g. ``"Order: X. Family: Y. ..."``) collapse into a narrow cone (training/inference mismatch).
        # See https://github.com/google-research/big_vision SigLIP2 demo colab: `lower(key="text")`.
        mt = str(getattr(getattr(self.model, "config", None), "model_type", "")).lower()
        if "siglip" in mt:
            texts = [t.lower() for t in texts]
        pos_cap, tok_rows, pad_id = _clip_loaded_text_limits(self.model)
        mlen = _text_max_length_from_model(self.model)
        if pos_cap is not None:
            mlen = pos_cap if mlen is None else min(mlen, pos_cap)
        proc_kw: dict[str, Any] = dict(return_tensors="pt", truncation=True)
        if mlen is not None and mlen > 0:
            proc_kw["max_length"] = mlen
            proc_kw["padding"] = "max_length"
        else:
            proc_kw["padding"] = True
        inputs = self.processor(text=texts, **proc_kw)  # type: ignore[misc]
        enc_in = dict(inputs)
        enc_in.pop("position_ids", None)
        if mlen is not None and mlen > 0:
            _truncate_text_batch(enc_in, mlen)
        if hasattr(self.model, "encode_text"):
            _clip_sanitize_token_ids(enc_in, num_token_rows=tok_rows, pad_token_id=pad_id)
            ids_chk = enc_in.get("input_ids")
            if isinstance(ids_chk, torch.Tensor) and tok_rows is not None:
                if int(ids_chk.max()) >= tok_rows or int(ids_chk.min()) < 0:
                    raise RuntimeError(
                        "EVA/CLIP text: token id out of range after sanitize — "
                        f"min={int(ids_chk.min())} max={int(ids_chk.max())} num_embeddings={tok_rows} "
                        f"shape={tuple(ids_chk.shape)}",
                    )
            if pos_cap is not None and isinstance(ids_chk, torch.Tensor):
                if ids_chk.shape[-1] > pos_cap:
                    raise RuntimeError(
                        f"EVA/CLIP text: seq_len={ids_chk.shape[-1]} > position cap={pos_cap}",
                    )
        out = _auto_text_features(self.model, enc_in, self.device)
        return self.l2norm(_features_to_tensor(out))


def _clip_processor_or_compose(model_name: str) -> CLIPProcessor:
    """``CLIPProcessor.from_pretrained`` requires ``preprocessor_config.json``; some Hub CLIPs (e.g. Apple DFN) ship tokenizer + ``config.json`` only."""
    try:
        return CLIPProcessor.from_pretrained(model_name)
    except OSError as e:
        err = str(e).lower()
        if "image processor" not in err and "preprocessor" not in err:
            raise
        tok = CLIPTokenizer.from_pretrained(model_name)
        cfg = CLIPConfig.from_pretrained(model_name)
        vc = cfg.vision_config
        sz = int(getattr(vc, "image_size", 224) or 224)
        ip = CLIPImageProcessor(
            size={"shortest_edge": sz},
            crop_size={"height": sz, "width": sz},
        )
        return CLIPProcessor(image_processor=ip, tokenizer=tok)


class CLIPWrapper(VLBackbone):
    """OpenAI-style CLIP on HF: ``CLIPModel`` + ``CLIPProcessor`` (e.g. ``openai/clip-vit-base-patch32``)."""

    def __init__(self, model_name: str, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = _clip_processor_or_compose(model_name)

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device)
        out = self.model.get_image_features(pixel_values)
        return self.l2norm(_features_to_tensor(out))

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        inp = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model.get_text_features(**inp)
        return self.l2norm(_features_to_tensor(out))


def build_vl(
    name_hint: str,
    *,
    trust_remote_code: bool = False,
    peft_adapter: str | Path | None = None,
) -> VLBackbone:
    """Pick loader by HF id heuristic.

    - **SigLIP / SigLIP2** (id contains ``siglip``): ``AutoModel`` + ``AutoProcessor`` (no remote code by default).
    - **MetaCLIP** (id contains ``metaclip``): ``AutoModel`` + ``AutoProcessor`` (covers MetaCLIP 2 / MetaClip2Model;
      v1 checkpoints on Hub are also resolved correctly by ``AutoModel``).
    - **EvaCLIP**-style (``evaclip`` in id, or both ``eva`` and ``clip``): ``AutoModel`` with
      ``trust_remote_code=True`` (required for common Hub ports such as ``BAAI/EVA-CLIP-*``).
    - **Otherwise** (typical ``openai/clip-*``, ``laion/CLIP-*``, **Apple DFN** ``apple/DFN*-CLIP-*`` in ``CLIPModel`` format): ``CLIPModel`` + ``CLIPProcessor`` (processor is composed from Hub tokenizer + ``CLIPConfig.vision_config.image_size`` when ``preprocessor_config.json`` is missing).

    EVA-CLIP Hub cards sometimes use a **different** image preprocessor than ``AutoProcessor``; if results look
    wrong, compare with the model card and adjust preprocessing.

    For any custom Hub repo that is **not** standard ``CLIPModel`` but exposes vision–text encoders
    (``get_*_features`` or EvaCLIP ``encode_image`` / ``encode_text``), use the **Auto** path
    (``siglip``, ``metaclip``, ``eva``+``clip``) with ``--trust-remote-code`` when the card requires it.

    ``peft_adapter``: directory from ``scripts/train_lora_vision.py``. Weights are **merged** into the base
    checkpoint for inference (no PEFT runtime dependency in ``encode_*``).
    """
    lh = name_hint.lower()

    use_auto = False
    trc = trust_remote_code

    if "siglip" in lh:
        use_auto = True
    elif "metaclip" in lh:
        # MetaCLIP 2 (MetaClip2Model) and other MetaCLIP Hub ports need AutoModel, not CLIPModel.
        use_auto = True
    elif "evaclip" in lh or ("eva" in lh and "clip" in lh):
        use_auto = True
        trc = True

    if use_auto:
        wrapper: VLBackbone = AutoHFVLWrapper(name_hint, trust_remote_code=trc)
    else:
        wrapper = CLIPWrapper(name_hint)

    if peft_adapter is not None:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("Loading a PEFT adapter requires `peft` (pip install peft).") from e
        m = PeftModel.from_pretrained(wrapper.model, str(peft_adapter))
        m.eval()
        wrapper.model = m.merge_and_unload()
        _repair_evaclip_style_position_id_buffers(wrapper.model)

    return wrapper


# Backwards alias
SigLIPWrapper = AutoHFVLWrapper
