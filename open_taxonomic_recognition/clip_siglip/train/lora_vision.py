"""Vision-tower-only LoRA for CLIP / SigLIP / SigLIP2 contrastive fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from clip_siglip.models.wrappers import _repair_evaclip_style_position_id_buffers


def _use_auto_model(name_hint: str, trust_remote_code: bool) -> tuple[bool, bool]:
    lh = name_hint.lower()
    trc = trust_remote_code
    if "siglip" in lh:
        return True, trc
    if "evaclip" in lh or ("eva" in lh and "clip" in lh):
        return True, True
    return False, trc


def load_model_for_contrastive_tune(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    torch_dtype: torch.dtype | None = None,
) -> tuple[torch.nn.Module, Any, str]:
    """Load HF CLIP-like model in train mode (freeze + LoRA applied by caller).

    Returns ``(model, processor, kind)`` where ``kind`` is ``"clip"`` or ``"auto"``.
    """
    use_auto, trc = _use_auto_model(model_id, trust_remote_code)
    kwargs: dict[str, Any] = {}
    if trc:
        kwargs["trust_remote_code"] = True
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    if use_auto:
        model = AutoModel.from_pretrained(model_id, **kwargs)
        _repair_evaclip_style_position_id_buffers(model)
        proc_kw = {"trust_remote_code": True} if trc else {}
        processor = AutoProcessor.from_pretrained(model_id, **proc_kw)
        if not (
            hasattr(model, "get_image_features")
            or hasattr(model, "encode_image")
        ):
            raise TypeError(
                f"{model_id} ({model.__class__.__name__}): need get_image_features or encode_image — "
                "use a CLIP/SigLIP/EvaCLIP-style Hub checkpoint.",
            )
        kind = "auto"
    else:
        clip_kw = {k: v for k, v in kwargs.items() if k == "torch_dtype"}
        model = CLIPModel.from_pretrained(model_id, **clip_kw)
        processor = CLIPProcessor.from_pretrained(model_id)
        kind = "clip"

    return model, processor, kind


def build_vision_lora_config(
    model: torch.nn.Module,
    *,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """PEFT LoRA on attention projections under the vision tower only.

    Uses ``exclude_modules`` so ``q_proj`` / ``v_proj`` inside ``text_model`` (etc.) are skipped.
    """
    try:
        from peft import LoraConfig
    except ImportError as e:
        raise ImportError("Vision LoRA requires `peft`. Install: pip install peft>=0.7") from e

    mods = target_modules or ["q_proj", "v_proj"]
    exclude: list[str] = []
    for tag in ("text_model", "text_encoder", "language_model"):
        if any(tag in n for n, _ in model.named_modules()):
            exclude.append(tag)

    if not exclude:
        raise ValueError(
            "Cannot locate text tower (no text_model / text_encoder / language_model in module names). "
            "Refusing to apply LoRA with target_modules=%s on the whole model." % mods,
        )

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=mods,
        exclude_modules=exclude,
    )


def save_tune_bundle(
    model: torch.nn.Module,
    processor: Any,
    output_dir: Path | str,
    *,
    base_model_id: str,
    train_meta: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    meta = {"base_model_id": base_model_id, **train_meta}
    (output_dir / "clip_siglip_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
