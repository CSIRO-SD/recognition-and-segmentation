"""Introspect SAM 3 image model vision trunk (ViTDet) and verify it matches public ``sam3.pt``."""

from __future__ import annotations

import torch

# Must stay aligned with ``sam3.model_builder._create_vit_backbone`` for the default image model.
SAM3_HF_IMAGE_TRUNK_EMBED_DIM = 1024
SAM3_HF_IMAGE_TRUNK_DEPTH = 32


def vision_trunk_embed_depth(model: torch.nn.Module) -> tuple[int, int]:
    """Return ``(embed_dim, depth)`` for the ViTDet trunk under ``model.backbone.vision_backbone``."""
    trunk = model.backbone.vision_backbone.trunk
    embed_dim = int(trunk.patch_embed.proj.out_channels)
    depth = len(trunk.blocks)
    return embed_dim, depth


def assert_trunk_matches_hf_sam3_image_checkpoint(model: torch.nn.Module) -> None:
    """Raise if the loaded trunk does not match the architecture of ``facebook/sam3`` ``sam3.pt``."""
    ed, d = vision_trunk_embed_depth(model)
    if ed != SAM3_HF_IMAGE_TRUNK_EMBED_DIM or d != SAM3_HF_IMAGE_TRUNK_DEPTH:
        raise RuntimeError(
            f"Vision trunk (embed_dim={ed}, depth={d}) does not match the public "
            f"Hugging Face sam3.pt architecture (expected embed_dim="
            f"{SAM3_HF_IMAGE_TRUNK_EMBED_DIM}, depth={SAM3_HF_IMAGE_TRUNK_DEPTH})."
        )


def describe_sam3_vision_trunk_vs_vit_b() -> str:
    return (
        "The released SAM 3 image checkpoint uses a ViTDet-style trunk with "
        f"embed_dim={SAM3_HF_IMAGE_TRUNK_EMBED_DIM} and depth={SAM3_HF_IMAGE_TRUNK_DEPTH}. "
        "That is not the classical ImageNet ViT-B/16 definition (typically embed_dim=768, "
        "depth=12). Meta has not published a separate SAM 3 image weight file for ViT-B/16."
    )
