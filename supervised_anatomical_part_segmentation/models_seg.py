"""Torchvision / HF segmentation and detection wrappers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from unet import UNet


def build_semantic_model(name: str, num_classes_with_bg: int, *, pretrained: bool = True) -> nn.Module:
    """num_classes_with_bg = N foreground + 1 background (mask labels 0..N)."""
    name = name.lower()
    if name == "unet":
        return UNet(in_ch=3, num_classes=num_classes_with_bg, base=64)

    if name == "deeplabv3plus":
        if not pretrained:
            return deeplabv3_resnet50(weights=None, num_classes=num_classes_with_bg)
        # VOC weights fix ``num_classes==21``; swap main + aux 1×1 heads only (keep ASPP / backbone).
        m = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        in_main = m.classifier[4].in_channels
        m.classifier[4] = nn.Conv2d(in_main, num_classes_with_bg, kernel_size=1, stride=1)
        if m.aux_classifier is not None:
            in_aux = m.aux_classifier[4].in_channels
            m.aux_classifier[4] = nn.Conv2d(in_aux, num_classes_with_bg, kernel_size=1, stride=1)
        return m

    if name == "segformer":
        from transformers import SegformerForSemanticSegmentation

        mid = "nvidia/mit-b0"
        m = SegformerForSemanticSegmentation.from_pretrained(
            mid,
            num_labels=num_classes_with_bg,
            ignore_mismatched_sizes=True,
        )
        return m

    raise ValueError(f"Unknown semantic model {name}; choose unet | deeplabv3plus | segformer")


def semantic_forward(model: nn.Module, images: torch.Tensor, name: str):
    name = name.lower()
    if name == "segformer":
        out = model(pixel_values=images)
        return out.logits
    out = model(images)
    if isinstance(out, dict):
        return out["out"]
    return out


def semantic_aux_logits(model: nn.Module, images: torch.Tensor, name: str):
    if name.lower() != "deeplabv3plus":
        return None
    out = model(images)
    return out.get("aux")


def build_mask_rcnn(num_classes: int, *, pretrained: bool = True):
    """num_classes = num foreground + background (TV convention: dense labels 0..N-1).

    Pretrained COCO weights fix ``num_classes==91``; for custom ``num_classes`` we load backbone+FPN
    then replace the box and mask heads (Torchvision finetuning pattern).
    """
    if not pretrained:
        return maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def build_mask2former(
    num_labels: int,
    *,
    mid: str = "facebook/mask2former-swin-small-coco-instance",
    class_names: list[str] | None = None,
):
    """Load pretrained Mask2Former; classification head is resized via ``config`` + ``ignore_mismatched_sizes``.

    COCO checkpoint uses 80 classes + no-object (``class_predictor`` out = 81); custom ``num_labels`` uses
    ``num_labels + 1``. Mismatched head weights are re-initialized — expected. We refresh ``id2label`` on
    ``config`` so HuggingFace does not warn about ``num_labels`` vs COCO ``id2label`` length.
    """
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    config = Mask2FormerConfig.from_pretrained(mid)
    if class_names is not None and len(class_names) == num_labels:
        config.id2label = {i: (class_names[i].strip() or "background") for i in range(num_labels)}
        config.label2id = {v: k for k, v in config.id2label.items()}
    else:
        config.num_labels = num_labels
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        mid,
        config=config,
        ignore_mismatched_sizes=True,
    )
