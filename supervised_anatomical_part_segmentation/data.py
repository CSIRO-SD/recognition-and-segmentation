from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes

from coco_mask import (
    ann_to_instance_masks,
    build_cat_maps,
    index_coco_by_image,
    load_coco,
    rasterize_semantic_mask,
)


def _empty_mask_rcnn_target(size: int) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "masks": torch.zeros((0, size, size), dtype=torch.bool),
    }


def _fix_degenerate_xyxy_boxes(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    """Torchvision Mask R-CNN requires x2>x1 and y2>y1. Thin masks can yield zero w/h."""
    out = boxes.clone()
    s = float(image_size)
    out[:, 2] = torch.maximum(out[:, 2], out[:, 0] + 1.0)
    out[:, 3] = torch.maximum(out[:, 3], out[:, 1] + 1.0)
    out[:, 0] = torch.clamp(out[:, 0], 0.0, s - 1.0)
    out[:, 1] = torch.clamp(out[:, 1], 0.0, s - 1.0)
    out[:, 2] = torch.clamp(out[:, 2], 1.0, s)
    out[:, 3] = torch.clamp(out[:, 3], 1.0, s)
    out[:, 2] = torch.maximum(out[:, 2], out[:, 0] + 1.0)
    out[:, 3] = torch.maximum(out[:, 3], out[:, 1] + 1.0)
    return out


class CocoSemanticSegDataset(Dataset):
    """Semantic segmentation: mask 0=bg, 1..N=parts."""

    def __init__(
        self,
        ann_path: str | Path,
        img_root: str | Path,
        size: int,
        image_set: str = "default",
        segformer_processor: Any = None,
    ):
        self.coco = load_coco(ann_path)
        self.img_root = Path(img_root)
        self.images = sorted(self.coco["images"], key=lambda x: x["id"])
        self.id2lbl, self.cat_names = build_cat_maps(self.coco["categories"])
        self._by_im = index_coco_by_image(self.coco)
        self.size = int(size)
        self.image_set = image_set
        self.segformer_processor = segformer_processor

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        im_info = self.images[idx]
        iid = int(im_info["id"])
        path = self.img_root / im_info["file_name"]
        im = Image.open(path).convert("RGB")
        w0, h0 = im.size
        anns = self._by_im.get(iid, [])
        mask = rasterize_semantic_mask(anns, self.id2lbl, w0, h0)

        if self.segformer_processor is not None:
            enc = self.segformer_processor(images=im, return_tensors="pt")
            pv = enc["pixel_values"]
            # HF returns [1, C, H, W] or [C, H, W]; collate stacks to [B, C, H, W].
            if pv.dim() == 4:
                pv = pv.squeeze(0)
            h, w = int(pv.shape[-2]), int(pv.shape[-1])
            mask_pil = Image.fromarray(mask, mode="L")
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            mask_t = torch.from_numpy(np.array(mask_pil, copy=True)).long()
            return {"pixel_values": pv, "labels": mask_t, "id": iid}
        # default: tensor image [-1 handled by caller] ImageNet
        im_res = im.resize((self.size, self.size), Image.BILINEAR)
        x = TF.to_tensor(im_res)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_pil = Image.fromarray(mask, mode="L")
        mask_pil = mask_pil.resize((self.size, self.size), Image.NEAREST)
        y = torch.from_numpy(np.array(mask_pil, copy=True)).long().clamp(min=0)
        return {"images": x, "labels": y, "id": iid}

    @property
    def num_fg_classes(self) -> int:
        return len(self.coco["categories"])

    @property
    def num_labels_with_bg(self) -> int:
        return self.num_fg_classes + 1


def collate_semantic(batch: list[dict]) -> dict[str, Any]:
    if "pixel_values" in batch[0]:
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
            "labels": torch.stack([b["labels"] for b in batch], dim=0),
            "ids": [b["id"] for b in batch],
        }
    return {
        "images": torch.stack([b["images"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "ids": [b["id"] for b in batch],
    }


class CocoInstanceTV513Dataset(Dataset):
    """Mask R-CNN targets (fixed resize); masks bool [K,H,W], labels 1..N."""

    def __init__(
        self,
        ann_path: str | Path,
        img_root: str | Path,
        size: int,
    ):
        self.coco = load_coco(ann_path)
        self.img_root = Path(img_root)
        self.images = sorted(self.coco["images"], key=lambda x: x["id"])
        self.id2lbl, self.cat_names = build_cat_maps(self.coco["categories"])
        self._by_im = index_coco_by_image(self.coco)
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        im_info = self.images[idx]
        iid = int(im_info["id"])
        path = self.img_root / im_info["file_name"]
        im = Image.open(path).convert("RGB")
        w0, h0 = im.size
        anns = self._by_im.get(iid, [])
        masks, labels = ann_to_instance_masks(anns, self.id2lbl, w0, h0)
        im = im.resize((self.size, self.size), Image.BILINEAR)
        x = TF.to_tensor(im)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if masks.numel() == 0 or masks.shape[0] == 0:
            return x, _empty_mask_rcnn_target(self.size)

        m = F.interpolate(masks.unsqueeze(1), (self.size, self.size), mode="nearest").squeeze(1)
        bmasks = m > 0.5
        # Only skip masks with no foreground after resize (nothing to supervise).
        nonempty = bmasks.flatten(1).sum(dim=1) > 0
        bmasks = bmasks[nonempty]
        labels = labels[nonempty]
        if bmasks.shape[0] == 0:
            return x, _empty_mask_rcnn_target(self.size)

        boxes = _fix_degenerate_xyxy_boxes(masks_to_boxes(bmasks), self.size)
        target = {
            "boxes": boxes.float(),
            "labels": labels,
            "masks": bmasks,
        }
        return x, target

    @property
    def num_fg_classes(self) -> int:
        return len(self.coco["categories"])


class CocoMask2FormerDataset(Dataset):
    """One crop per item; mask2former expects list-valued batch fields."""

    def __init__(self, ann_path: str | Path, img_root: str | Path, processor: Any, size: int | None = None):
        self.coco = load_coco(ann_path)
        self.img_root = Path(img_root)
        self.images = sorted(self.coco["images"], key=lambda x: x["id"])
        self.id2lbl, self.cat_names = build_cat_maps(self.coco["categories"])
        self._by_im = index_coco_by_image(self.coco)
        self.processor = processor
        self.size = size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        im_info = self.images[idx]
        iid = int(im_info["id"])
        path = self.img_root / im_info["file_name"]
        pil = Image.open(path).convert("RGB")
        w0, h0 = pil.size
        anns = self._by_im.get(iid, [])
        masks, labels = ann_to_instance_masks(anns, self.id2lbl, w0, h0)
        if self.size is not None:
            pil_in = pil.resize((self.size, self.size), Image.BILINEAR)
        else:
            pil_in = pil

        enc = self.processor(images=pil_in, return_tensors="pt")
        pv = enc["pixel_values"].squeeze(0)
        _, h, w = pv.shape
        if masks.numel() and masks.shape[0] > 0:
            m = F.interpolate(masks.unsqueeze(1).float(), (h, w), mode="nearest").squeeze(1)
            # HF Mask2Former / DETR-like: semantic class IDs are 0-indexed (0 .. num_semantic_classes-1).
            # CocoInstance / ann_to_instance_masks uses 1..N to match torchvision Mask R-CNN.
            cl = [(labels.long() - 1).clamp(min=0)]
        else:
            # Do not supervise a dummy all-zero mask (poisons bipartite matching + mask loss).
            m = torch.empty((0, h, w), dtype=torch.float32)
            cl = [torch.empty(0, dtype=torch.int64)]

        ml = [m]
        return {"pixel_values": pv, "mask_labels": ml, "class_labels": cl, "id": iid}


def collate_mask2former(batch: list[dict]) -> dict[str, Any]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    mask_labels = [b["mask_labels"][0] for b in batch]
    class_labels = [b["class_labels"][0] for b in batch]
    return {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
        "ids": [b["id"] for b in batch],
    }
