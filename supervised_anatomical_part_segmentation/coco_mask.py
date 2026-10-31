"""COCO polygon helpers: category maps, rasterization, instance targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_coco(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_cat_maps(categories: list[dict]) -> tuple[dict[int, int], list[str]]:
    """Map COCO category_id -> dense label in 1..N (N=num foreground classes). Return (id2lbl, names_by_dense_idx)."""
    cats = sorted(categories, key=lambda c: c["id"])
    id2lbl: dict[int, int] = {}
    names: list[str] = [""]  # index 0 unused; 1..N names
    for k, c in enumerate(cats, start=1):
        id2lbl[int(c["id"])] = k
        names.append(str(c["name"]))
    return id2lbl, names


def index_coco_by_image(coco: dict[str, Any]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        iid = int(a["image_id"])
        out.setdefault(iid, []).append(a)
    return out


def rasterize_semantic_mask(
    anns: list[dict],
    id2lbl: dict[int, int],
    width: int,
    height: int,
) -> np.ndarray:
    """HxW uint8: 0=bg, 1..N=part (later polygons overwrite earlier on overlap)."""
    m = np.zeros((height, width), dtype=np.uint8)
    for a in anns:
        if int(a.get("iscrowd", 0)) == 1:
            continue
        cid = int(a["category_id"])
        if cid not in id2lbl:
            continue
        lbl = id2lbl[cid]
        seg = a.get("segmentation")
        if not seg:
            continue
        for poly in seg:
            if not poly or len(poly) < 6:
                continue
            pts = list(zip(poly[0::2], poly[1::2]))
            img = Image.fromarray(m, mode="L")
            ImageDraw.Draw(img).polygon(pts, outline=int(lbl), fill=int(lbl))
            m = np.array(img, dtype=np.uint8)
    return m


def ann_to_instance_masks(
    anns: list[dict],
    id2lbl: dict[int, int],
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masks [K,H,W] float01, labels [K] with values 1..N (for Mask2Former / TV)."""
    masks: list[np.ndarray] = []
    labels: list[int] = []
    for a in anns:
        if int(a.get("iscrowd", 0)) == 1:
            continue
        cid = int(a["category_id"])
        if cid not in id2lbl:
            continue
        lbl = id2lbl[cid]
        seg = a.get("segmentation")
        if not seg:
            continue
        acc = np.zeros((height, width), dtype=np.uint8)
        for poly in seg:
            if not poly or len(poly) < 6:
                continue
            pts = list(zip(poly[0::2], poly[1::2]))
            img = Image.fromarray(acc, mode="L")
            ImageDraw.Draw(img).polygon(pts, outline=1, fill=1)
            acc = np.maximum(acc, np.array(img, dtype=np.uint8))
        if acc.any():
            masks.append(acc.astype(np.float32))
            labels.append(lbl)
    if not masks:
        return torch.empty(0, height, width), torch.empty(0, dtype=torch.int64)
    return torch.stack([torch.from_numpy(x) for x in masks], dim=0), torch.tensor(labels, dtype=torch.int64)
