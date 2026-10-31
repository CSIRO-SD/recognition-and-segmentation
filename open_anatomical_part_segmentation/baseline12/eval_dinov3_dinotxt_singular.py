#!/usr/bin/env python3
"""Baseline 14: DINOv3 DINOTxt zero-shot segmentation — singular prompts (task.md).

Runs the inference recipe from ``facebookresearch/dinov3`` notebook
``notebooks/dinotxt_segmentation_inference.ipynb``: CLIP-style prompt templates,
patch–text cosine maps, whole-image or sliding-window inference, then argmax among
normalized category names listed in task.md present in GT.

- **Split**: rows with ``split == test`` in ``labelingv3/data_split_balanced.csv``.
- **Ground truth**: LabelMe merged to COCO via ``baseline10.build_coco_for_balanced_csv_split``
  (``cleaned_json`` paths + ``label_normalization.json`` → singular names), then restricted
  to ``SINGULAR_TASK_CLASS_NAMES`` unless ``--no-restrict-to-singular-classes``.
- **Images**: resolved under ``--dataset-root/images`` via
  ``baseline_common.sam3_coco_utils.resolve_dataset_image_path``.

Metrics mirror baseline12/baseline13:
``coco_eval_segm_*`` (mask AP/AR breakdowns), ``mask_miou``, ``mask_dice``.

Requires a clone of facebookresearch/dinov3 on ``PYTHONPATH`` (``--dinov3-repo`` / ``DINOV3_REPO``).

For offline checkpoints (e.g. no ``dl.fbaipublicfiles.com`` on compute nodes), pass::

    --backbone-weights …/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
    --dinotxt-weights …/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth

(or set ``DINOV3_BACKBONE_WEIGHTS`` / ``DINOV3_DINOTXT_WEIGHTS``).

Example::

    /datasets/work/icvg-phd-marsfield/work/Feng/conda_envs/insect/bin/python \\
        -m baseline14.eval_dinov3_dinotxt_singular \\
        --dinov3-repo /scratch3/xu097/proj06/third_party/dinov3 \\
        --backbone-weights /scratch3/xu097/proj06/checkpoint/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \\
        --dinotxt-weights /path/to/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \\
        --device cuda \\
        --out-dir outputs/baseline14_dinov3_singular
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from PIL import Image
from torch import Tensor, nn
from tqdm import tqdm

_PROJ = Path(__file__).resolve().parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from baseline5.coco_standalone_seg import (  # noqa: E402
    dump_coco_segm_predictions_json,
    evaluate_coco_segm_offline,
)
from baseline8.balanced_coco_gt import build_coco_for_balanced_csv_split  # noqa: E402
from baseline_common.run_outputs import resolve_eval_output_dir  # noqa: E402
from baseline_common.sam3_coco_utils import (  # noqa: E402
    compute_mask_miou_dice,
    metrics_to_serializable,
    resolve_dataset_image_path,
)

# task.md — singular noun prompts after label normalization ("提示词下面一样").
SINGULAR_TASK_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "head",
        "thorax",
        "leg",
        "abdomen",
        "antenna",
        "wing",
        "cercus",
        "sting",
        "tongue",
        "thorax_abdomen",
        "eye",
    },
)

PROMPT_TEMPLATES = (
    "a bad photo of a {0}.",
    "a photo of many {0}.",
    "a sculpture of a {0}.",
    "a photo of the hard to see {0}.",
    "a low resolution photo of the {0}.",
    "a rendering of a {0}.",
    "graffiti of a {0}.",
    "a bad photo of the {0}.",
    "a cropped photo of the {0}.",
    "a tattoo of a {0}.",
    "the embroidered {0}.",
    "a photo of a hard to see {0}.",
    "a bright photo of a {0}.",
    "a photo of a clean {0}.",
    "a photo of a dirty {0}.",
    "a dark photo of the {0}.",
    "a drawing of a {0}.",
    "a photo of my {0}.",
    "the plastic {0}.",
    "a photo of the cool {0}.",
    "a close-up photo of a {0}.",
    "a black and white photo of the {0}.",
    "a painting of the {0}.",
    "a painting of a {0}.",
    "a pixelated photo of the {0}.",
    "a sculpture of the {0}.",
    "a bright photo of the {0}.",
    "a cropped photo of a {0}.",
    "a plastic {0}.",
    "a photo of the dirty {0}.",
    "a jpeg corrupted photo of a {0}.",
    "a blurry photo of the {0}.",
    "a photo of the {0}.",
    "a good photo of the {0}.",
    "a rendering of the {0}.",
    "a {0} in a video game.",
    "a photo of one {0}.",
    "a doodle of a {0}.",
    "a close-up photo of the {0}.",
    "a photo of a {0}.",
    "the origami {0}.",
    "the {0} in a video game.",
    "a sketch of a {0}.",
    "a doodle of the {0}.",
    "a origami {0}.",
    "a low resolution photo of a {0}.",
    "the toy {0}.",
    "a rendition of the {0}.",
    "a photo of the clean {0}.",
    "a photo of a large {0}.",
    "a rendition of a {0}.",
    "a photo of a nice {0}.",
    "a photo of a weird {0}.",
    "a blurry photo of a {0}.",
    "a cartoon {0}.",
    "art of a {0}.",
    "a sketch of the {0}.",
    "a embroidered {0}.",
    "a pixelated photo of a {0}.",
    "itap of the {0}.",
    "a jpeg corrupted photo of the {0}.",
    "a good photo of a {0}.",
    "a plushie {0}.",
    "a photo of the nice {0}.",
    "a photo of the small {0}.",
    "a photo of the weird {0}.",
    "the cartoon {0}.",
    "art of the {0}.",
    "a drawing of the {0}.",
    "a photo of the large {0}.",
    "a black and white photo of a {0}.",
    "the plushie {0}.",
    "a dark photo of a {0}.",
    "itap of a {0}.",
    "graffiti of the {0}.",
    "a toy {0}.",
    "itap of my {0}.",
    "a photo of a cool {0}.",
    "a photo of a small {0}.",
    "a tattoo of the {0}.",
)

FOCUS_METRIC_LABELS: list[tuple[str, str]] = [
    ("coco_eval_segm_AP", "coco_eval_segm_Average Precision (mask AP)"),
    ("coco_eval_segm_AP_large", "coco_eval_segm_AP_large"),
    ("coco_eval_segm_AP_medium", "coco_eval_segm_AP_medium"),
    ("coco_eval_segm_AP_small", "coco_eval_segm_AP_small"),
    ("coco_eval_segm_AR_maxDets@100", "coco_eval_segm_Average Recall (mask AR @100)"),
    ("coco_eval_segm_AR_large", "coco_eval_segm_AR_large"),
    ("coco_eval_segm_AR_medium", "coco_eval_segm_AR_medium"),
    ("coco_eval_segm_AR_small", "coco_eval_segm_AR_small"),
    ("mask_dice", "mask_dice"),
    ("mask_miou", "mask_miou"),
]

DEFAULT_SPLIT_CSV = _PROJ / "labelingv3" / "data_split_balanced.csv"
DEFAULT_NORM_MAP = _PROJ / "labelingv3" / "label_normalization.json"
DEFAULT_DATASET_ROOT = Path("/datasets/work/icvg-phd-marsfield/work/all_crops")
DEFAULT_DINOV3_REPO = _PROJ / "third_party" / "dinov3"

NORMALIZE_IMAGENET = TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _filter_coco_to_class_names(coco: dict, allowed_names: frozenset[str]) -> dict:
    """Keep only categories listed in ``allowed_names`` and their annotations/images still used."""
    name_by_id = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    keep_cat_ids = {cid for cid, n in name_by_id.items() if n in allowed_names}

    cats = [c for c in coco["categories"] if int(c["id"]) in keep_cat_ids]
    anns = [a for a in coco.get("annotations", []) if int(a["category_id"]) in keep_cat_ids]

    img_ids_ann = {int(a["image_id"]) for a in anns}
    imgs = [im for im in coco["images"] if int(im["id"]) in img_ids_ann]

    coco_f = dict(coco)
    coco_f["categories"] = cats
    coco_f["annotations"] = anns
    coco_f["images"] = imgs
    return coco_f


def _resolve_weight_ref(cli: str | None, env_key: str) -> str | None:
    """CLI flag or env → local path (must exist) or http(s) URL string for torch.hub."""
    val = (cli or "").strip() or os.environ.get(env_key, "").strip()
    if not val:
        return None
    if val.startswith(("http://", "https://")):
        return val
    p = Path(val).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Weight file not found ({env_key} / --arg): {p}")
    return str(p)


def _resolve_dinov3_repo(arg_path: Path | None) -> Path:
    env = os.environ.get("DINOV3_REPO", "").strip()
    if arg_path is not None:
        return arg_path.expanduser().resolve()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DINOV3_REPO.resolve()


def _inject_dinov3_repo(repo: Path) -> None:
    if not repo.is_dir():
        raise FileNotFoundError(
            f"DINOv3 repository not found: {repo}. Clone facebookresearch/dinov3 "
            "or set --dinov3-repo / DINOV3_REPO."
        )
    r = str(repo)
    if r not in sys.path:
        sys.path.insert(0, r)


def encode_image(model: nn.Module, img: Tensor) -> tuple[Tensor | None, Tensor]:
    """Extract patch-token grid from DINOTxt visual tower (notebook)."""
    B, _, H, W = img.shape
    P = int(model.visual_model.backbone.patch_size)
    new_H = math.ceil(H / P) * P
    new_W = math.ceil(W / P) * P
    if (H, W) != (new_H, new_W):
        img = F.interpolate(img, size=(new_H, new_W), mode="bicubic", align_corners=False)

    B, _, h_i, w_i = img.shape
    _cls_tokens, _patch_unused, patch_tokens = model.visual_model.get_class_and_patch_tokens(img)
    blocks_patches = patch_tokens.reshape(B, h_i // P, w_i // P, -1).contiguous()
    return None, blocks_patches


class ShortSideResize(nn.Module):
    """Resize so the shorter side equals ``size`` (notebook)."""

    def __init__(self, size: int, interpolation: TVT.InterpolationMode) -> None:
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, img: Tensor) -> Tensor:
        _, h, w = TVTF.get_dimensions(img)
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            new_w = self.size
            new_h = int(self.size * h / w)
            return TVTF.resize(img, [new_h, new_w], self.interpolation)
        new_h = self.size
        new_w = int(self.size * w / h)
        return TVTF.resize(img, [new_h, new_w], self.interpolation)


def predict_whole(model: nn.Module, img: Tensor, text_features: Tensor) -> Tensor:
    """Cosine similarity maps [num_classes, h, w] at patch resolution (notebook)."""
    _, blocks_feats = encode_image(model, img.unsqueeze(0))
    _, h, w, _ = blocks_feats.shape
    blocks_feats = blocks_feats.squeeze(0)
    blocks_feats = F.normalize(blocks_feats, p=2, dim=-1)
    cos = torch.einsum("cd,hwd->chw", text_features, blocks_feats)
    return cos


def predict_slide(
    model: nn.Module,
    img: Tensor,
    text_features: Tensor,
    side: int,
    stride: int,
    device: torch.device,
) -> Tensor:
    """Sliding-window softmax-averaged maps at the resized tensor resolution (notebook)."""
    _, H, W = img.shape
    num_classes, _ = text_features.shape
    probs = torch.zeros(num_classes, H, W, device=device, dtype=text_features.dtype)
    counts = torch.zeros(H, W, device=device, dtype=text_features.dtype)
    h_grids = max(H - side + stride - 1, 0) // stride + 1
    w_grids = max(W - side + stride - 1, 0) // stride + 1
    for i in range(h_grids):
        for j in range(w_grids):
            y1 = i * stride
            x1 = j * stride
            y2 = min(y1 + side, H)
            x2 = min(x1 + side, W)
            y1 = max(y2 - side, 0)
            x1 = max(x2 - side, 0)

            img_window = img[:, y1:y2, x1:x2]
            cos = predict_whole(model, img_window, text_features)
            cos = F.interpolate(
                cos.unsqueeze(0),
                size=img_window.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            probs[:, y1:y2, x1:x2] += cos.softmax(dim=0)
            counts[y1:y2, x1:x2] += 1.0

    probs = probs / counts.clamp(min=1e-8)
    return probs


def _binary_mask_to_bbox_xyxy(mask_hw: Tensor) -> tuple[float, float, float, float] | None:
    m = mask_hw > 0.5
    if m.ndim != 2:
        raise ValueError("expected mask [H,W]")
    ys, xs = torch.where(m)
    if ys.numel() == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return float(x0), float(y0), float(x1), float(y1)


def build_category_text_features(
    model: nn.Module,
    tokenizer_fn,
    coco_gt: dict,
    device: torch.device,
    prompt_style: str,
    allowed_names: frozenset[str],
) -> dict[int, Tensor]:
    """One embedding vector per COCO category whose name is allowed (notebook text pipeline)."""
    cid_to_name = {int(c["id"]): str(c["name"]) for c in coco_gt["categories"]}
    cat_ids = sorted(cid_to_name.keys())
    out: dict[int, Tensor] = {}
    model.eval()
    for cid in tqdm(cat_ids, desc="DINOv3 encode_text (categories)"):
        raw_name = cid_to_name[cid]
        if raw_name not in allowed_names:
            continue
        segment = _format_prompt_segment(raw_name, prompt_style)
        texts = [t.format(segment) for t in PROMPT_TEMPLATES]
        tokens = tokenizer_fn(texts)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.as_tensor(tokens)
        tokens = tokens.to(device, non_blocking=True)
        with torch.inference_mode():
            feats = model.encode_text(tokens)
        feats = feats[:, feats.shape[1] // 2 :]
        feats = F.normalize(feats, p=2, dim=-1)
        feats = feats.mean(dim=0)
        feats = F.normalize(feats, p=2, dim=-1)
        out[cid] = feats
    return out


def _format_prompt_segment(name: str, style: str) -> str:
    if style == "raw":
        return name
    if style == "underscore_to_space":
        return name.replace("_", " ").strip()
    raise ValueError(f"Unknown prompt style {style!r}")


def _image_ann_category_ids(coco: dict, image_ids: list[int] | None = None) -> dict[int, set[int]]:
    want = set(image_ids) if image_ids is not None else None
    out: dict[int, set[int]] = {}
    for ann in coco.get("annotations", []):
        iid = int(ann["image_id"])
        if want is not None and iid not in want:
            continue
        out.setdefault(iid, set()).add(int(ann["category_id"]))
    return out


def dinotxt_predictions_for_balanced_split(
    *,
    model: nn.Module,
    category_feats: dict[int, Tensor],
    coco_gt: dict,
    dataset_root: Path,
    device: str,
    resize: int,
    mode: str,
    side: int,
    stride: int,
    task_vocab: frozenset[str] | None,
) -> dict[int, dict[str, Tensor]]:
    """Run DINOTxt inference; argmax over categories present on each image (after vocab filter)."""
    torch_device = torch.device(device)
    dtype = torch.float32

    transform_img = TVT.Compose(
        [
            ShortSideResize(resize, TVT.InterpolationMode.BICUBIC),
            TVT.ToTensor(),
            NORMALIZE_IMAGENET,
        ]
    )

    id_to_file = {int(im["id"]): im["file_name"] for im in coco_gt["images"]}
    id_to_wh = {int(im["id"]): (int(im["height"]), int(im["width"])) for im in coco_gt["images"]}
    cid_to_name = {int(c["id"]): str(c["name"]) for c in coco_gt["categories"]}
    img_cats = _image_ann_category_ids(coco_gt)
    preds: dict[int, dict[str, Tensor]] = {}

    for image_id, cat_ids in tqdm(sorted(img_cats.items()), desc="DINOv3 DINOTxt inference"):
        if not cat_ids:
            continue

        labels_sorted: list[int] = []
        for c in sorted(cat_ids):
            raw_name = cid_to_name[c]
            if task_vocab is not None and raw_name not in task_vocab:
                continue
            if c not in category_feats:
                continue
            labels_sorted.append(c)

        if not labels_sorted:
            continue

        rel = id_to_file[image_id]
        img_path = resolve_dataset_image_path(dataset_root, rel)
        if img_path is None:
            logging.warning("Missing image image_id=%s path=%s", image_id, dataset_root / rel)
            continue

        pil = Image.open(img_path).convert("RGB")
        h0, w0 = pil.size[1], pil.size[0]
        eh, ew = id_to_wh[image_id]
        if (eh, ew) != (h0, w0):
            logging.warning(
                "Size mismatch image_id=%s coco=(%s,%s) file=(%s,%s); using file size",
                image_id,
                ew,
                eh,
                w0,
                h0,
            )

        img_in = transform_img(pil).to(torch_device, dtype=dtype)
        text_stack = torch.stack([category_feats[c] for c in labels_sorted]).to(
            torch_device, dtype=dtype
        )

        with torch.inference_mode():
            if mode == "whole":
                maps = predict_whole(model, img_in, text_stack)
            elif mode == "slide":
                maps = predict_slide(model, img_in, text_stack, side, stride, torch_device)
            else:
                raise ValueError(f"Unknown mode {mode!r}")

            maps_up = F.interpolate(
                maps.unsqueeze(0),
                size=(h0, w0),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            labels_hw = maps_up.argmax(dim=0)

        boxes: list[Tensor] = []
        masks_hw: list[Tensor] = []
        scores: list[float] = []
        label_ids: list[int] = []

        for k, cid in enumerate(labels_sorted):
            binary = (labels_hw == k).float()
            if float(binary.sum()) < 1e-6:
                continue
            box = _binary_mask_to_bbox_xyxy(binary)
            if box is None:
                continue
            xyxy_t = torch.tensor([[box[0], box[1], box[2], box[3]]], dtype=dtype, device="cpu")
            sel = labels_hw == k
            vals = maps_up[k][sel]
            score = float(vals.mean().item()) if vals.numel() else float(maps_up[k].max().item())
            boxes.append(xyxy_t)
            masks_hw.append(binary.cpu())
            scores.append(score)
            label_ids.append(cid)

        if not scores:
            continue

        preds[image_id] = {
            "boxes": torch.cat(boxes, dim=0),
            "scores": torch.tensor(scores, dtype=dtype),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "masks": torch.stack(masks_hw, dim=0),
        }

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return preds


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split-csv",
        type=Path,
        default=DEFAULT_SPLIT_CSV,
        help="CSV with json_path,split.",
    )
    ap.add_argument(
        "--norm-map",
        type=Path,
        default=DEFAULT_NORM_MAP,
        help="label_normalization.json.",
    )
    ap.add_argument(
        "--csv-split",
        type=str,
        default="test",
        help='Split bucket ("test", "val", "training").',
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root with images/ for COCO file_name lookup.",
    )
    ap.add_argument(
        "--dinov3-repo",
        type=Path,
        default=None,
        help="Path to facebookresearch/dinov3 clone (default: DINOV3_REPO env or proj06/third_party/dinov3).",
    )
    ap.add_argument(
        "--backbone-weights",
        type=str,
        default=None,
        help="ViT backbone .pth or URL (or env DINOV3_BACKBONE_WEIGHTS).",
    )
    ap.add_argument(
        "--dinotxt-weights",
        type=str,
        default=None,
        help="DINOTxt head + text checkpoint .pth or URL (or env DINOV3_DINOTXT_WEIGHTS).",
    )
    ap.add_argument(
        "--bpe-path",
        type=str,
        default=None,
        help="Optional local path to BPE vocab .gz for tokenizer (else hub URL).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--resize", type=int, default=512, help="Short-side resize before ViT.")
    ap.add_argument(
        "--mode",
        choices=("whole", "slide"),
        default="whole",
        help="whole = single forward; slide = sliding-window softmax blend.",
    )
    ap.add_argument("--side", type=int, default=384, help="Window side for mode=slide.")
    ap.add_argument("--stride", type=int, default=192, help="Stride for mode=slide.")
    ap.add_argument(
        "--prompt-style",
        choices=("raw", "underscore_to_space"),
        default="underscore_to_space",
        help="Category name formatting before PROMPT_TEMPLATES.",
    )
    ap.add_argument(
        "--restrict-to-singular-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true (default), only evaluate task.md singular classes listed in "
            "SINGULAR_TASK_CLASS_NAMES. Use --no-restrict-to-singular-classes for all GT categories."
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJ / "outputs" / "baseline14_dinov3_singular",
        help="Output directory (timestamped subfolder unless --flat).",
    )
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--flat", action="store_true")
    args = ap.parse_args()

    repo = _resolve_dinov3_repo(args.dinov3_repo)
    _inject_dinov3_repo(repo)

    from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l  # noqa: E402

    run_dir, rid = resolve_eval_output_dir(args.out_dir, run_id=args.run_id, flat=args.flat)
    gt_path = run_dir / f"baseline14_singular_{args.csv_split}_gt_{rid}.json"
    pred_path = run_dir / f"baseline14_dinov3_singular_{args.csv_split}_pred_{rid}.json"
    metrics_path = run_dir / f"baseline14_dinov3_singular_{args.csv_split}_metrics_{rid}.json"
    summary_path = run_dir / f"baseline14_dinov3_singular_{args.csv_split}_summary_{rid}.txt"

    vocab = SINGULAR_TASK_CLASS_NAMES if args.restrict_to_singular_classes else None

    print(f"Run directory: {run_dir}", flush=True)
    coco = build_coco_for_balanced_csv_split(
        split_csv=args.split_csv, norm_map=args.norm_map, split_name=args.csv_split
    )
    if args.restrict_to_singular_classes:
        coco = _filter_coco_to_class_names(coco, SINGULAR_TASK_CLASS_NAMES)
    gt_path.write_text(json.dumps(coco, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"GT: images={len(coco['images'])} "
        f"annotations={len(coco['annotations'])} "
        f"categories={len(coco['categories'])}",
        flush=True,
    )

    backbone_ref = _resolve_weight_ref(args.backbone_weights, "DINOV3_BACKBONE_WEIGHTS")
    dinotxt_ref = _resolve_weight_ref(args.dinotxt_weights, "DINOV3_DINOTXT_WEIGHTS")

    hub_kwargs: dict = {}
    if backbone_ref:
        hub_kwargs["backbone_weights"] = backbone_ref
    if dinotxt_ref:
        hub_kwargs["weights"] = dinotxt_ref
    if args.bpe_path:
        hub_kwargs["bpe_path_or_url"] = args.bpe_path

    dev = torch.device(args.device)
    model, tok = dinov3_vitl16_dinotxt_tet1280d20h24l(**hub_kwargs)
    model.to(dev, non_blocking=True)
    model.eval()
    tokenizer_fn = tok.tokenize if hasattr(tok, "tokenize") else tok

    allowed_names = SINGULAR_TASK_CLASS_NAMES if args.restrict_to_singular_classes else frozenset(
        str(c["name"]) for c in coco["categories"]
    )
    cat_feats = build_category_text_features(
        model, tokenizer_fn, coco, dev, args.prompt_style, allowed_names
    )

    preds = dinotxt_predictions_for_balanced_split(
        model=model,
        category_feats=cat_feats,
        coco_gt=coco,
        dataset_root=args.dataset_root,
        device=args.device,
        resize=args.resize,
        mode=args.mode,
        side=args.side,
        stride=args.stride,
        task_vocab=vocab,
    )
    dump_coco_segm_predictions_json(preds, pred_path)

    coco_metrics = evaluate_coco_segm_offline(gt_path, pred_path)
    mask_md = compute_mask_miou_dice(gt_path, pred_path)
    metrics = {**coco_metrics, **mask_md}
    serial = metrics_to_serializable(metrics)
    metrics_path.write_text(json.dumps(serial, indent=2) + "\n", encoding="utf-8")

    insect_py = "/datasets/work/icvg-phd-marsfield/work/Feng/conda_envs/insect/bin/python"
    lines = [
        f"baseline14 DINOv3 DINOTxt singular split={args.csv_split} run_id={rid}\n",
        f"dinov3_repo={repo} mode={args.mode} resize={args.resize} "
        f"prompt_style={args.prompt_style} singular_only={args.restrict_to_singular_classes}\n",
        f"Example: {insect_py} -m baseline14.eval_dinov3_dinotxt_singular "
        f"--dinov3-repo {repo} --device cuda\n",
    ]
    for key, label in FOCUS_METRIC_LABELS:
        if key in serial:
            lines.append(f"  {label}: {serial[key]:.6f}\n")
    summary_path.write_text("".join(lines), encoding="utf-8")
    print("".join(lines), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
