"""Shared SAM3 inference + COCO-format dumping + offline COCO metrics (mask AP).

Requires the official ``sam3`` package (facebookresearch/sam3) on PYTHONPATH / installed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SAM3_BPE_ASSET = "bpe_simple_vocab_16e6.txt.gz"

import torch
from PIL import Image
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Project root on sys.path (proj06)
# -----------------------------------------------------------------------------
_PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))


def resolve_sam3_bpe_path(preferred: Path | str | None = None) -> Path:
    """Locate ``sam3/assets/bpe_simple_vocab_16e6.txt.gz`` without ``pkg_resources``.

    Editable / namespace installs can make ``pkg_resources.resource_filename("sam3", ...)``
    fail; ``build_sam3_image_model(bpe_path=...)`` avoids that code path.
    """
    if preferred is not None:
        p = Path(preferred).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"BPE file not found (--bpe-path): {p}")

    env = os.environ.get("SAM3_BPE_PATH", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"SAM3_BPE_PATH is set but file missing: {p}")

    import sam3

    for root in getattr(sam3, "__path__", []):
        cand = Path(root).resolve() / "assets" / SAM3_BPE_ASSET
        if cand.is_file():
            return cand

    mod_file = getattr(sam3, "__file__", None)
    if mod_file:
        cand = Path(mod_file).resolve().parent / "assets" / SAM3_BPE_ASSET
        if cand.is_file():
            return cand

    # Prefer canonical clone at proj06/sam3, then optional third_party copy.
    for base in (_PROJ_ROOT / "sam3", _PROJ_ROOT / "third_party" / "sam3"):
        cand = base / "sam3" / "assets" / SAM3_BPE_ASSET
        if cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        f"Could not find sam3/assets/{SAM3_BPE_ASSET}. "
        "Install or clone SAM 3 with assets present, or set SAM3_BPE_PATH / pass --bpe-path."
    )


def load_coco_gt(coco_path: Path) -> dict:
    with coco_path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_image_path(
    dataset_root: Path, relative_file_name: str
) -> Path | None:
    """Return ``dataset_root / relative_file_name`` if that path is a regular file, else ``None``."""
    p = dataset_root / relative_file_name
    return p if p.is_file() else None


def category_name_to_id(coco: dict) -> dict[str, int]:
    return {c["name"]: int(c["id"]) for c in coco.get("categories", [])}


def phrases_per_image(
    coco: dict, *, category_names_only: bool = False
) -> dict[int, list[str]]:
    """Unique prompts per ``image_id``.

    By default uses ``annotation["noun_phrase"]`` when present, else the COCO category
    ``name``. With ``category_names_only=True``, only category ``name`` strings are used
    (ground-truth class names), ignoring ``noun_phrase``.
    """
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    out: dict[int, set[str]] = defaultdict(set)
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if category_names_only:
            phrase = id_to_name.get(cid)
        else:
            phrase = ann.get("noun_phrase") or id_to_name.get(cid)
        if phrase:
            out[int(ann["image_id"])].add(str(phrase))
    return {k: sorted(v) for k, v in out.items()}


def run_sam3_image_inference_to_coco_predictions(
    *,
    coco_gt: dict,
    dataset_root: Path,
    processor,
    confidence_threshold: float = 0.5,
    shard: tuple[int, int] | None = None,
    empty_cuda_cache_each_image: bool = False,
    category_names_only: bool = False,
) -> dict[int, dict[str, torch.Tensor]]:
    """Run SAM3 text-prompted grounding for each GT phrase per image.

    Returns a mapping ``image_id -> {boxes, scores, labels, masks}`` suitable for
    ``PredictionDumper.prepare(..., iou_type='segm')``.

    If ``shard=(rank, world_size)``, only images at indices ``rank, rank+world_size, ...``
    are processed (for multi-GPU data parallelism).

    ``category_names_only``: use COCO category ``name`` as the text prompt for each
    annotation (see ``phrases_per_image``).

    If ``dataset_root / file_name`` is missing, that image is skipped (warning logged);
    evaluation on GT still includes those images unless you filter the COCO JSON separately.
    """
    id_to_file = {int(im["id"]): im["file_name"] for im in coco_gt["images"]}
    name_to_cat = category_name_to_id(coco_gt)
    img_to_phrases = phrases_per_image(
        coco_gt, category_names_only=category_names_only
    )

    processor.confidence_threshold = confidence_threshold
    predictions: dict[int, dict[str, torch.Tensor]] = {}

    items = sorted(img_to_phrases.items())
    if shard is not None:
        rank, world_size = shard
        items = items[rank::world_size]

    rank_tag = f" GPU{shard[0]}" if shard else ""
    skipped_missing_file = 0
    for image_id, phrases in tqdm(items, desc=f"SAM3 inference{rank_tag}"):
        rel = id_to_file[image_id]
        img_path = resolve_dataset_image_path(dataset_root, rel)
        if img_path is None:
            skipped_missing_file += 1
            miss = dataset_root / rel
            logging.warning("Skipping image_id=%s: missing file %s", image_id, miss)
            continue

        image = Image.open(img_path).convert("RGB")
        state = processor.set_image(image)

        box_chunks: list[torch.Tensor] = []
        score_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []

        for pi, phrase in enumerate(phrases):
            if pi > 0:
                processor.reset_all_prompts(state)
            state = processor.set_text_prompt(phrase, state)
            if state["scores"].numel() == 0:
                continue

            cat_id = name_to_cat.get(phrase)
            if cat_id is None:
                logging.warning("Unknown phrase (no category): %s", phrase)
                continue

            n = state["scores"].shape[0]
            labels = torch.full(
                (n,), cat_id, dtype=torch.int64, device=state["scores"].device
            )
            box_chunks.append(state["boxes"])
            score_chunks.append(state["scores"])
            label_chunks.append(labels)
            # masks: bool [N, H, W] -> float for dumper
            m = state["masks"].float()
            # Sam3Processor returns [N, 1, H, W]; keep [N, H, W] until final stack so we
            # do not stack an extra singleton (dump does masks.squeeze(1) -> must be 3D).
            if m.ndim == 4 and m.shape[1] == 1:
                m = m.squeeze(1)
            mask_chunks.append(m)

        if not score_chunks:
            continue

        boxes = torch.cat(box_chunks, dim=0)
        scores = torch.cat(score_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        masks = torch.cat(mask_chunks, dim=0)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        predictions[image_id] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "masks": masks,
        }

        if empty_cuda_cache_each_image and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if skipped_missing_file:
        msg = (
            f"[sam3] skipped {skipped_missing_file} image(s) with missing files "
            f"under {dataset_root}"
        )
        logging.warning(msg)
        print(msg, flush=True)

    # Move to CPU for JSON / pycocotools
    cpu_preds: dict[int, dict[str, torch.Tensor]] = {}
    for image_id, p in predictions.items():
        cpu_preds[image_id] = {
            "boxes": p["boxes"].cpu(),
            "scores": p["scores"].cpu(),
            "labels": p["labels"].cpu(),
            "masks": p["masks"].cpu(),
        }
    return cpu_preds


def _predictions_to_numpy_for_mp_queue(
    preds: dict[int, dict[str, torch.Tensor]],
) -> dict[int, dict[str, np.ndarray]]:
    """Serialize prediction tensors as NumPy for ``multiprocessing.Queue``.

    PyTorch's default pickling for large CPU tensors uses file-descriptor sharing
    (``rebuild_storage_fd``). That often fails on SLURM / shared clusters with
    ``FileNotFoundError`` / broken socket paths when the parent unpickles results.
    NumPy arrays pickle as plain data and avoid that path.
    """
    out: dict[int, dict[str, np.ndarray]] = {}
    for image_id, p in preds.items():
        out[int(image_id)] = {
            "boxes": np.asarray(p["boxes"].detach().cpu(), order="C"),
            "scores": np.asarray(p["scores"].detach().cpu(), order="C"),
            "labels": np.asarray(p["labels"].detach().cpu(), order="C"),
            "masks": np.asarray(p["masks"].detach().cpu(), order="C"),
        }
    return out


def _predictions_from_numpy_after_mp_queue(
    blob: dict[int, dict[str, np.ndarray]],
) -> dict[int, dict[str, torch.Tensor]]:
    return {
        int(k): {
            "boxes": torch.from_numpy(v["boxes"]),
            "scores": torch.from_numpy(v["scores"]),
            "labels": torch.from_numpy(v["labels"]),
            "masks": torch.from_numpy(v["masks"]),
        }
        for k, v in blob.items()
    }


def _sam3_mp_inference_worker(
    rank: int, world_size: int, kw: dict[str, Any], result_queue: Any
) -> None:
    # Set before importing torch/sam3: expose one physical GPU as ``cuda:0`` in this
    # process. SAM3 otherwise mixes default cuda:0 tensors with cuda:{k}, causing
    # "indices should be on the same device as the indexed tensor" errors.
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    import time

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    try:
        time.sleep(min(rank * 1.5, 20.0))
        coco_gt = load_coco_gt(Path(kw["gt_json"]))
        dataset_root = Path(kw["dataset_root"])
        ck_raw = (kw.get("checkpoint_path") or "").strip()
        model = build_sam3_image_model(
            bpe_path=kw["bpe_path"],
            device="cuda",
            eval_mode=True,
            checkpoint_path=ck_raw if ck_raw else None,
            load_from_HF=bool(kw["load_from_hf"]),
            enable_segmentation=True,
        )
        processor = Sam3Processor(model, device="cuda")
        preds = run_sam3_image_inference_to_coco_predictions(
            coco_gt=coco_gt,
            dataset_root=dataset_root,
            processor=processor,
            confidence_threshold=float(kw["confidence_threshold"]),
            shard=(rank, world_size),
            empty_cuda_cache_each_image=bool(kw.get("empty_cuda_cache_each_image", True)),
            category_names_only=bool(kw.get("category_names_only", False)),
        )
        result_queue.put((rank, _predictions_to_numpy_for_mp_queue(preds)))
    except Exception as exc:
        result_queue.put((rank, exc))


def run_sam3_image_inference_multiprocess(
    *,
    gt_json: Path,
    dataset_root: Path,
    bpe_path: Path,
    checkpoint_path: Path | None,
    load_from_hf: bool,
    confidence_threshold: float,
    num_workers: int,
    empty_cuda_cache_each_image: bool = True,
    category_names_only: bool = False,
) -> dict[int, dict[str, torch.Tensor]]:
    """Shard images across ``num_workers`` processes (one GPU each; ``spawn`` for CUDA).

    Worker *rank* sets ``CUDA_VISIBLE_DEVICES=<rank>`` so that process only sees one
    physical GPU (as ``cuda:0``), avoiding device-mismatch bugs inside SAM3. Assumes
    the machine exposes GPU indices ``0 .. num_workers-1`` (adjust env before launch
    if you use a non-contiguous GPU set).

    Results are sent through the queue as NumPy arrays (rebuilt to CPU tensors in the
    parent) so PyTorch's FD-based tensor pickling does not fail on SLURM/HPC.
    """
    import multiprocessing as mp

    if num_workers <= 1:
        raise ValueError("num_workers must be >= 2")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    kw: dict[str, Any] = {
        "gt_json": str(gt_json.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "bpe_path": str(bpe_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else "",
        "load_from_hf": load_from_hf,
        "confidence_threshold": confidence_threshold,
        "empty_cuda_cache_each_image": empty_cuda_cache_each_image,
        "category_names_only": category_names_only,
    }
    procs = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=_sam3_mp_inference_worker,
            args=(rank, num_workers, kw, q),
        )
        p.start()
        procs.append(p)

    merged: dict[int, dict[str, torch.Tensor]] = {}
    try:
        for _ in range(num_workers):
            _r, payload = q.get()
            if isinstance(payload, Exception):
                raise payload
            merged.update(_predictions_from_numpy_after_mp_queue(payload))
    except Exception:
        for pr in procs:
            pr.terminate()
            pr.join(timeout=20)
        raise
    for pr in procs:
        pr.join(timeout=3600)

    return merged


def dump_coco_segmentation_predictions_json(
    predictions: dict[int, dict[str, torch.Tensor]],
    dump_path: Path,
) -> None:
    from sam3.eval.coco_writer import PredictionDumper
    from sam3.eval.postprocessors import PostProcessImage

    dumper = PredictionDumper(
        dump_dir=str(dump_path.parent),
        postprocessor=PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_ids=True,
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=True,
            use_presence=True,
        ),
        maxdets=300,
        iou_type="segm",
        merge_predictions=True,
    )
    results: list[dict[str, Any]] = []
    for image_id, pred in predictions.items():
        if pred["scores"].numel() == 0:
            continue
        results.extend(dumper.prepare({image_id: pred}, "segm"))

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text(json.dumps(results), encoding="utf-8")


def evaluate_coco_offline(
    gt_json: Path,
    pred_json: Path,
    *,
    iou_type: str = "segm",
    positive_split: bool = False,
    tide: bool = False,
) -> dict[str, float]:
    from sam3.eval.coco_eval_offline import CocoEvaluatorOfflineWithPredFileEvaluators

    ev = CocoEvaluatorOfflineWithPredFileEvaluators(
        str(gt_json),
        tide=tide,
        iou_type=iou_type,
        positive_split=positive_split,
    )
    return ev.evaluate(str(pred_json))


def compute_mask_miou_dice(gt_json: Path, pred_json: Path) -> dict[str, float]:
    """Per-GT-instance mask mIoU and Dice after greedy IoU matching within each image/class.

    For each ``(image_id, category_id)``, pairs GT masks to prediction masks by sorting
    all GT–pred pairs by IoU (descending) and taking non-conflicting assignments.
    Unmatched GT instances contribute IoU 0 and Dice 0. Reported ``mask_miou`` /
    ``mask_dice`` are means over **all GT instances** in ``gt_json``.
    """
    from pycocotools.coco import COCO

    coco_gt = COCO(str(gt_json))
    raw = Path(pred_json).read_text(encoding="utf-8").strip()
    if not raw or raw == "[]":
        return {"mask_miou": 0.0, "mask_dice": 0.0}

    coco_dt = coco_gt.loadRes(str(pred_json))

    all_ious: list[float] = []
    all_dices: list[float] = []

    for img_id in coco_gt.getImgIds():
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[img_id]))

        by_cat_gt: dict[int, list] = defaultdict(list)
        by_cat_dt: dict[int, list] = defaultdict(list)
        for a in gt_anns:
            by_cat_gt[int(a["category_id"])].append(a)
        for a in dt_anns:
            by_cat_dt[int(a["category_id"])].append(a)

        for cat_id, ga in by_cat_gt.items():
            da = sorted(
                by_cat_dt.get(cat_id, []),
                key=lambda x: float(x.get("score", 0.0)),
                reverse=True,
            )
            gt_masks = [coco_gt.annToMask(x).astype(bool) for x in ga]
            dt_masks = [coco_dt.annToMask(x).astype(bool) for x in da]

            n, m = len(gt_masks), len(dt_masks)
            pairs: list[tuple[float, float, int, int]] = []
            for i in range(n):
                gm = gt_masks[i]
                for j in range(m):
                    dm = dt_masks[j]
                    inter = np.logical_and(gm, dm).sum(dtype=np.float64)
                    union = np.logical_or(gm, dm).sum(dtype=np.float64)
                    if union <= 0:
                        iou = 1.0 if inter <= 0 else 0.0
                    else:
                        iou = float(inter / union)
                    denom = float(gm.sum() + dm.sum())
                    dice = float((2.0 * inter / denom) if denom > 0 else 0.0)
                    pairs.append((iou, dice, i, j))

            pairs.sort(key=lambda t: -t[0])
            matched_g: set[int] = set()
            matched_d: set[int] = set()
            gt_iou = [0.0] * n
            gt_dice = [0.0] * n
            for iou, dice, i, j in pairs:
                if i in matched_g or j in matched_d:
                    continue
                matched_g.add(i)
                matched_d.add(j)
                gt_iou[i] = iou
                gt_dice[i] = dice

            all_ious.extend(gt_iou)
            all_dices.extend(gt_dice)

    if not all_ious:
        return {"mask_miou": 0.0, "mask_dice": 0.0}
    return {
        "mask_miou": float(np.mean(all_ious)),
        "mask_dice": float(np.mean(all_dices)),
    }


def metrics_to_serializable(m: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in m.items()}
