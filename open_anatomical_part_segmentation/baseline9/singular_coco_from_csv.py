"""Build COCO GT from balanced split CSV using singular-style LabelMe JSONs.

Prefer ``cleaned_singular_json`` when present on disk (per task docs). Many deployments
expose the same basename under ``noun_only_json`` instead — that folder is tried next
automatically because ``cleaned_singular_json`` may exist but be empty."""

from __future__ import annotations

import csv
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

# Order matters: task.md names cleaned_singular_json first; noun_only_json is populated
# on datasets where singular exports live there instead.
DEFAULT_SINGULAR_REL_DIRS: tuple[str, ...] = (
    "cleaned_singular_json",
    "noun_only_json",
)


def _load_export_sam3_coco(split_csv_dir: Path) -> Any:
    script = split_csv_dir / "export_sam3_coco.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"Need export_sam3_coco.py next to split CSV directory: {script}"
        )
    spec = importlib.util.spec_from_file_location("export_sam3_coco", script)
    if spec is None or spec.loader is None:
        raise ImportError(script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CSV_SPLIT_ALIASES: dict[str, str] = {
    "training": "training",
    "train": "training",
    "val": "val",
    "validation": "val",
    "test": "test",
}


def canonical_csv_split(name: str) -> str:
    k = name.strip().lower()
    if k not in _CSV_SPLIT_ALIASES:
        raise ValueError(
            f"Unknown split name {name!r}; expected one of {sorted(set(_CSV_SPLIT_ALIASES))}"
        )
    return _CSV_SPLIT_ALIASES[k]


def resolve_singular_labelme_path(
    cleaned_json_path: Path,
    *,
    relative_dirs: Sequence[str] | None = None,
) -> Path | None:
    """Given ``…/all_crops/cleaned_json/<stem>.json``, find the same ``<stem>`` under
    the first existing directory in ``relative_dirs`` (siblings of ``cleaned_json``).
    """
    rel = list(relative_dirs or DEFAULT_SINGULAR_REL_DIRS)
    if not rel:
        raise ValueError("relative_dirs must be non-empty")

    if cleaned_json_path.parent.name != "cleaned_json":
        raise ValueError(
            f"Expected CSV json_path under a cleaned_json directory: {cleaned_json_path}"
        )

    crops_dir = cleaned_json_path.parent.parent
    stem = cleaned_json_path.name
    for sub in rel:
        cand = crops_dir / sub / stem
        if cand.is_file():
            return cand
    return None


def collect_singular_paths_for_csv(
    split_csv: Path,
    *,
    singular_relative_dirs: Sequence[str] | None = None,
) -> tuple[dict[str, list[Path]], list[Path]]:
    """Bucket CSV rows by split; paths are resolved singular LabelMe JSONs that exist."""
    rel = tuple(singular_relative_dirs or DEFAULT_SINGULAR_REL_DIRS)
    by_bucket: dict[str, list[Path]] = defaultdict(list)
    all_existing: list[Path] = []

    with split_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if (
            reader.fieldnames is None
            or "json_path" not in reader.fieldnames
            or "split" not in reader.fieldnames
        ):
            raise ValueError(f"{split_csv}: need columns json_path,split")

        for row in reader:
            raw = row["split"].strip().lower()
            if raw in ("training", "train"):
                bucket = "training"
            elif raw in ("val", "validation"):
                bucket = "val"
            elif raw == "test":
                bucket = "test"
            else:
                bucket = raw

            jp_clean = Path(row["json_path"].strip())
            if not jp_clean.is_file():
                continue
            jp_sing = resolve_singular_labelme_path(
                jp_clean, relative_dirs=rel
            )
            if jp_sing is None:
                continue

            by_bucket[bucket].append(jp_sing)
            all_existing.append(jp_sing)

    for k in list(by_bucket.keys()):
        by_bucket[k] = sorted(set(by_bucket[k]), key=lambda p: p.name)

    all_unique = sorted(set(all_existing), key=lambda p: p.name)
    return by_bucket, all_unique


def build_coco_for_singular_balanced_split(
    *,
    split_csv: Path,
    norm_map: Path,
    split_name: str,
    singular_relative_dirs: Sequence[str] | None = None,
) -> dict:
    """COCO dict for one split; categories from all splits' singular JSONs (consistent ids)."""
    mod = _load_export_sam3_coco(split_csv.parent.resolve())
    norm = mod.load_norm_map(norm_map)

    rel = tuple(singular_relative_dirs or DEFAULT_SINGULAR_REL_DIRS)
    by_bucket, all_paths = collect_singular_paths_for_csv(
        split_csv, singular_relative_dirs=rel
    )
    if not all_paths:
        raise ValueError(
            f"No singular-style LabelMe files found for CSV rows in {split_csv}. "
            f"Tried, under each image's all_crops parent: {list(rel)}. "
            f"Ensure at least one of those directories contains JSON basenames matching "
            f"the cleaned_json entries, or pass --singular-subdirs."
        )

    wanted = canonical_csv_split(split_name)
    paths = by_bucket.get(wanted, [])
    if not paths:
        raise ValueError(
            f"No images for split {wanted!r} with existing singular JSON in {split_csv}."
        )

    categories = mod.collect_categories(all_paths, norm)
    return mod.labelme_to_coco_for_split(paths, categories, norm, split_name=wanted)


def write_singular_gt_json(
    split_csv: Path,
    norm_map: Path,
    split_name: str,
    dest: Path,
    *,
    singular_relative_dirs: Sequence[str] | None = None,
) -> dict:
    import json

    coco = build_coco_for_singular_balanced_split(
        split_csv=split_csv,
        norm_map=norm_map,
        split_name=split_name,
        singular_relative_dirs=singular_relative_dirs,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(coco, ensure_ascii=False) + "\n", encoding="utf-8")
    return coco
