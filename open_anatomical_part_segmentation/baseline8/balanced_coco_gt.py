"""Build COCO-style GT JSON for a CSV split row (balanced LabelMe merged export)."""

from __future__ import annotations

import csv
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any

_CSV_SPLIT_ALIASES: dict[str, str] = {
    "training": "training",
    "train": "training",
    "val": "val",
    "validation": "val",
    "test": "test",
}


def _load_export_sam3_coco(split_csv_dir: Path) -> Any:
    """Load labelingv3 ``export_sam3_coco`` as a module (not a package)."""
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


def canonical_csv_split(name: str) -> str:
    k = name.strip().lower()
    if k not in _CSV_SPLIT_ALIASES:
        raise ValueError(f"Unknown split name {name!r}; expected one of {sorted(set(_CSV_SPLIT_ALIASES))}")
    return _CSV_SPLIT_ALIASES[k]


def build_coco_for_balanced_csv_split(
    *,
    split_csv: Path,
    norm_map: Path,
    split_name: str,
) -> dict:
    """Return COCO dict for all LabelMe JSONs in ``split_csv`` with the given split tag.

    Category vocabulary is collected from **all** valid rows in the CSV (same as
    ``export_sam3_coco.py``), so splits share consistent ``category_id`` space.
    """
    mod = _load_export_sam3_coco(split_csv.parent.resolve())
    norm = mod.load_norm_map(norm_map)

    by_bucket: dict[str, list[Path]] = defaultdict(list)
    all_paths: list[Path] = []

    with split_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "json_path" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError(f"{split_csv}: need columns json_path,split")

        for row in reader:
            jp = Path(row["json_path"].strip())
            if not jp.is_file():
                continue
            raw_sp = row["split"].strip().lower()
            if raw_sp in ("training", "train"):
                bucket = "training"
            elif raw_sp in ("val", "validation"):
                bucket = "val"
            elif raw_sp == "test":
                bucket = "test"
            else:
                bucket = raw_sp
            by_bucket[bucket].append(jp)
            all_paths.append(jp)

    wanted = canonical_csv_split(split_name)
    paths = sorted(set(by_bucket.get(wanted, [])), key=lambda p: p.name)

    if not all_paths:
        raise ValueError(f"No valid LabelMe paths found (files missing?) in {split_csv}")

    if not paths:
        raise ValueError(
            f"No images for split {wanted!r} in {split_csv} (empty bucket or all files missing)."
        )

    categories = mod.collect_categories(all_paths, norm)
    return mod.labelme_to_coco_for_split(paths, categories, norm, split_name=wanted)


def write_balanced_gt_json(split_csv: Path, norm_map: Path, split_name: str, dest: Path) -> dict:
    """Build COCO and write UTF-8 JSON to ``dest``."""
    import json

    coco = build_coco_for_balanced_csv_split(
        split_csv=split_csv, norm_map=norm_map, split_name=split_name
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(coco, ensure_ascii=False) + "\n", encoding="utf-8")
    return coco
