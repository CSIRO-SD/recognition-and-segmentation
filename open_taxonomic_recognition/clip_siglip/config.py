"""Paths from environment (HF cache etc.)."""

import os
from dataclasses import dataclass
from pathlib import Path


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", os.environ.get("TRANSFORMERS_CACHE", "~/.cache/huggingface"))).expanduser()


@dataclass
class Paths:
    """Directory layout."""

    csv_path: Path
    images_root: Path
    output_dir: Path


def taxonomy_csv_columns() -> tuple[str, ...]:
    """Expected CSV header (your GT file). Extend if needed."""
    return (
        "image_name",
        "catalog_number",
        "order",
        "family",
        "genus",
        "species",
        "scientific_name",
        "authority",
        "year",
        "locality",
        "original_locality",
        "state_province",
        "verbatim_coordinates",
        "latitude",
        "longitude",
        "collection_date",
        "life_stage",
        "sex",
        "collector",
        "determiner",
        "determination_date",
        "notes_clean",
        "split",
        "species_split",
        "taxon_key",
    )
