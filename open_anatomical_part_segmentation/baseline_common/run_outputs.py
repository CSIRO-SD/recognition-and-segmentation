"""Timestamped output paths so concurrent or repeated runs do not overwrite files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_id(explicit: str | None = None) -> str:
    """Return ``explicit`` if set, else ``YYYYMMDD_HHMMSS``."""
    return explicit or datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_eval_output_dir(
    out_dir: Path,
    *,
    run_id: str | None = None,
    flat: bool = False,
) -> tuple[Path, str]:
    """Directory for eval artifacts and the run id string.

    Default: ``out_dir / <run_id>`` (one subfolder per run).
    With ``flat=True``: use ``out_dir`` and embed ``run_id`` in filenames instead.
    """
    rid = make_run_id(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    if flat:
        return out_dir, rid
    d = out_dir / rid
    d.mkdir(parents=True, exist_ok=True)
    return d, rid
