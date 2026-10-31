"""Load GT CSV; filter by ``species_split`` (train | seen | unseen | test …)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import pandas as pd

SplitName = Literal["train", "seen", "unseen", "test"]


@dataclass
class Sample:
    """One row wired for dataloaders."""

    image_path: Path
    taxon_key: str
    species_split: str
    extras: dict  # locality, notes_clean, scientific_name, …


class TaxonomyTable:
    def __init__(self, df: pd.DataFrame, images_root: Path | str):
        self.df = df
        self.images_root = Path(images_root)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        images_root: str | Path,
        sep: str = ",",
    ) -> TaxonomyTable:
        df = pd.read_csv(csv_path, sep=sep, low_memory=False)
        df["species_split"] = df["species_split"].astype(str).str.strip()
        return cls(df=df, images_root=images_root)

    def subset(
        self,
        species_splits: Sequence[str] | None = None,
        split_assignments: Sequence[str] | None = None,
        split_column: str = "split",
    ) -> pd.DataFrame:
        """Filter rows.

        ``species_splits``: values of ``species_split`` (seen | unseen).
        ``split_assignments``: values of ``split`` column (train | val | test_seen | test_unseen).
        Both filters apply when provided (logical AND).
        """
        import pandas as pd

        m = pd.Series(True, index=self.df.index)
        if species_splits:
            m &= self.df["species_split"].isin(list(species_splits))
        if split_assignments:
            if split_column not in self.df.columns:
                raise KeyError(f"CSV has no column {split_column!r}; cannot filter eval split.")
            col = self.df[split_column].astype(str).str.strip()
            m &= col.isin(list(split_assignments))
        return self.df.loc[m].copy()

    def iter_samples(self, species_splits: Sequence[str]) -> Iterable[Sample]:
        sub = self.subset(species_splits=species_splits)
        for _, row in sub.iterrows():
            name = row["image_name"]
            img = self.images_root / str(name)
            extras = {k: row[k] for k in row.index if k not in ("image_name", "taxon_key", "species_split")}
            yield Sample(
                image_path=img,
                taxon_key=str(row["taxon_key"]),
                species_split=str(row["species_split"]),
                extras=extras,
            )


def vocab_for_eval(df_train: pd.DataFrame) -> list[str]:
    """Seen train taxon_keys for classifier head / prompting (closed eval on train vocab)."""
    return sorted(df_train["taxon_key"].dropna().astype(str).unique().tolist())


def unseen_and_test_specs() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Default names; align with your CSV if you use synonyms."""
    unseen = ("unseen",)
    test = ("test",)
    return unseen, test
