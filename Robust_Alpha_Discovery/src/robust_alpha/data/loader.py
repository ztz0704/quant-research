"""File-format-aware panel loading and saving."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_alpha.data.schema import normalize_and_validate_panel


def load_panel(
    path: str | Path,
    *,
    validate: bool = True,
    date_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load CSV or Parquet daily panel data."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Panel data not found: {source}")
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        filters = None
        if date_end is not None:
            filters = [("date", "<=", pd.Timestamp(date_end))]
        frame = pd.read_parquet(source, filters=filters)
    elif suffix in {".csv", ".csv.gz"} or source.name.lower().endswith(".csv.gz"):
        if date_end is not None:
            raise ValueError(
                "A CSV cannot enforce a pre-read final-holdout boundary. Convert it to a "
                "date-partitioned or filterable Parquet dataset first."
            )
        frame = pd.read_csv(source, dtype={"asset": "string"})
    else:
        raise ValueError(f"Unsupported panel format: {source.suffix}")
    return normalize_and_validate_panel(frame) if validate else frame


def save_panel(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist a panel in CSV or Parquet format, creating parent directories."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(destination, index=False)
    elif suffix == ".csv":
        frame.to_csv(destination, index=False)
    else:
        raise ValueError(f"Unsupported output panel format: {destination.suffix}")
    return destination
