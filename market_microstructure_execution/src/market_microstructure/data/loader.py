"""Read/write CSV or Parquet snapshots through the same validation boundary."""

from pathlib import Path

import pandas as pd

from market_microstructure.data.validation import ValidationReport, validate_orderbook


def load_orderbook(path: str | Path, **validation_options) -> tuple[pd.DataFrame, ValidationReport]:
    """Load vendor-neutral data; caller receives the removal audit even in drop mode."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Only CSV and Parquet book inputs supported")
    return validate_orderbook(frame, **validation_options)


def save_orderbook(frame: pd.DataFrame, path: str | Path) -> None:
    """Validate and persist a book without implicitly dropping rows."""
    clean, _ = validate_orderbook(frame)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        clean.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        clean.to_csv(path, index=False)
    else:
        raise ValueError("Only CSV and Parquet book outputs supported")
