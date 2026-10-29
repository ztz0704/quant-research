"""Supplier-independent parquet/CSV loading and one-to-one alpha integration."""

from pathlib import Path

import pandas as pd

from risk_aware_portfolio.data.validation import validate_panel


def read_frame(path: Path) -> pd.DataFrame:
    """Read parquet or CSV with asset identifiers preserved as strings."""
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path, dtype={"asset": str})
    else:
        raise ValueError("Only .parquet and .csv are supported")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def load_panel(
    path: Path, alpha_path: Path | None = None, max_abs_return: float = 0.95
) -> pd.DataFrame:
    """Merge external signals without fill or fallback to stale signals."""
    panel = read_frame(path)
    if alpha_path is not None:
        alpha = read_frame(alpha_path)
        if set(("date", "asset", "alpha")) - set(alpha.columns):
            raise ValueError("External alpha requires date, asset, alpha")
        if alpha.duplicated(["date", "asset"]).any():
            raise ValueError("Duplicate external alpha records")
        metadata = [c for c in ("return_convention", "return_unit", "model_fit_cutoff",
                                "calibration_fit_cutoff") if c in alpha]
        # Retain protocol metadata at the engine boundary. Never erase it while
        # reducing an external forecast to the three legacy score columns.
        for column in metadata:
            if column in panel:
                raise ValueError(f"Ambiguous external signal metadata: {column}")
        panel = panel.drop(columns="alpha", errors="ignore").merge(
            alpha[["date", "asset", "alpha", *metadata]],
            on=["date", "asset"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
    validate_panel(panel, max_abs_return)
    return panel
