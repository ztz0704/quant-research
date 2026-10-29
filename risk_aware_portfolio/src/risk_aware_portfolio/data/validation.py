"""Strict long-panel contract. No implicit sorting, filling, or record dropping."""

import numpy as np
import pandas as pd

REQUIRED = ("date", "asset", "return", "alpha", "market_cap", "industry")


def validate_panel(panel: pd.DataFrame, max_abs_return: float = 0.95) -> None:
    """Reject ambiguous dates, duplicate keys, missing values and invalid returns.

    Optional numeric fields, when present, must also be finite. Missing optional
    columns are allowed; a risk model requiring one fails at its own boundary.
    Rows must be lexicographically ordered by date then asset.
    """
    missing = set(REQUIRED) - set(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if panel.empty:
        raise ValueError("Panel is empty")
    if any(
        not pd.api.types.is_numeric_dtype(panel[column])
        for column in ("return", "alpha", "market_cap")
    ):
        raise ValueError("return, alpha and market_cap must be numeric")
    if not pd.api.types.is_datetime64_any_dtype(panel["date"]):
        raise ValueError("date must be datetime64; loader parses it explicitly")
    if panel["date"].dt.tz is not None:
        raise ValueError("Use timezone-naive, normalized trading dates")
    if not panel["date"].eq(panel["date"].dt.normalize()).all():
        raise ValueError("Dates must be normalized trading dates")
    if panel.duplicated(["date", "asset"]).any():
        raise ValueError("Duplicate (date, asset) records")
    if panel[list(REQUIRED)].isna().any().any():
        raise ValueError("Missing required data; no implicit forward fill")
    if not all(isinstance(x, str) and x for x in panel["asset"]):
        raise ValueError("asset must contain nonempty string identifiers")
    keys = pd.MultiIndex.from_frame(panel[["date", "asset"]])
    if not keys.is_monotonic_increasing:
        raise ValueError("Panel must be ordered by date and asset")
    numeric = panel.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Nonfinite or missing numeric value")
    if (panel["market_cap"] <= 0).any():
        raise ValueError("market_cap must be positive")
    if (panel["return"] <= -1).any() or (panel["return"].abs() > max_abs_return).any():
        raise ValueError("Abnormal simple return; investigate corporate actions")
