"""Leakage-safe cross-sectional transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


def _rank_gaussian(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if valid.empty:
        return result
    ranks = valid.rank(method="average")
    uniforms = ranks / (len(valid) + 1.0)
    result.loc[valid.index] = norm.ppf(uniforms)
    return result


def _percentile_rank(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if valid.empty:
        return result
    result.loc[valid.index] = valid.rank(method="average") / (len(valid) + 1.0)
    return result


def _zscore(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if len(valid) < 2:
        return result
    standard_deviation = valid.std(ddof=1)
    if standard_deviation == 0 or np.isnan(standard_deviation):
        result.loc[valid.index] = 0.0
    else:
        result.loc[valid.index] = (valid - valid.mean()) / standard_deviation
    return result


def cross_sectional_transform(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    *,
    method: str = "rank_gaussian",
    eligible_column: str = "eligible",
    prefix: str = "z__",
) -> pd.DataFrame:
    """Transform signals using only contemporaneous eligible observations."""

    transforms = {
        "rank_gaussian": _rank_gaussian,
        "percentile_rank": _percentile_rank,
        "zscore": _zscore,
    }
    if method not in transforms:
        raise ValueError(f"Unknown cross-sectional transform: {method}")
    missing = [name for name in signal_columns if name not in frame]
    if missing:
        raise KeyError(f"Missing signal columns: {missing}")
    if eligible_column not in frame:
        raise KeyError(f"Missing eligibility column: {eligible_column}")

    panel = frame.copy()
    eligible = panel[eligible_column].astype(bool)
    transform = transforms[method]
    for signal in signal_columns:
        input_values = panel[signal].where(eligible)
        panel[f"{prefix}{signal}"] = input_values.groupby(
            panel["date"], sort=False, observed=True, group_keys=False
        ).transform(transform)
    return panel

