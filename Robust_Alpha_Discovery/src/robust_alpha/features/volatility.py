"""Trailing volatility and range features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def negative_volatility(close: pd.Series, asset: pd.Series, *, window: int) -> pd.Series:
    """Negative trailing standard deviation of log close returns."""

    log_return = np.log(close).groupby(asset, sort=False, observed=True).diff()
    realized = log_return.groupby(asset, sort=False, observed=True).transform(
        lambda values: values.rolling(window, min_periods=window).std(ddof=1)
    )
    return -realized


def negative_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    asset: pd.Series,
    *,
    window: int,
) -> pd.Series:
    """Negative trailing mean of the daily high-low range scaled by close."""

    daily_range = (high - low) / close
    trailing = daily_range.groupby(asset, sort=False, observed=True).transform(
        lambda values: values.rolling(window, min_periods=window).mean()
    )
    return -trailing

