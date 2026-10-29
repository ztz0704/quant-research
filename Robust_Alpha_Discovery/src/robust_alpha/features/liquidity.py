"""Liquidity and volume features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def negative_amihud(
    close: pd.Series,
    amount: pd.Series,
    asset: pd.Series,
    *,
    window: int,
    scale: float = 1e8,
) -> pd.Series:
    """Negative trailing Amihud illiquidity proxy."""

    log_return = np.log(close).groupby(asset, sort=False, observed=True).diff().abs()
    daily = log_return / amount.where(amount > 0)
    trailing = daily.groupby(asset, sort=False, observed=True).transform(
        lambda values: values.rolling(window, min_periods=window).mean()
    )
    return -scale * trailing


def volume_shock(
    volume: pd.Series,
    asset: pd.Series,
    *,
    short: int,
    long: int,
) -> pd.Series:
    """Log ratio of short- and long-window average volume."""

    grouped = volume.groupby(asset, sort=False, observed=True)
    short_mean = grouped.transform(
        lambda values: values.rolling(short, min_periods=short).mean()
    )
    long_mean = grouped.transform(lambda values: values.rolling(long, min_periods=long).mean())
    ratio = short_mean / long_mean.where(long_mean > 0)
    # A zero-volume window cannot supply a finite logarithmic factor value.
    return np.log(ratio.where(ratio > 0))
