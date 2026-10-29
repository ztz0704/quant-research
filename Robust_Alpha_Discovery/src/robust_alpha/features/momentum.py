"""Momentum-family feature functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(close: pd.Series, asset: pd.Series, *, horizon: int) -> pd.Series:
    """Log close-to-close momentum ending on the current signal date."""

    log_close = np.log(close)
    grouped = log_close.groupby(asset, sort=False, observed=True)
    return log_close - grouped.shift(horizon)


def skip_momentum(
    close: pd.Series,
    asset: pd.Series,
    *,
    horizon: int,
    skip: int,
) -> pd.Series:
    """Log momentum ending ``skip`` sessions before the current signal date."""

    if not 0 < skip < horizon:
        raise ValueError("skip must be positive and smaller than horizon.")
    log_close = np.log(close)
    grouped = log_close.groupby(asset, sort=False, observed=True)
    return grouped.shift(skip) - grouped.shift(horizon)


def risk_adjusted_momentum(
    close: pd.Series,
    asset: pd.Series,
    *,
    horizon: int,
    vol_window: int,
) -> pd.Series:
    """Momentum divided by trailing realized volatility."""

    raw_momentum = momentum(close, asset, horizon=horizon)
    log_return = np.log(close).groupby(asset, sort=False, observed=True).diff()
    volatility = log_return.groupby(asset, sort=False, observed=True).transform(
        lambda values: values.rolling(vol_window, min_periods=vol_window).std(ddof=1)
    )
    return raw_momentum / volatility.replace(0.0, np.nan)

