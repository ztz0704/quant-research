"""Price-relative features."""

from __future__ import annotations

import pandas as pd


def moving_average_gap(
    close: pd.Series,
    asset: pd.Series,
    *,
    window: int,
) -> pd.Series:
    """Current close divided by its trailing moving average, minus one."""

    moving_average = close.groupby(asset, sort=False, observed=True).transform(
        lambda values: values.rolling(window, min_periods=window).mean()
    )
    return close / moving_average - 1.0

