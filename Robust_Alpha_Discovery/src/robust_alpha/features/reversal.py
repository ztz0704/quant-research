"""Short-horizon reversal features."""

from __future__ import annotations

import pandas as pd

from robust_alpha.features.momentum import momentum


def reversal(close: pd.Series, asset: pd.Series, *, horizon: int) -> pd.Series:
    """Negative trailing log return, with positive values denoting prior losers."""

    return -momentum(close, asset, horizon=horizon)

