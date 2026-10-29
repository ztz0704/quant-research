"""Asset-local trailing realized volatility."""

import numpy as np
import pandas as pd


def realized_volatility(returns: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Return RMS of trailing event returns (return units, not annualized)."""
    if not 1 <= min_periods <= window:
        raise ValueError("Require 1 <= min_periods <= rolling window")
    return np.sqrt(returns.pow(2).rolling(window, min_periods=min_periods).mean())
