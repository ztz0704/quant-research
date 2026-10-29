"""Trailing sample and normalized finite-window EWMA covariance estimators."""

import numpy as np
import pandas as pd


def observation_weights(n: int, decay: float) -> np.ndarray:
    """Recent observations receive larger weights, summing exactly to one."""
    if n < 2 or not 0 < decay < 1:
        raise ValueError("Invalid EWMA inputs")
    weights = decay ** np.arange(n - 1, -1, -1, dtype=float)
    return weights / weights.sum()


def estimate_covariance(history: pd.DataFrame, method: str, decay: float = 0.94) -> pd.DataFrame:
    """Estimate daily covariance on an already as-of-sliced trailing window.

    EWMA uses a weighted mean and finite-window Bessel correction. It is a windowed
    centered estimator, rather than an uncentered infinite recursive second moment.
    """
    values = history.to_numpy()
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Covariance needs at least two complete observations")
    if method == "sample":
        covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    elif method == "ewma":
        weights = observation_weights(len(values), decay)
        centered = values - weights @ values
        covariance = (centered.T * weights) @ centered / (1 - weights @ weights)
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    return pd.DataFrame(covariance, index=history.columns, columns=history.columns)
