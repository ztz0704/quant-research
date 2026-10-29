"""Asset-specific trailing variance with explicit cross-sectional shrinkage."""

import numpy as np
import pandas as pd

from risk_aware_portfolio.risk.factor_covariance import observation_weights


def estimate_specific_variance(
    history: pd.DataFrame,
    method: str,
    decay: float = 0.94,
    floor: float = 1e-6,
    shrinkage: float = 0.15,
) -> pd.Series:
    """Shrink finite-window residual variance toward the same-date median."""
    values = history.to_numpy()
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Specific variance needs complete historical residuals")
    if not 0 <= shrinkage <= 1 or floor <= 0:
        raise ValueError("Invalid shrinkage or variance floor")
    if method == "sample":
        variance = values.var(axis=0, ddof=1)
    elif method == "ewma":
        weights = observation_weights(len(values), decay)
        centered = values - weights @ values
        variance = weights @ centered**2 / (1 - weights @ weights)
    else:
        raise ValueError("Unknown specific variance method")
    variance = (1 - shrinkage) * variance + shrinkage * np.median(variance)
    return pd.Series(np.maximum(variance, floor), index=history.columns, name="specific_variance")
