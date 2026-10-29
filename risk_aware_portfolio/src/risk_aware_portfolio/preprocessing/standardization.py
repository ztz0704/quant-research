"""Cross-sectional standardization without fitted full-sample statistics."""

import numpy as np


def zscore(values: np.ndarray) -> np.ndarray:
    """Use population cross-sectional standard deviation on the supplied date only."""
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Standardization requires finite values")
    std = x.std()
    return np.zeros_like(x) if std <= np.finfo(float).eps else (x - x.mean()) / std
