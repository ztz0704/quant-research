"""Same-date cross-sectional robust clipping."""

import numpy as np


def mad_winsorize(values: np.ndarray, multiplier: float = 5.0) -> np.ndarray:
    """Clip around the median using scaled MAD; constant inputs remain constant."""
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all() or multiplier <= 0:
        raise ValueError("Winsorization requires finite values and positive multiplier")
    center = np.median(x)
    mad = np.median(np.abs(x - center))
    if mad <= np.finfo(float).eps:
        return np.full_like(x, center)
    width = multiplier * 1.4826 * mad
    return np.clip(x, center - width, center + width)
