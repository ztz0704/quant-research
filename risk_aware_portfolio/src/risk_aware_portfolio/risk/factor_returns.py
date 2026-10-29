"""Cross-sectional OLS/WLS using lagged exposures and a rank-revealing solve."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import lstsq


@dataclass(frozen=True)
class FactorRegression:
    factor_returns: pd.Series
    specific_returns: pd.Series
    diagnostics: dict[str, float]


def estimate_factor_returns(
    exposures: pd.DataFrame,
    returns: pd.Series,
    market_cap: pd.Series,
    method: str = "wls",
    weight_power: float = 0.5,
) -> FactorRegression:
    """Estimate returns with exposures/caps known before the realized return period.

    WLS minimizes sum(cap**weight_power * residual**2). Multiplication by sqrt of
    these normalized weights is the linear-algebra transform, not an extra cap weight.
    """
    if not exposures.index.equals(returns.index) or not returns.index.equals(market_cap.index):
        raise ValueError("Asset alignment mismatch")
    x, y = exposures.to_numpy(), returns.to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (market_cap <= 0).any():
        raise ValueError("Invalid regression inputs")
    if method not in {"ols", "wls"}:
        raise ValueError("Unknown regression method")
    weights = np.ones(len(y)) if method == "ols" else market_cap.to_numpy() ** weight_power
    weights /= weights.mean()
    root = np.sqrt(weights)
    design, response = x * root[:, None], y * root
    coefficients, _, rank, _ = lstsq(design, response, lapack_driver="gelsy")
    if rank != x.shape[1] or len(y) <= rank:
        raise ValueError(f"Unidentified factor model: rank={rank}, shape={x.shape}")
    residual = y - x @ coefficients
    sse = float(weights @ residual**2)
    centered = y - np.average(y, weights=weights)
    tss = float(weights @ centered**2)
    diagnostics = {
        "rank": float(rank),
        "n_assets": float(len(y)),
        "condition_number": float(np.linalg.cond(design)),
        "weighted_r_squared": 1 - sse / tss if tss else np.nan,
        "residual_std": float(np.sqrt(sse / (len(y) - rank))),
    }
    return FactorRegression(
        pd.Series(coefficients, index=exposures.columns),
        pd.Series(residual, index=returns.index),
        diagnostics,
    )
