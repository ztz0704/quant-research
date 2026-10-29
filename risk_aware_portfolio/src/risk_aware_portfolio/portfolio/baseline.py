"""Transparent alpha rank baseline; it does not inherit optimizer constraints."""

import numpy as np
import pandas as pd


def rank_portfolio(alpha: pd.Series, quantile: float = 0.2, long_only: bool = True) -> pd.Series:
    """Equal-weight top quantile; long/short uses +0.5/-0.5 gross legs.

    Stable mergesort makes ties deterministic in already sorted asset order.
    """
    if not 0 < quantile <= 0.5 or not np.isfinite(alpha).all():
        raise ValueError("Invalid rank inputs")
    count = max(1, int(np.ceil(len(alpha) * quantile)))
    order = np.argsort(alpha.to_numpy(), kind="stable")
    weights = np.zeros(len(alpha))
    weights[order[-count:]] = (1.0 if long_only else 0.5) / count
    if not long_only:
        weights[order[:count]] = -0.5 / count
    return pd.Series(weights, index=alpha.index, name="weight")


def equal_weight_benchmark(assets: pd.Index, long_only: bool = True) -> pd.Series:
    """Fully invested long-only benchmark or cash benchmark for dollar neutral research."""
    return pd.Series((1.0 / len(assets)) if long_only else 0.0, index=assets)
