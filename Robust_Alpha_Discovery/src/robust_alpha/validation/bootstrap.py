"""Joint moving-block bootstrap max-statistic calibration."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _t_statistics(values: np.ndarray) -> np.ndarray:
    count = values.shape[0]
    means = np.nanmean(values, axis=0)
    standard_deviations = np.nanstd(values, axis=0, ddof=1)
    denominator = standard_deviations / np.sqrt(count)
    return np.divide(
        means,
        denominator,
        out=np.full_like(means, np.nan, dtype=float),
        where=denominator > 0,
    )


def _moving_block_indices(
    n_observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    block_count = math.ceil(n_observations / block_length)
    maximum_start = n_observations - block_length
    starts = rng.integers(0, maximum_start + 1, size=block_count)
    indices = np.concatenate(
        [np.arange(start, start + block_length, dtype=int) for start in starts]
    )
    return indices[:n_observations]


def joint_moving_block_max_t(
    daily_factor_returns: pd.DataFrame,
    *,
    replications: int = 2000,
    block_length: int = 20,
    seed: int = 0,
    two_sided: bool = True,
) -> dict[str, Any]:
    """Calibrate the maximum t-statistic while preserving joint signal dependence.

    Rows (dates) are sampled jointly for all signal columns. Series are centered under the global
    null before resampling. Complete joint rows are used so every bootstrap draw has a common time
    index and candidate-set definition.
    """

    if replications < 1:
        raise ValueError("replications must be positive.")
    matrix = daily_factor_returns.dropna(axis=0, how="any").astype(float)
    if matrix.shape[1] < 1:
        raise ValueError("At least one signal column is required.")
    if block_length < 1 or block_length > len(matrix):
        raise ValueError("block_length must be between 1 and the number of complete dates.")
    if len(matrix) < 3:
        raise ValueError("At least three complete dates are required.")

    observed_t = _t_statistics(matrix.to_numpy())
    centered = matrix.to_numpy() - matrix.to_numpy().mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    maximum_statistics = np.empty(replications, dtype=float)
    for replication in range(replications):
        indices = _moving_block_indices(len(matrix), block_length, rng)
        bootstrap_t = _t_statistics(centered[indices])
        statistic = np.abs(bootstrap_t) if two_sided else bootstrap_t
        maximum_statistics[replication] = np.nanmax(statistic)

    observed_comparison = np.abs(observed_t) if two_sided else observed_t
    observed_maximum = float(np.nanmax(observed_comparison))
    global_pvalue = float(
        (1 + np.count_nonzero(maximum_statistics >= observed_maximum)) / (replications + 1)
    )
    adjusted = (1 + (maximum_statistics[:, None] >= observed_comparison).sum(axis=0)) / (
        replications + 1
    )
    signal_results = pd.DataFrame(
        {
            "signal": matrix.columns.astype(str),
            "observed_t_stat": observed_t,
            "max_t_adjusted_pvalue": adjusted,
        }
    ).sort_values("signal")
    return {
        "signal_results": signal_results.reset_index(drop=True),
        "max_statistics": maximum_statistics,
        "observed_maximum": observed_maximum,
        "global_pvalue": global_pvalue,
        "n_complete_dates": len(matrix),
        "block_length": block_length,
        "replications": replications,
        "two_sided": two_sided,
    }

