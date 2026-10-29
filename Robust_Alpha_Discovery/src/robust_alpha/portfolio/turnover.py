"""Turnover relative to post-return drifted prior weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_turnover(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    target_column: str = "target_o2o",
) -> pd.DataFrame:
    """Calculate one-way turnover as half the absolute weight change."""

    weight_matrix = weights.pivot(index="date", columns="asset", values="weight").fillna(0.0)
    return_matrix = panel.pivot(index="date", columns="asset", values=target_column).reindex(
        index=weight_matrix.index, columns=weight_matrix.columns
    )

    if weight_matrix.empty:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "turnover": pd.Series(dtype=float),
            }
        )

    weight_values = weight_matrix.to_numpy(dtype=float)
    return_values = return_matrix.fillna(0.0).to_numpy(dtype=float)
    turnover = np.empty(len(weight_values), dtype=float)
    turnover[0] = 0.5 * np.abs(weight_values[0]).sum()
    if len(weight_values) > 1:
        previous = weight_values[:-1]
        drifted = previous * np.exp(return_values[:-1])

        target_long = np.clip(previous, 0.0, None).sum(axis=1)
        target_short = -np.clip(previous, None, 0.0).sum(axis=1)
        drifted_long = np.clip(drifted, 0.0, None).sum(axis=1)
        drifted_short = -np.clip(drifted, None, 0.0).sum(axis=1)
        long_scale = np.divide(
            target_long,
            drifted_long,
            out=np.ones_like(target_long),
            where=drifted_long > 0,
        )
        short_scale = np.divide(
            target_short,
            drifted_short,
            out=np.ones_like(target_short),
            where=drifted_short > 0,
        )
        drifted = np.where(
            drifted > 0,
            drifted * long_scale[:, None],
            np.where(drifted < 0, drifted * short_scale[:, None], drifted),
        )
        turnover[1:] = 0.5 * np.abs(weight_values[1:] - drifted).sum(axis=1)
    return pd.DataFrame({"date": weight_matrix.index, "turnover": turnover})
