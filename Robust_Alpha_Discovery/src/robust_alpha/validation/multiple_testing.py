"""Transparent family-wise and false-discovery-rate adjustments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def adjust_pvalues(pvalues: pd.Series, *, alpha: float = 0.05) -> pd.DataFrame:
    """Compare naive, Bonferroni, and Benjamini-Hochberg decisions."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    values = pd.Series(pvalues, dtype=float).dropna()
    if ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be in [0, 1].")
    count = len(values)
    if count == 0:
        return pd.DataFrame(
            columns=[
                "signal",
                "pvalue",
                "naive_reject",
                "bonferroni_pvalue",
                "bonferroni_reject",
                "bh_pvalue",
                "bh_reject",
            ]
        )

    order = np.argsort(values.to_numpy(), kind="mergesort")
    sorted_values = values.to_numpy()[order]
    ranks = np.arange(1, count + 1)
    bh_sorted = sorted_values * count / ranks
    bh_sorted = np.minimum.accumulate(bh_sorted[::-1])[::-1]
    bh_sorted = np.clip(bh_sorted, 0.0, 1.0)
    bh_adjusted = np.empty(count, dtype=float)
    bh_adjusted[order] = bh_sorted

    result = pd.DataFrame(
        {
            "signal": values.index.astype(str),
            "pvalue": values.to_numpy(),
            "naive_reject": values.to_numpy() <= alpha,
            "bonferroni_pvalue": np.minimum(values.to_numpy() * count, 1.0),
            "bh_pvalue": bh_adjusted,
        }
    )
    result["bonferroni_reject"] = result["bonferroni_pvalue"] <= alpha
    result["bh_reject"] = result["bh_pvalue"] <= alpha
    return result.sort_values(["pvalue", "signal"], kind="mergesort").reset_index(drop=True)

