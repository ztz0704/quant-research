"""Structure-preserving null transformations for pipeline diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def circular_shift_signals_within_asset(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    *,
    minimum_shift: int = 60,
    seed: int = 0,
) -> pd.DataFrame:
    """Break signal/return alignment with asset-specific large circular date shifts."""

    if minimum_shift < 1:
        raise ValueError("minimum_shift must be positive.")
    rng = np.random.default_rng(seed)
    shifted = frame.sort_values(["asset", "date"], kind="mergesort").copy()
    for _, indices in shifted.groupby("asset", sort=False, observed=True).groups.items():
        locations = np.asarray(indices)
        length = len(locations)
        if length <= 2 * minimum_shift:
            raise ValueError("Each asset needs more than twice minimum_shift observations.")
        offset = int(rng.integers(minimum_shift, length - minimum_shift + 1))
        for signal in signal_columns:
            shifted.loc[locations, signal] = np.roll(
                shifted.loc[locations, signal].to_numpy(), offset
            )
    return shifted

