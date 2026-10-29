"""Point-in-time universe eligibility construction."""

from __future__ import annotations

import pandas as pd


def add_eligibility(frame: pd.DataFrame, *, min_history: int = 1) -> pd.DataFrame:
    """Add a signal-date eligibility mask without looking forward.

    History is counted from non-missing closes observed for the same asset up to and including the
    current date. Membership and tradeability remain point-in-time input fields.
    """

    if min_history < 1:
        raise ValueError("min_history must be positive.")
    panel = frame.sort_values(["asset", "date"], kind="mergesort").copy()
    valid_close = panel["close"].notna().astype(int)
    panel["history_count"] = valid_close.groupby(panel["asset"], sort=False).cumsum()
    panel["eligible"] = (
        panel["universe_member"].astype(bool)
        & panel["tradable"].astype(bool)
        & panel["close"].notna()
        & (panel["history_count"] >= min_history)
    )
    return panel

