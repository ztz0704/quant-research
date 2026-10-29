"""Leakage-aware feature/target time alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_open_to_open_target(
    frame: pd.DataFrame,
    *,
    target_column: str = "target_o2o",
    require_entry_tradable: bool = True,
    require_exit_tradable: bool = True,
) -> pd.DataFrame:
    """Attach next-open-to-open returns to signal-date rows.

    For a feature known after close on date t, entry is open(t+1), exit is open(t+2), and the label
    is log(exit / entry). Future prices are labels only and are never supplied to feature functions.
    """

    panel = frame.sort_values(["asset", "date"], kind="mergesort").copy()
    grouped = panel.groupby("asset", sort=False, observed=True)
    panel["entry_date"] = grouped["date"].shift(-1)
    panel["exit_date"] = grouped["date"].shift(-2)
    panel["entry_open"] = grouped["open"].shift(-1)
    panel["exit_open"] = grouped["open"].shift(-2)
    panel["entry_tradable"] = grouped["tradable"].shift(-1).fillna(False).astype(bool)
    panel["exit_tradable"] = grouped["tradable"].shift(-2).fillna(False).astype(bool)
    panel[target_column] = np.log(panel["exit_open"] / panel["entry_open"])

    valid = panel["entry_open"].gt(0) & panel["exit_open"].gt(0)
    if require_entry_tradable:
        valid &= panel["entry_tradable"]
    if require_exit_tradable:
        valid &= panel["exit_tradable"]
    panel["target_valid"] = valid & panel[target_column].notna()
    panel.loc[~panel["target_valid"], target_column] = np.nan
    return panel
