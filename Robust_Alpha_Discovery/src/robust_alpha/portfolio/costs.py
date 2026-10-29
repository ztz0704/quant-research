"""Simple, explicit transaction-cost sensitivity model."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def apply_linear_costs(
    portfolio: pd.DataFrame,
    turnover: pd.DataFrame,
    *,
    cost_bps: Iterable[float],
) -> pd.DataFrame:
    """Apply ``net_return = gross_return - cost_rate * turnover`` for each cost level."""

    merged = portfolio.merge(turnover, on="date", how="left", validate="one_to_one")
    rows: list[pd.DataFrame] = []
    for basis_points in cost_bps:
        if basis_points < 0:
            raise ValueError("Transaction costs cannot be negative.")
        scenario = merged.copy()
        scenario["cost_bps"] = float(basis_points)
        scenario["cost"] = scenario["turnover"] * float(basis_points) / 10_000.0
        scenario["net_return"] = scenario["gross_return"] - scenario["cost"]
        rows.append(scenario)
    return pd.concat(rows, ignore_index=True) if rows else merged.iloc[0:0].copy()

