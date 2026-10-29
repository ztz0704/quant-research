"""Equal-weight quantile portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_weights(
    frame: pd.DataFrame,
    signal_column: str,
    *,
    quantile: float = 0.10,
    kind: str = "long_short",
    eligible_column: str = "eligible",
    min_assets: int = 1,
) -> pd.DataFrame:
    """Create deterministic signal-date weights without using future execution state."""

    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5].")
    if kind not in {"long_short", "long_only"}:
        raise ValueError("kind must be 'long_short' or 'long_only'.")
    if isinstance(min_assets, bool) or not isinstance(min_assets, int) or min_assets < 1:
        raise ValueError("min_assets must be a positive integer.")
    required = {"date", "asset", signal_column, eligible_column}
    missing = sorted(required - set(frame))
    if missing:
        raise KeyError(f"Missing portfolio inputs: {missing}")

    result = frame[["date", "asset"]].copy()
    result["weight"] = 0.0
    result["signal"] = signal_column
    result["portfolio"] = kind
    eligibility = frame[eligible_column].astype(bool) & frame[signal_column].notna()
    eligible = frame.loc[eligibility, ["date", "asset", signal_column]].copy()
    # The input panel has a deterministic asset/date order, so rank(method="first") also gives a
    # deterministic tie break without a Python loop over dates.
    grouped = eligible.groupby("date", sort=False, observed=True)[signal_column]
    eligible["rank"] = grouped.rank(method="first", ascending=True)
    eligible["count"] = grouped.transform("count")
    eligible["selected_count"] = np.floor(eligible["count"] * quantile).clip(lower=1)
    minimum_count = max(min_assets, 2 if kind == "long_short" else 1)
    sufficient = eligible["count"].ge(minimum_count)
    top = sufficient & (eligible["rank"] > eligible["count"] - eligible["selected_count"])
    bottom = sufficient & (eligible["rank"] <= eligible["selected_count"])
    result.loc[eligible.index[top], "weight"] = (
        1.0 / eligible.loc[top, "selected_count"].to_numpy()
    )
    if kind == "long_short":
        result.loc[eligible.index[bottom], "weight"] = (
            -1.0 / eligible.loc[bottom, "selected_count"].to_numpy()
        )
    return result


def portfolio_returns(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    target_column: str = "target_o2o",
) -> pd.DataFrame:
    """Calculate portfolio simple returns from asset log-return targets."""

    required_weights = {"date", "asset", "weight"}
    required_panel = {"date", "asset", target_column}
    if not required_weights.issubset(weights):
        raise KeyError(f"weights must contain {sorted(required_weights)}")
    if not required_panel.issubset(panel):
        raise KeyError(f"panel must contain {sorted(required_panel)}")

    merged = weights.merge(
        panel[["date", "asset", target_column]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    active = merged["weight"].ne(0)
    unavailable = active & merged[target_column].isna()
    merged["contribution"] = merged["weight"] * np.expm1(merged[target_column])
    merged.loc[~active, "contribution"] = 0.0

    merged["active"] = active
    merged["unavailable"] = unavailable
    merged["abs_weight"] = merged["weight"].abs()
    grouped = merged.groupby("date", sort=True, observed=True)
    result = grouped.agg(
        gross_return=("contribution", "sum"),
        n_names=("active", "sum"),
        max_abs_weight=("abs_weight", "max"),
        missing_active_return=("unavailable", "any"),
    ).reset_index()
    no_active = result["n_names"].eq(0)
    result.loc[result["missing_active_return"] | no_active, "gross_return"] = np.nan
    result.loc[no_active, "max_abs_weight"] = np.nan
    result["n_names"] = result["n_names"].astype(int)
    return result
