"""Rolling-year out-of-sample signal selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def walk_forward_ic_selection(
    daily_ic: pd.DataFrame,
    *,
    train_years: int = 5,
    top_k: int = 5,
    min_train_dates: int = 100,
) -> dict[str, pd.DataFrame]:
    """Select signals only on trailing train years and evaluate the next year."""

    if train_years < 1 or top_k < 1:
        raise ValueError("train_years and top_k must be positive.")
    data = daily_ic.dropna(subset=["ic"]).copy()
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    years = sorted(data["year"].unique())

    selection_rows: list[dict[str, Any]] = []
    combination_rows: list[dict[str, Any]] = []
    for test_position in range(train_years, len(years)):
        train_window = years[test_position - train_years : test_position]
        test_year = years[test_position]
        train = data[data["year"].isin(train_window)]
        test = data[data["year"] == test_year]
        train_stats = (
            train.groupby("signal", observed=True)["ic"]
            .agg(train_mean_ic="mean", train_n_dates="count")
            .query("train_n_dates >= @min_train_dates")
            .sort_values(["train_mean_ic"], ascending=False, kind="mergesort")
        )
        selected = train_stats.head(top_k).index.tolist()
        if not selected:
            continue
        for signal in selected:
            test_values = test.loc[test["signal"] == signal, "ic"].dropna()
            selection_rows.append(
                {
                    "test_year": int(test_year),
                    "train_start_year": int(train_window[0]),
                    "train_end_year": int(train_window[-1]),
                    "signal": signal,
                    "train_mean_ic": float(train_stats.loc[signal, "train_mean_ic"]),
                    "test_mean_ic": float(test_values.mean()) if len(test_values) else np.nan,
                    "test_n_dates": len(test_values),
                }
            )
        selected_signal_ic_average = (
            test[test["signal"].isin(selected)]
            .pivot(index="date", columns="signal", values="ic")
            .mean(axis=1)
            .dropna()
        )
        combination_rows.append(
            {
                "test_year": int(test_year),
                "train_start_year": int(train_window[0]),
                "train_end_year": int(train_window[-1]),
                "selected_signals": ",".join(selected),
                "mean_selected_signal_test_ic": (
                    float(selected_signal_ic_average.mean())
                    if len(selected_signal_ic_average)
                    else np.nan
                ),
                "selected_signal_test_n_dates": len(selected_signal_ic_average),
            }
        )
    return {
        "selections": pd.DataFrame(
            selection_rows,
            columns=[
                "test_year",
                "train_start_year",
                "train_end_year",
                "signal",
                "train_mean_ic",
                "test_mean_ic",
                "test_n_dates",
            ],
        ),
        "combinations": pd.DataFrame(
            combination_rows,
            columns=[
                "test_year",
                "train_start_year",
                "train_end_year",
                "selected_signals",
                "mean_selected_signal_test_ic",
                "selected_signal_test_n_dates",
            ],
        ),
    }


def attach_walk_forward_score(
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    signal_prefix: str = "z__",
    output_column: str = "walk_forward_score",
) -> pd.DataFrame:
    """Attach an equal-weight score using only signals selected before each test year."""

    required_panel = {"date", "asset"}
    required_selection = {"test_year", "signal"}
    if not required_panel.issubset(panel):
        raise KeyError(f"panel must contain {sorted(required_panel)}")
    if not required_selection.issubset(selections):
        raise KeyError(f"selections must contain {sorted(required_selection)}")
    result = panel.copy()
    result[output_column] = np.nan
    result["walk_forward_signal_count"] = 0
    years = pd.to_datetime(result["date"]).dt.year
    for test_year, group in selections.groupby("test_year", sort=True, observed=True):
        signal_names = sorted(group["signal"].astype(str).unique())
        columns = [f"{signal_prefix}{signal}" for signal in signal_names]
        missing = sorted(set(columns) - set(result.columns))
        if missing:
            raise KeyError(f"Missing selected signal columns: {missing}")
        mask = years.eq(int(test_year))
        values = result.loc[mask, columns]
        result.loc[mask, output_column] = values.mean(axis=1, skipna=False)
        result.loc[mask, "walk_forward_signal_count"] = len(columns)
    return result
