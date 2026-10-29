"""Simple ex-ante market-regime definitions and IC summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_market_regimes(
    panel: pd.DataFrame,
    *,
    volatility_window: int = 20,
    trend_window: int = 120,
    minimum_history: int = 60,
) -> pd.DataFrame:
    """Build volatility, trend, and dispersion states using signal-date information only."""

    data = panel.sort_values(["asset", "date"], kind="mergesort").copy()
    data["asset_log_return"] = np.log(data["close"]).groupby(
        data["asset"], sort=False, observed=True
    ).diff()
    eligible = data.get("eligible", pd.Series(True, index=data.index)).astype(bool)
    same_day = data[eligible].groupby("date", sort=True, observed=True)["asset_log_return"]
    market_return = same_day.mean().rename("market_return")
    dispersion = same_day.std(ddof=1).rename("dispersion")
    regime = pd.concat([market_return, dispersion], axis=1).sort_index()

    regime["market_volatility"] = regime["market_return"].rolling(
        volatility_window, min_periods=volatility_window
    ).std(ddof=1)
    volatility_threshold = regime["market_volatility"].expanding(
        min_periods=minimum_history
    ).median().shift(1)
    dispersion_threshold = regime["dispersion"].expanding(
        min_periods=minimum_history
    ).median().shift(1)

    regime["benchmark_level"] = np.exp(regime["market_return"].fillna(0).cumsum())
    regime["benchmark_ma"] = regime["benchmark_level"].rolling(
        trend_window, min_periods=trend_window
    ).mean()
    regime["volatility_regime"] = np.where(
        regime["market_volatility"] > volatility_threshold, "high", "low"
    )
    regime["trend_regime"] = np.where(
        regime["benchmark_level"] >= regime["benchmark_ma"], "up", "down"
    )
    regime["dispersion_regime"] = np.where(
        regime["dispersion"] > dispersion_threshold, "high", "low"
    )
    unavailable = volatility_threshold.isna() | dispersion_threshold.isna() | regime[
        "benchmark_ma"
    ].isna()
    regime.loc[unavailable, ["volatility_regime", "trend_regime", "dispersion_regime"]] = (
        "unavailable"
    )
    return regime.reset_index().rename(columns={"index": "date"})


def regime_ic_summary(daily_ic: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Summarize each signal's IC within each pre-defined regime dimension."""

    merged = daily_ic.merge(
        regimes[["date", "volatility_regime", "trend_regime", "dispersion_regime"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    rows: list[pd.DataFrame] = []
    for dimension in ("volatility_regime", "trend_regime", "dispersion_regime"):
        summary = (
            merged[merged[dimension].ne("unavailable")]
            .groupby(["signal", dimension], observed=True)["ic"]
            .agg(mean_ic="mean", median_ic="median", n_dates="count")
            .reset_index()
            .rename(columns={dimension: "state"})
        )
        summary["regime_dimension"] = dimension.removesuffix("_regime")
        rows.append(summary)
    if not rows:
        return pd.DataFrame(
            columns=["signal", "state", "mean_ic", "median_ic", "n_dates", "regime_dimension"]
        )
    return pd.concat(rows, ignore_index=True)[
        ["signal", "regime_dimension", "state", "mean_ic", "median_ic", "n_dates"]
    ]

