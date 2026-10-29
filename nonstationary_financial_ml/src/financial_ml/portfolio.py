"""Explicitly diagnostic quantile portfolios, cost sensitivity and return coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weights(section: pd.DataFrame, prediction: str, kind: str, quantiles: int) -> pd.Series:
    """Select current scores without reading future labels or target-valid flags."""
    if kind not in ("long_only", "long_short") or quantiles < 2:
        raise ValueError("Invalid portfolio specification")
    values = pd.Series(0., index=section.asset.astype(str))
    if section[prediction].std(ddof=0) < 1e-12:
        if kind == "long_only":
            values[:] = 1 / len(values)
        return values
    count = max(1, len(section) // quantiles)
    ordered = section.sort_values([prediction, "asset"])
    values.loc[ordered.asset.iloc[-count:]] = 1 / count
    if kind == "long_short":
        values.loc[ordered.asset.iloc[:count]] = -1 / count
    return values


def half_l1(current: pd.Series, previous: pd.Series) -> float:
    """Half L1 weight turnover, including opening from cash."""
    universe = current.index.union(previous.index)
    return float((current.reindex(universe, fill_value=0) - previous.reindex(universe, fill_value=0)).abs().sum() / 2)


def portfolio_daily(predictions: pd.DataFrame, column: str, kind: str, cfg: dict) -> pd.DataFrame:
    """Compute non-overlapping horizon returns; label validity cannot affect weights."""
    horizon = cfg["labels"]["forward_horizon"]
    rows = []
    previous = pd.Series(dtype=float)
    reset = False
    for index, (date, section) in enumerate(predictions.groupby("date", sort=True)):
        if index % horizon:
            continue
        current = weights(section, column, kind, cfg["portfolio"]["quantiles"])
        turnover = half_l1(current, previous)
        targets = section.set_index("asset").target.reindex(current.index)
        active_missing = targets.loc[current.abs().gt(1e-12)].isna().any()
        gross = np.nan if active_missing else float((current * targets.fillna(0)).sum())
        benchmark = float(section.target.mean()) if section.target.notna().all() else np.nan
        for cost_bps in cfg["portfolio"]["cost_bps"]:
            cost = turnover * cost_bps / 10000
            rows.append({"date": date, "model": section.model.iloc[0], "prediction": column, "portfolio": kind,
                         "cost_bps_per_half_l1": cost_bps, "gross_return": gross, "turnover": turnover,
                         "cost": cost, "net_return": gross - cost, "benchmark_gross_return": benchmark,
                         "active_excess_return": gross - cost - benchmark, "active_return_missing": bool(active_missing),
                         "weight_reset_after_missing": reset, "assets": len(section), "active_assets": int(current.ne(0).sum())})
        if active_missing or not np.isfinite(gross) or 1 + gross <= 0:
            previous, reset = pd.Series(dtype=float), True
        else:
            previous, reset = current * (1 + targets.fillna(0)) / (1 + gross), False
    return pd.DataFrame(rows)


def performance(frame: pd.DataFrame, horizon: int = 1) -> dict:
    """Report coverage first; incomplete return series cannot receive a full Sharpe."""
    values = frame.net_return
    result = {"periods": len(values), "missing_periods": int(values.isna().sum()),
              "return_coverage": float(values.notna().mean()), "mean_turnover": frame.turnover.mean(),
              "total_model_cost": frame.cost.sum(), "annualized_return": np.nan, "annualized_volatility": np.nan,
              "sharpe": np.nan, "max_drawdown": np.nan, "annualized_active_mean": np.nan}
    if values.isna().any() or len(values) < 10:
        return result
    annual = 252 / horizon
    nav = (1 + values).cumprod()
    peak = np.maximum.accumulate(np.r_[1., nav.to_numpy()])[1:]
    result.update(annualized_return=float(nav.iloc[-1] ** (annual / len(values)) - 1),
                  annualized_volatility=values.std(ddof=1) * np.sqrt(annual),
                  sharpe=values.mean() / values.std(ddof=1) * np.sqrt(annual) if values.std(ddof=1) > 0 else np.nan,
                  max_drawdown=float((nav.to_numpy() / peak - 1).min()),
                  annualized_active_mean=frame.active_excess_return.mean() * annual if frame.active_excess_return.notna().all() else np.nan)
    return result
