"""Economic performance metrics for simple-return series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(
    returns: pd.Series,
    *,
    annualization: int = 252,
) -> dict[str, float | int]:
    """Calculate annualized simple-return metrics and maximum drawdown."""

    raw = pd.Series(returns, dtype=float)
    values = raw.dropna()
    total_periods = len(raw)
    missing_periods = int(raw.isna().sum())
    if values.empty:
        return {
            "n_periods": 0,
            "n_missing_periods": missing_periods,
            "return_coverage": 0.0,
            "annualized_return": np.nan,
            "cagr": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "win_rate": np.nan,
            "cumulative_return": np.nan,
        }
    annualized_return = values.mean() * annualization
    annualized_volatility = values.std(ddof=1) * np.sqrt(annualization)
    sharpe = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )
    if (values <= -1.0).any():
        raise ValueError("Simple returns must be greater than -100%.")
    wealth = (1.0 + values).cumprod()
    # Include the initial unit of capital so an immediate loss is measured from 1.0
    # instead of incorrectly treating the first underwater observation as a new peak.
    wealth_with_origin = pd.Series(
        np.concatenate(([1.0], wealth.to_numpy(dtype=float))), dtype=float
    )
    drawdown = wealth_with_origin / wealth_with_origin.cummax() - 1.0
    years = len(values) / annualization
    final_wealth = float(wealth.iloc[-1])
    cagr = final_wealth ** (1.0 / years) - 1.0 if years > 0 and final_wealth > 0 else np.nan
    downside = np.minimum(values.to_numpy(dtype=float), 0.0)
    downside_volatility = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(annualization))
    sortino = (
        annualized_return / downside_volatility
        if downside_volatility > 0
        else np.nan
    )
    maximum_drawdown = float(drawdown.min())
    calmar = cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else np.nan
    return {
        "n_periods": len(values),
        "n_missing_periods": missing_periods,
        "return_coverage": float(len(values) / total_periods) if total_periods else np.nan,
        "annualized_return": float(annualized_return),
        "cagr": float(cagr),
        "annualized_volatility": float(annualized_volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": maximum_drawdown,
        "calmar": float(calmar),
        "win_rate": float(values.gt(0).mean()),
        "cumulative_return": float(final_wealth - 1.0),
    }
