"""Reconciled daily ledgers and NAV-consistent cumulative attribution."""

import numpy as np
import pandas as pd

from risk_aware_portfolio.backtest.engine import BacktestResult


def validate_attribution(result: BacktestResult, tolerance: float = 1e-10) -> None:
    """Assert forecast-inclusive, factor/specific, and active identities."""
    a, daily = result.attribution, result.returns
    checks = (
        (a["factor"] + a["specific_total"] + a["transaction_cost"], daily["net_return"]),
        (
            a["factor"] + a["alpha_forecast"] + a["specific_after_alpha"] + a["transaction_cost"],
            daily["net_return"],
        ),
        (
            a["active_factor"] + a["active_specific"] + a["transaction_cost"],
            daily["net_return"] - daily["benchmark_return"],
        ),
    )
    for reconstructed, observed in checks:
        if not np.allclose(reconstructed, observed, atol=tolerance, rtol=0):
            raise ValueError("Attribution failed to reconcile to portfolio accounting")


def cumulative_attribution(result: BacktestResult) -> pd.DataFrame:
    """Multiply each return contribution by prior NAV, then accumulate P&L.

    Independent compounding of each component is invalid: components add to
    return_t, and prior portfolio NAV scales their contribution to cumulative P&L.
    """
    validate_attribution(result)
    prior_nav = result.returns["nav"].shift(1, fill_value=1.0)
    columns = ["factor", "alpha_forecast", "specific_after_alpha", "transaction_cost"]
    pnl = result.attribution[columns].mul(prior_nav, axis=0).cumsum()
    pnl["total"] = pnl.sum(axis=1)
    np.testing.assert_allclose(pnl["total"], result.returns["nav"] - 1, atol=1e-10)
    return pnl


def truth_attribution(result: BacktestResult, truth: pd.DataFrame) -> pd.DataFrame:
    """Optional synthetic causal decomposition; never used to construct weights."""
    truth = truth.set_index(["date", "asset"])
    rows = []
    for _, daily in result.returns.iterrows():
        date = daily["return_date"]
        w = result.weights.loc[daily["decision_date"]]
        q = 1 - daily["transaction_cost"]
        components = truth.loc[date].reindex(w.index)
        rows.append(
            {
                "date": date,
                "true_factor": q * float(w @ components["true_factor_return"]),
                "true_alpha": q * float(w @ components["true_alpha_return"]),
                "true_specific": q * float(w @ components["true_specific_return"]),
                "transaction_cost": -daily["transaction_cost"],
            }
        )
    frame = pd.DataFrame(rows).set_index("date")
    if not np.allclose(frame.sum(axis=1), result.returns["net_return"], atol=1e-10, rtol=0):
        raise ValueError("Synthetic truth does not match current panel; refusing false attribution")
    return frame
