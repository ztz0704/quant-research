import numpy as np
import pandas as pd

from risk_aware_portfolio.backtest.engine import run_backtest
from risk_aware_portfolio.risk.model import estimate_risk_history


def test_future_data_cannot_change_risk(synthetic, config):
    panel = synthetic.panel.copy()
    cutoff = panel["date"].unique()[30]
    reference = estimate_risk_history(panel, config)
    mask = panel["date"] > cutoff
    panel.loc[mask, "return"] *= -3
    panel.loc[mask, "alpha"] += 100
    panel.loc[mask, "value"] *= -10
    changed = estimate_risk_history(panel, config)
    for date in reference.snapshots:
        if date <= cutoff:
            np.testing.assert_array_equal(
                reference.snapshots[date].covariance, changed.snapshots[date].covariance
            )
    pd.testing.assert_frame_equal(
        reference.factor_returns.loc[:cutoff], changed.factor_returns.loc[:cutoff]
    )


def test_prefix_invariance(synthetic, config):
    panel = synthetic.panel
    cutoff = panel["date"].unique()[30]
    full = estimate_risk_history(panel, config)
    prefix = estimate_risk_history(panel[panel["date"] <= cutoff], config)
    np.testing.assert_array_equal(
        full.snapshots[pd.Timestamp(cutoff)].covariance,
        prefix.snapshots[pd.Timestamp(cutoff)].covariance,
    )


def test_future_returns_and_signals_cannot_change_weights(synthetic, config):
    panel = synthetic.panel.copy()
    cutoff = pd.Timestamp(panel["date"].unique()[30])
    reference = run_backtest(panel, config)
    mask = panel["date"] > cutoff
    panel.loc[mask, "return"] *= -3
    panel.loc[mask, "alpha"] += 100
    panel.loc[mask, "value"] *= -10
    changed = run_backtest(panel, config)
    # Includes weight at cutoff, which must not react to return_(cutoff+1).
    np.testing.assert_array_equal(reference.weights.loc[:cutoff], changed.weights.loc[:cutoff])


def test_same_date_return_is_not_earned_by_new_weight(synthetic, config):
    result = run_backtest(synthetic.panel, config, baseline=True)
    for _, row in result.returns.iterrows():
        w = result.weights.loc[row["decision_date"]]
        realized = synthetic.panel[synthetic.panel["date"].eq(row["return_date"])]
        np.testing.assert_allclose(row["gross_return"], w @ realized.set_index("asset")["return"])
