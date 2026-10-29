import numpy as np
import pandas as pd
import pytest

from risk_aware_portfolio.attribution.factor_attribution import (
    cumulative_attribution,
    truth_attribution,
    validate_attribution,
)
from risk_aware_portfolio.attribution.performance import drawdown, hac_mean_test
from risk_aware_portfolio.backtest.engine import run_backtest
from risk_aware_portfolio.diagnostics.portfolio_diagnostics import risk_contributions
from risk_aware_portfolio.diagnostics.risk_diagnostics import risk_calibration
from risk_aware_portfolio.diagnostics.robustness import experiment_configs, perturb_covariance
from risk_aware_portfolio.risk.model import estimate_risk_history


def test_attribution_and_risk_reconcile(synthetic, config):
    history = estimate_risk_history(synthetic.panel, config)
    result = run_backtest(synthetic.panel, config, history=history)
    validate_attribution(result)
    cumulative = cumulative_attribution(result)
    assert cumulative["total"].iloc[-1] == pytest.approx(result.returns["nav"].iloc[-1] - 1)
    truth = truth_attribution(result, synthetic.truth)
    np.testing.assert_allclose(truth.sum(axis=1), result.returns["net_return"], atol=1e-12)
    factor, asset = risk_contributions(result, history)
    np.testing.assert_allclose(
        factor.groupby("date")["variance_contribution"].sum(),
        result.risk_contributions["factor_variance"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        asset.groupby("date")["total_variance_contribution"].sum(),
        result.risk_contributions["total_variance"],
        atol=1e-12,
    )


def test_forward_risk_is_diagnostic_only(synthetic, config):
    result = run_backtest(synthetic.panel, config, baseline=True)
    frame, buckets = risk_calibration(result, synthetic.panel, 5)
    assert (frame["forward_realized_volatility"] > 0).all()
    assert buckets["n_observations"].sum() == len(frame)


def test_stress_covariance_remains_psd(synthetic, config):
    history = estimate_risk_history(synthetic.panel, config)
    changed = perturb_covariance(history, 0.25)
    for snapshot in changed.snapshots.values():
        assert np.linalg.eigvalsh(snapshot.covariance).min() > 0
    assert len(experiment_configs(config)) == 16


def test_drawdown_includes_initial_nav():
    dd = drawdown(pd.Series([-0.1, 0.0]))
    np.testing.assert_allclose(dd, [-0.1, -0.1])
    test = hac_mean_test(np.array([0.1, -0.1, 0.05, -0.05]))
    assert test["hac_standard_error"] > 0
