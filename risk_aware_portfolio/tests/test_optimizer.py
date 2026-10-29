import numpy as np
import pandas as pd
import pytest

from risk_aware_portfolio.portfolio.baseline import equal_weight_benchmark, rank_portfolio
from risk_aware_portfolio.portfolio.constraints import constraint_exposures
from risk_aware_portfolio.portfolio.optimizer import OptimizationError, optimize_portfolio
from risk_aware_portfolio.preprocessing.neutralization import neutralize, preprocess_alpha
from risk_aware_portfolio.risk.exposures import build_exposures
from risk_aware_portfolio.risk.model import estimate_risk_history
from risk_aware_portfolio.utils.config import variant


@pytest.mark.parametrize("long_only", [True, False])
def test_all_optimizer_constraints(synthetic, config, long_only):
    config = variant(
        config,
        portfolio={
            "long_only": long_only,
            "gross_limit": 1.0,
            "industry_tolerance": 0.001,
            "style_tolerance": 0.001,
            "turnover_limit": 0.20,
            "tracking_error_annual": 0.08,
        },
    )
    snapshot = next(iter(estimate_risk_history(synthetic.panel, config).snapshots.values()))
    section = synthetic.panel[synthetic.panel["date"].eq(snapshot.date)]
    alpha = preprocess_alpha(section, config)["selected_alpha"]
    benchmark = equal_weight_benchmark(alpha.index, long_only)
    previous = benchmark.copy()
    result = optimize_portfolio(alpha, snapshot, previous, benchmark, config)
    w = result.weights.to_numpy()
    assert w.sum() == pytest.approx(1 if long_only else 0, abs=2e-6)
    assert w.max() <= config["portfolio"]["max_weight"] + 2e-6
    assert w.min() >= (0 if long_only else -config["portfolio"]["max_weight"]) - 2e-6
    assert abs(w).sum() <= 1.0 + 2e-6
    assert abs(w - previous.to_numpy()).sum() <= 0.20 + 2e-6
    industry, style = constraint_exposures(snapshot, config)
    assert np.abs(industry.T @ (w - benchmark.to_numpy())).max() <= 0.001 + 2e-6
    assert np.abs(style.T @ (w - benchmark.to_numpy())).max() <= 0.001 + 2e-6
    active = w - benchmark.to_numpy()
    assert np.sqrt(active @ snapshot.covariance.to_numpy() @ active) * np.sqrt(252) <= 0.08 + 1e-5


def test_infeasible_is_explicit(synthetic, config):
    config = variant(config, portfolio={"max_weight": 0.001})
    snapshot = next(iter(estimate_risk_history(synthetic.panel, config).snapshots.values()))
    alpha = pd.Series(np.zeros(40), index=snapshot.exposures.index)
    benchmark = equal_weight_benchmark(alpha.index)
    with pytest.raises(OptimizationError, match="infeasible"):
        optimize_portfolio(alpha, snapshot, benchmark, benchmark, config)


def test_rank_portfolio():
    alpha = pd.Series(np.arange(10.0))
    long_only = rank_portfolio(alpha, 0.2)
    long_short = rank_portfolio(alpha, 0.2, False)
    assert long_only.sum() == pytest.approx(1)
    assert long_only.iloc[-2:].sum() == 1
    assert long_short.sum() == pytest.approx(0)
    assert long_short.abs().sum() == pytest.approx(1)


@pytest.mark.parametrize("mode", ["industry", "size", "style", "joint"])
def test_alpha_neutralization(synthetic, config, mode):
    section = synthetic.panel.iloc[:40]
    x = build_exposures(section, config)
    residual = neutralize(section["alpha"].to_numpy(), x, mode)
    assert abs(residual.mean()) < 1e-12
    if mode == "joint":
        np.testing.assert_allclose(x.to_numpy().T @ residual, 0, atol=1e-10)
