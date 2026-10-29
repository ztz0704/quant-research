import numpy as np
import pandas as pd
import pytest

from risk_aware_portfolio.data.synthetic import generate_synthetic
from risk_aware_portfolio.risk.exposures import build_exposures
from risk_aware_portfolio.risk.factor_returns import estimate_factor_returns
from risk_aware_portfolio.risk.model import estimate_risk_history
from risk_aware_portfolio.utils.config import variant


@pytest.mark.parametrize("method", ["ols", "wls"])
def test_exact_factor_recovery(synthetic, config, method):
    section = synthetic.panel[synthetic.panel["date"].eq(synthetic.panel["date"].iloc[0])]
    x = build_exposures(section, config)
    truth = np.linspace(-0.01, 0.01, x.shape[1])
    returns = pd.Series(x.to_numpy() @ truth, index=x.index)
    result = estimate_factor_returns(x, returns, section.set_index("asset")["market_cap"], method)
    np.testing.assert_allclose(result.factor_returns, truth, atol=1e-12)
    np.testing.assert_allclose(result.specific_returns, 0, atol=1e-12)
    assert result.diagnostics["rank"] == x.shape[1]


def test_stochastic_factor_recovery(synthetic, config):
    # Independent identification experiment, not a tuned research backtest.
    validation = variant(
        config,
        synthetic={
            "n_assets": 240,
            "n_dates": 80,
            "alpha_strength": 0.0,
            "specific_volatility": 0.002,
        },
    )
    data = generate_synthetic(validation)
    estimates, previous_x, previous_cap = [], None, None
    for _, section in data.panel.groupby("date", sort=True):
        indexed = section.set_index("asset")
        if previous_x is not None:
            estimates.append(
                estimate_factor_returns(previous_x, indexed["return"], previous_cap).factor_returns
            )
        previous_x = build_exposures(section, validation)
        previous_cap = indexed["market_cap"]
    estimated = pd.DataFrame(estimates).to_numpy()
    truth = data.factor_returns.iloc[1:].to_numpy()
    # Bound scales with the known idiosyncratic noise and sample size.
    bound = 5 * 0.002 / np.sqrt(240)
    assert np.sqrt(np.mean((estimated - truth) ** 2)) < bound
    assert (
        min(np.corrcoef(estimated[:, j], truth[:, j])[0, 1] for j in range(truth.shape[1])) > 0.95
    )


def test_rank_deficiency_rejected(synthetic, config):
    section = synthetic.panel.iloc[:40].set_index("asset")
    x = pd.DataFrame(np.ones((40, 2)), index=section.index)
    with pytest.raises(ValueError, match="Unidentified"):
        estimate_factor_returns(x, section["return"], section["market_cap"])


def test_regression_uses_lagged_exposures(synthetic, config):
    panel = synthetic.panel
    dates = panel["date"].unique()
    history = estimate_risk_history(panel, config)
    x = build_exposures(panel[panel["date"].eq(dates[0])], config)
    now = panel[panel["date"].eq(dates[1])].set_index("asset")
    cap = panel[panel["date"].eq(dates[0])].set_index("asset")["market_cap"]
    fit = estimate_factor_returns(x, now["return"], cap)
    np.testing.assert_allclose(history.factor_returns.iloc[0], fit.factor_returns)
