import numpy as np
import pytest

from risk_aware_portfolio.backtest.accounting import account_period
from risk_aware_portfolio.costs.transaction_cost import TransactionCost


def test_self_financing_initial_cost():
    target, prior = np.array([0.5, 0.5]), np.zeros(2)
    cost = TransactionCost()
    result = account_period(target, prior, np.array([0.1, -0.1]), cost)
    assert result.capital_ratio + result.transaction_cost == pytest.approx(1, abs=1e-12)
    assert result.turnover == pytest.approx(result.capital_ratio)
    assert result.net_return == pytest.approx(-result.transaction_cost)
    assert result.transaction_cost == pytest.approx(cost.realized(result.trades))


def test_exact_cost_return_interaction_and_drift():
    target = np.array([0.5, 0.5])
    result = account_period(target, np.zeros(2), np.array([0.10, 0.0]), TransactionCost())
    assert result.net_return == pytest.approx(result.capital_ratio * 1.05 - 1)
    np.testing.assert_allclose(result.end_weights, np.array([0.55, 0.5]) / 1.05)
    held = account_period(
        result.end_weights, result.end_weights, np.zeros(2), TransactionCost(), False
    )
    assert held.transaction_cost == held.turnover == 0


def test_long_short_collateral_and_insolvency():
    target = np.array([0.5, -0.5])
    result = account_period(target, np.zeros(2), np.array([0.1, -0.1]), TransactionCost())
    assert result.gross_return == pytest.approx(0.1)
    assert result.end_weights.sum() == pytest.approx(0.1 / 1.1)
    with pytest.raises(ValueError, match="insolvency"):
        account_period(np.array([-2.0]), np.zeros(1), np.array([0.9]), TransactionCost())
