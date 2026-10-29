import cvxpy as cp
import numpy as np
import pytest

from risk_aware_portfolio.costs.transaction_cost import TransactionCost


def test_zero_and_monotonic():
    cost = TransactionCost()
    assert cost.realized(np.zeros(4)) == 0
    small = np.array([0.01, -0.01])
    assert cost.realized(2 * small) > cost.realized(small) > 0


def test_convex_fractional_cost_matches_accounting():
    cost = TransactionCost()
    trade = np.array([0.03, -0.04, 0.0])
    expression = cost.expression(cp.Constant(trade))
    assert expression.is_convex()
    assert float(expression.value) == pytest.approx(cost.realized(trade))


def test_linear_optimization_distinct_from_nonlinear_realized():
    cost = TransactionCost(optimization_impact=False)
    trade = np.array([0.1])
    assert cost.predicted(trade) < cost.realized(trade)
    with pytest.raises(ValueError):
        TransactionCost(linear_bps=-1)
