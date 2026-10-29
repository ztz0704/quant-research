"""Dated asymmetric costs and convex/numeric reconciliation."""

import cvxpy as cp
import numpy as np
import pytest

from risk_aware_portfolio.costs.transaction_cost import TransactionCost


def test_sell_tax_is_asymmetric_and_changes_on_effective_date():
    schedule = (("2022-01-01", 10.0), ("2023-08-28", 5.0))
    model = TransactionCost(linear_bps=0, impact_coefficient=0, sell_bps_schedule=schedule)
    assert model.at_date("2023-08-27").realized(np.array([0.1])) == 0
    assert model.at_date("2023-08-27").realized(np.array([-0.1])) == pytest.approx(0.0001)
    assert model.at_date("2023-08-28").realized(np.array([-0.1])) == pytest.approx(0.00005)
    with pytest.raises(ValueError, match="No sell-tax"):
        model.at_date("2021-12-31")


@pytest.mark.parametrize("impact", [False, True])
def test_convex_sell_cost_matches_predicted(impact):
    model = TransactionCost(sell_bps=10, optimization_impact=impact)
    trades = np.array([0.01, -0.05, 0])
    expression = model.expression(cp.Constant(trades))
    assert expression.is_convex()
    assert expression.value == pytest.approx(model.predicted(trades))
