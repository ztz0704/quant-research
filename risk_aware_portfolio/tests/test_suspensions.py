"""Suspension marking and deferred execution must be explicit and causal."""

import numpy as np
import pandas as pd
import pytest

from risk_aware_portfolio.backtest.engine import run_backtest
from risk_aware_portfolio.data.sina import decode_share_history
from risk_aware_portfolio.data.suspensions import insert_suspension_marks
from risk_aware_portfolio.risk.model import estimate_risk_history


def test_unknown_gaps_and_late_notices_fail():
    calendar = pd.bdate_range("2023-01-02", periods=3)
    prices = pd.DataFrame(
        {
            "date": calendar[[0, 2]],
            "close": [10.0, 11.0],
            "volume": [100.0, 120.0],
            "amount": [1000.0, 1320.0],
        }
    )
    declared = pd.DataFrame(
        {"date": calendar[[1]], "asset": ["sh600000"], "announcement_date": calendar[[0]]}
    )
    marked = insert_suspension_marks(prices, calendar, "sh600000", declared)
    assert marked["close"].tolist() == [10, 10, 11]
    assert marked["tradable"].tolist() == [True, False, True]
    assert marked["volume"].tolist() == [100, 0, 120]
    with pytest.raises(ValueError, match="Unclassified"):
        insert_suspension_marks(prices, calendar, "sh600001", declared)
    declared["announcement_date"] = calendar[2]
    with pytest.raises(ValueError, match="announced after"):
        insert_suspension_marks(prices, calendar, "sh600000", declared)


def test_dated_chart_discards_only_overwritten_events():
    text = "流通A股 万股 <graph><set value='0' hoverText='2000-01-01'/><set value='100' hoverText='2021-01-01'/><set value='200' hoverText='2023-01-01'/></graph>"
    # Caption authenticity check is part of the archived chart contract.
    text = text.replace("<graph>", "<graph caption='流通A股'>")
    events = decode_share_history(text, "2022-01-01", "2023-12-31")
    assert events["outstanding_share"].tolist() == [1e6, 2e6]


def test_suspension_defers_entire_rebalance_without_fee_trades(synthetic, config):
    panel = synthetic.panel.copy()
    panel["tradable"] = True
    reference = run_backtest(panel, config)
    scheduled = reference.returns[reference.returns["rebalance"]]["decision_date"].iloc[1]
    asset = panel["asset"].iloc[0]
    mask = panel["date"].eq(scheduled) & panel["asset"].eq(asset)
    panel.loc[mask, "tradable"] = False
    panel.loc[mask, "return"] = 0.0
    result = run_backtest(panel, config)
    row = result.returns[result.returns["decision_date"].eq(scheduled)].iloc[0]
    assert row["deferred_rebalance"]
    assert not row["rebalance"]
    assert row["transaction_cost"] == 0
    np.testing.assert_array_equal(
        result.weights.loc[scheduled], result.pretrade_weights.loc[scheduled]
    )
    next_row = result.returns[result.returns["decision_date"] > scheduled].iloc[0]
    assert next_row["rebalance"]
    history = estimate_risk_history(panel, config)
    assert history.regressions.loc[scheduled, "n_assets"] == config["synthetic"]["n_assets"] - 1
