import numpy as np
import pandas as pd
import pytest

from risk_aware_portfolio.backtest.open_events import at_time, run_open_events
from risk_aware_portfolio.costs.transaction_cost import TransactionCost


def scenario():
    dates = pd.bdate_range("2020-01-02", periods=7)
    calendar = pd.DataFrame({"date": dates, "exchange": "SSE"})
    rows = []
    for date in dates:
        for asset in ("A", "B"):
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "exchange": "SSE",
                    "raw_open": 10.0,
                    "raw_close": 10.0,
                    "adj_factor": 1.0,
                    "universe_member": True,
                    "buyable": True,
                    "sellable": True,
                    "execution_asof": at_time(date, 9, 30),
                    "action_indicator_asof": at_time(date, 9, 30),
                    "open_price_asof": at_time(date, 9, 30),
                    "close_price_asof": at_time(date, 18),
                    "buy_fill_fraction": 1.0,
                    "sell_fill_fraction": 1.0,
                    "action_complete": False,
                    "delisted": False,
                }
            )
    return pd.DataFrame(rows), calendar, dates


def engine(panel, calendar, decisions, weights, bps=0):
    def build(date, rows, state):
        return pd.Series(weights.get(date, {"A": 1.0}), dtype=float), {"status": "known_target"}

    return run_open_events(
        panel,
        calendar,
        decisions,
        build,
        TransactionCost(linear_bps=bps, impact_coefficient=0),
        initial_cash=1000.0,
    )


def test_normal_trade_and_cash_fee_nav_hand_reconciliation():
    panel, calendar, dates = scenario()
    result = engine(panel, calendar, dates[:2], {dates[0]: {"A": 1.0}, dates[1]: {"B": 1.0}}, 100)
    entry = result.events.set_index("date").loc[dates[1]]
    assert entry.post_nav == pytest.approx(1000 / 1.01)
    assert entry.fees == pytest.approx(1000 - 1000 / 1.01)
    assert entry.cash == pytest.approx(0.0, abs=1e-8)
    assert result.final_state.quantities["A"] == pytest.approx(0.0)
    assert result.final_state.quantities["B"] == pytest.approx(0.0)
    assert result.events.iloc[-1].post_nav == pytest.approx(result.final_state.cash)
    for _, event in result.events.iterrows():
        assert event.post_nav == pytest.approx(event.pre_nav - event.fees, abs=1e-7)


def test_halt_pending_exit_inventory_and_price_recovery_not_zero_filled():
    panel, calendar, dates = scenario()
    halted = panel.asset.eq("A") & panel.date.isin(dates[2:4])
    panel.loc[halted, ["raw_open", "raw_close"]] = np.nan
    panel.loc[halted, ["buyable", "sellable"]] = False
    panel.loc[panel.asset.eq("A") & panel.date.ge(dates[4]), ["raw_open", "raw_close"]] = 12.0
    result = engine(
        panel, calendar, dates[:4], {dates[0]: {"A": 1.0}, **{d: {"B": 1.0} for d in dates[1:4]}}
    )
    holding = result.holdings.loc[result.holdings.asset.eq("A")].set_index("date")
    assert holding.loc[dates[2], "quantity"] == 100
    assert holding.loc[dates[3], "quantity"] == 100
    assert result.events.set_index("date").loc[dates[2], "nav_status"] == "BLOCKED"
    recovered = result.events.set_index("date").loc[dates[4]]
    assert recovered.pre_nav == pytest.approx(1200.0)
    assert recovered.post_nav == pytest.approx(1200.0)
    assert result.periods.net_return.isna().any()
    assert result.orders.event.str.contains("sell_not_executable|missing_price").any()


def test_partial_fill_and_gap_cash_limit_preserve_residual_order():
    panel, calendar, dates = scenario()
    panel.loc[panel.date.eq(dates[1]) & panel.asset.eq("A"), "buy_fill_fraction"] = 0.5
    result = engine(panel, calendar, dates[:1], {}, 0)
    event = result.events.set_index("date").loc[dates[1]]
    assert event.cash == 500
    assert event.post_nav == 1000
    assert result.holdings.query("asset == 'A'").iloc[0].pending_quantity == 50
    panel.loc[panel.date.eq(dates[1]) & panel.asset.eq("A"), "raw_open"] = 40.0
    panel.loc[panel.date.eq(dates[1]) & panel.asset.eq("A"), "buy_fill_fraction"] = 1.0
    limited = engine(panel, calendar, dates[:1], {})
    entry = limited.holdings.query("asset == 'A'").iloc[0]
    assert entry.quantity == pytest.approx(25.0)
    assert entry.pending_quantity == pytest.approx(75.0)
    assert limited.events.cash.min() >= -1e-8


def test_delisting_without_terminal_value_retains_shares_and_blocks():
    panel, calendar, dates = scenario()
    mask = panel.asset.eq("A") & panel.date.ge(dates[2])
    panel.loc[mask, "delisted"] = True
    panel.loc[mask, ["raw_open", "raw_close"]] = np.nan
    result = engine(panel, calendar, dates[:3], {})
    assert result.final_state.quantities["A"] == 100
    assert result.events.iloc[-1].nav_status == "BLOCKED"
    assert result.orders.event.eq("delisted_unknown_terminal").any()


def test_unknown_action_cannot_use_adjusted_price_as_cash_return():
    panel, calendar, dates = scenario()
    mask = panel.asset.eq("A") & panel.date.ge(dates[2])
    panel.loc[mask, "adj_factor"] = 2.0
    panel.loc[mask, ["raw_open", "raw_close"]] = 5.0
    result = engine(panel, calendar, dates[:2], {})
    assert result.final_state.quantities["A"] == 100
    assert "A" in result.final_state.uncertain_actions
    assert result.events.iloc[-1].nav_status == "BLOCKED"


def test_certified_split_and_dividend_known_answer():
    panel, calendar, dates = scenario()
    mask = panel.asset.eq("A") & panel.date.ge(dates[2])
    panel.loc[mask, "adj_factor"] = 2.0
    panel.loc[mask, ["raw_open", "raw_close"]] = 5.0
    action = panel.asset.eq("A") & panel.date.eq(dates[2])
    panel.loc[action, "action_complete"] = True
    panel.loc[action, "action_id"] = "split_dividend_one"
    panel.loc[action, "action_asof"] = at_time(dates[2], 9)
    panel.loc[action, "split_ratio"] = 2.0
    panel.loc[action, "cash_dividend"] = 1.0
    result = engine(panel, calendar, dates[:1], {})
    assert result.events.iloc[-1].pre_nav == pytest.approx(1100.0)
    assert result.final_state.cash == pytest.approx(1100.0)
    assert result.orders.event.eq("corporate_action_applied").sum() == 1


def test_pool_exit_can_sell_but_cannot_buy_and_unknown_sell_retains():
    panel, calendar, dates = scenario()
    mask = panel.asset.eq("A") & panel.date.ge(dates[2])
    panel.loc[mask, "universe_member"] = False
    panel["sellable"] = panel.sellable.astype("boolean")
    panel.loc[mask, "sellable"] = pd.NA
    result = engine(
        panel,
        calendar,
        dates[:3],
        {dates[0]: {"A": 1.0}, dates[1]: {"B": 1.0}, dates[2]: {"B": 1.0}},
    )
    assert result.final_state.quantities["A"] == 100
    assert result.final_state.orders["A"].remaining == -100
    assert result.events.iloc[-1].post_nav == 1000  # native quote exists, execution is unknown


@pytest.mark.parametrize("cut_position", [1, 2, 3])
def test_future_prices_and_execution_states_do_not_change_historical_targets(cut_position):
    panel, calendar, dates = scenario()
    decision_dates = dates[:4]
    weights = {d: {"A": 0.6, "B": 0.4} for d in decision_dates}
    before = engine(panel, calendar, decision_dates, weights)
    future = panel.date.gt(dates[cut_position])
    changed = panel.copy()
    changed.loc[future, ["raw_open", "raw_close"]] *= 10
    changed.loc[future, "sellable"] = False
    after = engine(changed, calendar, decision_dates, weights)
    mask = before.targets.date.le(dates[cut_position])
    pd.testing.assert_frame_equal(before.targets.loc[mask], after.targets.loc[mask])


def test_late_open_state_rejects_before_trade_and_invalid_schema_fails():
    panel, calendar, dates = scenario()
    panel.loc[panel.date.eq(dates[1]), "execution_asof"] = at_time(dates[1], 18)
    result = engine(panel, calendar, dates[:1], {})
    assert result.events.set_index("date").loc[dates[1], "trade_amount"] == 0
    assert result.orders.event.eq("rejected_late_or_unknown_execution_state").any()
    with pytest.raises(ValueError, match="missing fields"):
        engine(panel.drop(columns="sellable"), calendar, dates[:1], {})


def test_unheld_target_requires_timely_close_price_and_unknown_membership_blocks_buy():
    panel, calendar, dates = scenario()
    panel.loc[panel.date.eq(dates[0]), "close_price_asof"] = at_time(dates[1], 18)
    result = engine(panel, calendar, dates[:1], {})
    assert result.events.trade_amount.sum() == 0
    assert result.decisions.order_plan_status.eq(
        "blocked_unknown_close_valuation_orders_retained"
    ).all()
    panel, calendar, dates = scenario()
    panel["universe_member"] = panel.universe_member.astype("boolean")
    panel.loc[panel.date.eq(dates[1]), "universe_member"] = pd.NA
    result = engine(panel, calendar, dates[:1], {})
    assert result.events.trade_amount.sum() == 0
    assert result.orders.event.eq("rejected_not_eligible_for_new_buy").any()


def test_delisting_cash_settlement_is_explicit_and_unknown_fill_is_not_assumed():
    panel, calendar, dates = scenario()
    action = panel.asset.eq("A") & panel.date.eq(dates[2])
    panel.loc[action, "delisted"] = True
    panel.loc[action, "delisting_complete"] = True
    panel.loc[action, "delisting_cash_per_share"] = 3.0
    panel.loc[action, "action_asof"] = at_time(dates[2], 9)
    result = engine(panel, calendar, dates[:1], {})
    assert result.final_state.cash == 300.0
    assert result.final_state.quantities["A"] == 0.0
    panel, calendar, dates = scenario()
    result = engine(
        panel.drop(columns=["buy_fill_fraction", "sell_fill_fraction"]), calendar, dates[:1], {}
    )
    assert result.events.trade_amount.sum() == 0
    assert result.orders.event.eq("rejected_no_fill_evidence_or_zero_fill").any()


def test_certified_action_identifier_prevents_duplicate_dividend_or_split():
    panel, calendar, dates = scenario()
    mask = panel.asset.eq("A") & panel.date.ge(dates[2])
    panel.loc[mask, "adj_factor"] = 2.0
    panel.loc[mask, ["raw_open", "raw_close"]] = 5.0
    panel.loc[mask, "action_complete"] = True
    panel.loc[mask, "action_id"] = "same_primary_event"
    panel.loc[mask, "split_ratio"] = 2.0
    panel.loc[mask, "cash_dividend"] = 1.0
    panel.loc[mask, "action_asof"] = at_time(dates[2], 9)
    result = engine(panel, calendar, dates[:2], {})
    assert result.final_state.cash == pytest.approx(1100.0)
    assert result.orders.event.eq("corporate_action_applied").sum() == 1


def test_restoring_unknown_action_requires_complete_bridge_from_last_known_factor():
    panel, calendar, dates = scenario()
    mask = panel.asset.eq("A") & panel.date.ge(dates[2])
    panel.loc[mask, "adj_factor"] = 2.0
    panel.loc[mask, ["raw_open", "raw_close"]] = 5.0
    action = panel.asset.eq("A") & panel.date.eq(dates[3])
    panel.loc[action, "action_complete"] = True
    panel.loc[action, "action_id"] = "cumulative_reconciliation"
    panel.loc[action, "split_ratio"] = 2.0
    panel.loc[action, "cash_dividend"] = 1.0
    panel.loc[action, "action_asof"] = at_time(dates[3], 9)
    panel.loc[action, "reconciliation_from_factor"] = 0.5
    blocked = engine(panel, calendar, dates[:2], {})
    assert blocked.events.iloc[-1].nav_status == "BLOCKED"
    assert blocked.final_state.quantities["A"] == 100.0
    panel.loc[action, "reconciliation_from_factor"] = 1.0
    restored = engine(panel, calendar, dates[:2], {})
    assert restored.final_state.cash == pytest.approx(1100.0)
    assert restored.events.iloc[-1].nav_status == "valid"


def test_late_action_indicator_blocks_without_reading_future_factor_and_schema_boundaries():
    panel, calendar, dates = scenario()
    mask = panel.asset.eq("A") & panel.date.eq(dates[2])
    panel.loc[mask, "action_indicator_asof"] = at_time(dates[2], 18)
    before = engine(panel, calendar, dates[:2], {})
    changed = panel.copy()
    changed.loc[mask, "adj_factor"] = 100.0
    after = engine(changed, calendar, dates[:2], {})
    assert before.events.iloc[-1].nav_status == "BLOCKED"
    assert before.orders.event.eq("action_indicator_unavailable_at_open").any()
    pd.testing.assert_frame_equal(before.targets, after.targets)
    assert before.final_state.cash == after.final_state.cash
    with pytest.raises(ValueError, match="unique independent"):
        engine(panel, calendar, pd.DatetimeIndex([]), {})
    with pytest.raises(ValueError, match="local datetime64"):
        engine(panel.assign(open_price_asof="not_parsed"), calendar, dates[:2], {})


def test_decision_view_hides_joined_future_outcomes_and_unavailable_close_quote():
    panel, calendar, dates = scenario()
    panel["target"], panel["target_valid"], panel["exit_sellable"] = 0.9, True, True
    panel["label_available_at"] = at_time(dates[-1], 18)
    mask = panel.date.eq(dates[0]) & panel.asset.eq("B")
    panel.loc[mask, "close_price_asof"] = at_time(dates[1], 18)

    def build(date, rows, state):
        assert not {"target", "target_valid", "exit_sellable", "label_available_at"}.intersection(
            rows
        )
        if date == dates[0]:
            assert pd.isna(rows.loc["B", "raw_close"])
        return pd.Series({"A": 1.0}), {"status": "known_target"}

    result = run_open_events(
        panel,
        calendar,
        dates[:2],
        build,
        TransactionCost(linear_bps=0, impact_coefficient=0),
        initial_cash=1000.0,
    )
    assert result.final_state.cash == pytest.approx(1000.0)
