"""Mechanical guard tests; fixture prices are NOT an empirical alpha experiment."""

import copy

import pandas as pd
import pytest
from order_planner import plan_orders
from recent_data_probe import audit_bars, normalize_bars, tencent_volume_multiplier


def inputs():
    targets = pd.DataFrame(
        [{"asset": "600000", "target_weight": 0.2, "signal_asof": "2026-09-29T16:00:00+08:00"}]
    )
    quotes = pd.DataFrame(
        [
            {
                "asset": "600000",
                "quote_at": "2026-09-30T09:31:00+08:00",
                "reference_price": 10.0,
                "bid": 9.99,
                "ask": 10.01,
                "upper_limit": 11.0,
                "lower_limit": 9.0,
                "suspended": False,
                "price_adjustment": "none",
            }
        ]
    )
    account = {
        "source": "paper_ledger",
        "asof": "2026-09-30T09:31:00+08:00",
        "equity": 10000.0,
        "cash_available": 10000.0,
        "positions": [],
        "open_orders": [],
        "policy": {
            "fee_reserve_bps": 10,
            "minimum_commission": 5,
            "cash_buffer": 0,
            "max_position_weight": 0.3,
        },
    }
    calendar = pd.DataFrame([{"date": "2026-09-30", "is_open": 1}])
    return targets, quotes, account, calendar


def run(data):
    return plan_orders(*data, "2026-09-30T09:31:00+08:00")


def test_lots_fee_reservation_and_no_live_path():
    result = run(inputs())
    order = result["orders"][0]
    assert order["proposed_quantity"] == 200
    assert order["reserved_cash"] == pytest.approx(2007.0)
    assert result["unreserved_available_cash"] == pytest.approx(7993.0)
    assert result["ready_for_live"] is False
    assert result["risk_review_required"] is True


@pytest.mark.parametrize("field,value", [("ask", 11.0), ("suspended", True)])
def test_blocked_buy_at_limit_or_suspension(field, value):
    data = inputs()
    data[1].loc[0, field] = value
    assert run(data)["orders"][0]["status"] == "blocked"


def test_t1_sellable_not_total_inventory():
    data = inputs()
    data[0].loc[0, "target_weight"] = 0
    data[2]["positions"] = [{"asset": "600000", "quantity": 250, "sellable_quantity": 100}]
    row = run(data)["orders"][0]
    assert row["proposed_quantity"] == 100
    assert row["projected_quantity_if_filled"] == 150


def test_odd_lot_complete_settled_exit():
    data = inputs()
    data[0].loc[0, "target_weight"] = 0
    data[2]["positions"] = [{"asset": "600000", "quantity": 250, "sellable_quantity": 250}]
    assert run(data)["orders"][0]["proposed_quantity"] == 250


def test_cash_reserve_never_uses_unfilled_sell_proceeds():
    data = inputs()
    data[0].loc[len(data[0])] = ["000001", 0.2, "2026-09-29T16:00:00+08:00"]
    data[0].loc[0, "target_weight"] = 0
    quote = data[1].iloc[0].to_dict()
    quote["asset"] = "000001"
    data[1].loc[len(data[1])] = quote
    data[2]["cash_available"] = 0
    data[2]["positions"] = [{"asset": "600000", "quantity": 200, "sellable_quantity": 200}]
    result = run(data)
    rows = {x["asset"]: x for x in result["orders"]}
    assert rows["600000"]["proposed_quantity"] == 200
    assert rows["000001"]["proposed_quantity"] == 0


@pytest.mark.parametrize(
    "case",
    [
        "stale_quote",
        "future_signal",
        "adjusted",
        "closed",
        "open_order",
        "missing_holding",
        "unsupported",
        "unknown_status",
        "stale_account",
        "negative_weight",
        "fractional_inventory",
    ],
)
def test_bad_inputs_fail_closed(case):
    data = inputs()
    if case == "stale_quote":
        data[1].loc[0, "quote_at"] = "2026-09-30T09:30:00+08:00"
    elif case == "future_signal":
        data[0].loc[0, "signal_asof"] = "2026-09-30T09:32:00+08:00"
    elif case == "adjusted":
        data[1].loc[0, "price_adjustment"] = "qfq"
    elif case == "closed":
        data[3].loc[0, "is_open"] = 0
    elif case == "open_order":
        data[2]["open_orders"] = [{"order_id": "pending"}]
    elif case == "missing_holding":
        data[2]["positions"] = [{"asset": "000001", "quantity": 100, "sellable_quantity": 100}]
    elif case == "unsupported":
        data[0].loc[0, "asset"] = "688001"
    elif case == "unknown_status":
        data[1]["suspended"] = data[1]["suspended"].astype(object)
        data[1].loc[0, "suspended"] = None
    elif case == "stale_account":
        data[2]["asof"] = "2026-09-30T09:29:00+08:00"
    elif case == "negative_weight":
        data[0].loc[0, "target_weight"] = -0.1
    else:
        data[2]["positions"] = [{"asset": "600000", "quantity": 10.5, "sellable_quantity": 0}]
    with pytest.raises(ValueError):
        run(data)


def test_account_not_mutated_and_deterministic():
    data = inputs()
    before = copy.deepcopy(data[2])
    assert run(data) == run(data)
    assert data[2] == before


def test_recent_probe_catches_lot_share_units():
    frame = pd.DataFrame(
        [
            {
                "date": "2026-09-30",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10,
                "volume": 100,
                "amount": 100000,
            }
        ]
    )
    bars = normalize_bars(frame, "tencent")
    end = pd.Timestamp("2026-09-30")
    assert "amount_volume_implied_price_outside_bar_check_units" in audit_bars(bars, end, end)
    bars["volume"] *= 100
    assert audit_bars(bars, end, end) == []


def test_missing_last_session_fails_probe():
    frame = pd.DataFrame(
        [
            {
                "date": "2026-09-29",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10,
                "volume": 100,
                "amount": 1000,
            }
        ]
    )
    bars = normalize_bars(frame, "tencent")
    assert "not_at_expected_last_session" in audit_bars(
        bars, pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-30")
    )


def test_tencent_unit_fix_requires_known_version_and_implementation():
    source = 'if not symbol.startswith(("sh688", "sz399", "sh000", "sz000")):'
    assert tencent_volume_multiplier("000001", "1.19.1", source) == 100
    assert tencent_volume_multiplier("600000", "1.19.1", source) == 1
    assert tencent_volume_multiplier("000001", "1.19.2", source) == 1
    assert tencent_volume_multiplier("000001", "1.19.1", "already patched") == 1
