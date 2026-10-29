"""Cash/cost reconciliation, inventory caps, latency and multi-asset clocks."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from market_microstructure.backtest.engine import BacktestConfig, run_backtest
from market_microstructure.backtest.metrics import performance_metrics


def fixed_book(book):
    frame = book.loc[book.asset == "SYN_A", ["timestamp", "asset", "bid_price_1",
               "ask_price_1", "bid_size_1", "ask_size_1"]].iloc[:10].copy()
    frame["bid_price_1"], frame["ask_price_1"] = 99.0, 101.0
    frame["bid_size_1"], frame["ask_size_1"] = 1000.0, 1000.0
    signal = frame[["timestamp", "asset"]].copy()
    signal["signal"] = 1.0
    return frame, signal


def test_round_trip_cost_and_inventory(book):
    frame, signal = fixed_book(book)
    config = BacktestConfig(max_inventory=2, trade_size=2, latency_events=0,
                            taker_fee_bps=0, fixed_slippage_bps=0, depth_slippage_bps=0)
    result = run_backtest(frame, signal, config)
    assert result.events.position.abs().max() <= 2
    assert result.events.position.iloc[-1] == 0
    assert result.events.net_pnl.iloc[-1] == pytest.approx(-4)
    assert result.events.spread_cost.iloc[-1] == 4
    assert result.fills.price.tolist() == [101, 99]
    assert performance_metrics(result)["hit_rate"] == 0
    np.testing.assert_allclose(result.events.net_pnl,
        result.events.gross_pnl - result.events.spread_cost - result.events.fees
        - result.events.slippage - result.events.impact, atol=1e-8)


@pytest.mark.parametrize("latency", [0, 1, 5])
def test_latency_uses_arrival_quotes(book, latency):
    frame, signal = fixed_book(book)
    frame["bid_price_1"] += np.arange(len(frame))
    frame["ask_price_1"] += np.arange(len(frame))
    config = BacktestConfig(latency_events=latency, max_inventory=2, trade_size=2,
                            taker_fee_bps=0, fixed_slippage_bps=0, depth_slippage_bps=0)
    result = run_backtest(frame, signal, config)
    assert result.fills.fill_event.iloc[0] == latency
    assert result.fills.price.iloc[0] == 101 + latency


def test_multi_asset_cap_and_all_event_reconciliation(book):
    signal = book[["timestamp", "asset"]].copy()
    signal["signal"] = np.tile([1.0, -1.0], len(book) // 2)
    result = run_backtest(book, signal, BacktestConfig(latency_events=1, max_inventory=3, trade_size=2))
    assert result.events.position.abs().max() <= 3
    np.testing.assert_array_equal(result.fills.fill_event.iloc[:2], [1, 1])
    np.testing.assert_allclose(result.events.net_pnl,
        result.events.gross_pnl - result.events.spread_cost - result.events.fees
        - result.events.slippage - result.events.impact, atol=1e-8)
    np.testing.assert_allclose(result.events.gross_pnl,
                              result.events.realized_gross_alpha + result.events.inventory_mark_to_market,
                              atol=1e-8)


def test_zero_signal_and_unfilled_late_orders(book):
    frame, signal = fixed_book(book)
    signal["signal"] = 0
    result = run_backtest(frame, signal, BacktestConfig())
    assert result.fills.empty and result.events.net_pnl.eq(0).all()
    signal["signal"] = 1
    result = run_backtest(frame, signal, BacktestConfig(latency_events=100))
    assert result.fills.empty and result.orders.status.iloc[-1] == "session_end_cancel"


def test_backtest_future_perturbation_prefix(book):
    frame, signal = fixed_book(book)
    config = BacktestConfig(latency_events=1)
    result = run_backtest(frame, signal, config)
    changed = frame.copy()
    changed.loc[changed.index[5:], ["bid_price_1", "ask_price_1"]] += 10
    perturbed = run_backtest(changed, signal, config)
    pd.testing.assert_frame_equal(result.events.iloc[:5], perturbed.events.iloc[:5])
    unreal = run_backtest(frame, signal, replace(config, mode="mid"))
    assert unreal.events.net_pnl.iloc[-1] == 0
