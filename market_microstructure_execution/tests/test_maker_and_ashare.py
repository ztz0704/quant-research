"""Maker causality, deterministic RNG, impact, and cash-stock restrictions."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_event_loop import fixed_book

from market_microstructure.backtest.engine import BacktestConfig, run_backtest
from market_microstructure.costs.market_impact import temporary_impact
from market_microstructure.execution.fills import FillParameters, fill_probability


def test_impact_monotone_and_empty():
    assert temporary_impact(100, 10, 100, 1) > temporary_impact(100, 1, 100, 1) > 0
    assert temporary_impact(100, 0, 0, 1) == 0


def test_maker_probability_queue_and_adverse_pressure():
    params = FillParameters()
    assert fill_probability(1, 100, 50, -0.8, 0.5, params) > fill_probability(1, 100, 50, 0.8, 0.5, params)
    assert fill_probability(1, 100, 50, 0, 0.5, params) > fill_probability(1, 10000, 50, 0, 0.5, params)


def test_maker_resting_no_same_event_and_cash_bridge(book):
    frame, signal = fixed_book(book)
    frame["trade_size"], frame["trade_side"], frame["trade_price"] = 1000.0, -1, 99.0
    config = BacktestConfig(mode="maker", max_inventory=2, trade_size=2, latency_events=0,
                            fill_parameters=FillParameters(intercept=100), seed=5,
                            taker_fee_bps=0, maker_fee_bps=-0.5, fixed_slippage_bps=0, depth_slippage_bps=0)
    result = run_backtest(frame, signal, config)
    assert result.fills.fill_event.iloc[0] == 1
    assert result.fills.price.iloc[0] == 99
    assert result.fills.fees.iloc[0] < 0
    assert result.fills.liquidity.iloc[-1] == "taker"
    pd.testing.assert_frame_equal(result.fills, run_backtest(frame, signal, config).fills)
    np.testing.assert_allclose(result.events.net_pnl,
        result.events.gross_pnl - result.events.spread_cost - result.events.fees
        - result.events.slippage - result.events.impact, atol=1e-8)


def test_maker_cancels_or_expires_without_trade_flow(book):
    frame, signal = fixed_book(book)
    config = BacktestConfig(mode="maker", latency_events=0, maker_ttl_events=1)
    result = run_backtest(frame, signal, config)
    assert result.fills.empty
    assert "maker_expired" in set(result.orders.status)
    signal.loc[signal.index[2:], "signal"] = -1
    result = run_backtest(frame, signal, replace(config, maker_ttl_events=10))
    assert "signal_cancel" in set(result.orders.status)


def test_ashare_tplus1_no_short_lots_and_next_day_sell(book):
    frame, signal = fixed_book(book)
    config = BacktestConfig(market_type="ashare_cash", lot_size=100, max_inventory=100,
                            trade_size=100, latency_events=0, fixed_slippage_bps=0, depth_slippage_bps=0,
                            minimum_commission=5, taker_fee_bps=3, maker_fee_bps=3)
    result = run_backtest(frame, signal, config)
    assert result.events.position.min() >= 0 and result.events.position.iloc[-1] == 100
    assert result.fills.quantity.tolist() == [100]
    assert result.fills.fees.iloc[0] == pytest.approx(5)
    tomorrow = frame.copy()
    tomorrow["timestamp"] += pd.Timedelta(days=1)
    next_signal = tomorrow[["timestamp", "asset"]].copy()
    next_signal["signal"] = -1.0
    both = pd.concat([frame, tomorrow], ignore_index=True)
    both_signal = pd.concat([signal, next_signal], ignore_index=True)
    result = run_backtest(both, both_signal, config)
    assert result.fills.quantity.tolist() == [100, -100]
    assert result.events.position.iloc[-1] == 0
    assert result.events.cash.min() >= 0


def test_ashare_no_fractional_fill_and_affordability(book):
    frame, signal = fixed_book(book)
    config = BacktestConfig(market_type="ashare_cash", lot_size=100, max_inventory=100,
                            trade_size=100, latency_events=0, maker_fee_bps=3, taker_fee_bps=3,
                            initial_cash=1000, minimum_commission=5)
    result = run_backtest(frame, signal, config)
    assert result.fills.empty
    frame["ask_size_1"] = 113
    result = run_backtest(frame, signal, replace(config, initial_cash=1e6))
    assert (result.fills.quantity % 1 == 0).all()
    assert result.events.position.max() <= 100


def test_new_signal_cannot_cancel_past_interval_trade(book):
    frame, signal = fixed_book(book)
    frame["trade_size"], frame["trade_side"], frame["trade_price"] = 1000.0, -1, 99.0
    signal.loc[signal.index[1:], "signal"] = 0.0
    config = BacktestConfig(mode="maker", max_inventory=2, trade_size=2,
                            latency_events=0, liquidate_at_end=False,
                            fill_parameters=FillParameters(intercept=100))
    result = run_backtest(frame, signal, config)
    assert result.fills.iloc[0].fill_event == 1
    assert result.fills.iloc[0].quantity == 2
