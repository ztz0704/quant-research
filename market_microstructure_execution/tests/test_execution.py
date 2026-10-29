"""Quote side, signs, depth capacity and fee/rebate mechanics."""

import pytest

from market_microstructure.execution.latency import Order
from market_microstructure.execution.taker import execute_taker


def test_quote_side_and_zero_quantity():
    buy = execute_taker(3, 99, 101, 100, 100)
    sell = execute_taker(-3, 99, 101, 100, 100)
    assert buy.price == 101 and sell.price == 99
    assert buy.spread_cost == 3 and sell.spread_cost == 3
    zero = execute_taker(0, 99, 101, 100, 100, fee_bps=20, fixed_slippage_bps=10)
    assert zero.fees == zero.slippage == zero.spread_cost == 0


def test_slippage_and_capacity():
    buy = execute_taker(20, 99, 101, 100, 100, fixed_slippage_bps=10, max_participation=0.1)
    sell = execute_taker(-20, 99, 101, 100, 100, fixed_slippage_bps=10, max_participation=0.1)
    assert buy.quantity == 10 and sell.quantity == -10
    assert buy.price > 101 and sell.price < 99
    assert buy.slippage > 0 and sell.slippage > 0
    assert execute_taker(1, 99, 101, 100, 0).quantity == 0


def test_time_latency(book):
    timestamp = book.timestamp.iloc[0]
    order = Order(1, "A", 1, timestamp, 0, latency_events=None, latency_ms=150)
    assert not order.due(timestamp, 5)
    assert order.due(timestamp + __import__("pandas").Timedelta(milliseconds=200), 2)
    with pytest.raises(ValueError):
        Order(1, "A", 1, timestamp, 0, latency_events=1, latency_ms=1).due(timestamp, 1)
