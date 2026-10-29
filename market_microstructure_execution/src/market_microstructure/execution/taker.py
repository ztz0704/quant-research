"""Displayed Level-1 execution with optional capacity, fees and slippage."""

from dataclasses import dataclass

import numpy as np

from market_microstructure.costs.fees import transaction_fee
from market_microstructure.execution.slippage import slippage_per_share


@dataclass(frozen=True)
class Fill:
    """All cost fields are currency totals; spread may be negative for maker capture."""

    quantity: float
    price: float
    mid: float
    quoted_price: float
    spread_cost: float
    fees: float
    slippage: float
    impact: float = 0.0
    liquidity: str = "taker"


def execute_taker(
    quantity: float, bid: float, ask: float, bid_size: float, ask_size: float,
    fee_bps: float = 0.0, fixed_slippage_bps: float = 0.0,
    depth_slippage_bps: float = 0.0, max_participation: float = 1.0,
    impact_per_share: float = 0.0,
) -> Fill:
    """Buy at ask, sell at bid; clip quantity to displayed opposite-side capacity."""
    if not 0 < bid < ask or min(bid_size, ask_size) < 0 or not 0 < max_participation <= 1:
        raise ValueError("Invalid execution quote or participation limit")
    mid = (bid + ask) / 2
    if quantity == 0:
        return Fill(0, mid, mid, mid, 0, 0, 0)
    depth = ask_size if quantity > 0 else bid_size
    quantity = float(np.sign(quantity) * min(abs(quantity), depth * max_participation))
    if quantity == 0:
        return Fill(0, mid, mid, mid, 0, 0, 0)
    quote = ask if quantity > 0 else bid
    slip = slippage_per_share(quote, quantity, depth, fixed_slippage_bps, depth_slippage_bps)
    price = quote + np.sign(quantity) * (slip + impact_per_share)
    if price <= 0 or impact_per_share < 0:
        raise ValueError("Invalid executed price or impact")
    return Fill(quantity, price, mid, quote, quantity * (quote - mid),
                transaction_fee(quantity, price, fee_bps), abs(quantity) * slip,
                abs(quantity) * impact_per_share)


def execute_mid(quantity: float, mid: float) -> Fill:
    """UNREALISTIC UPPER BOUND: frictionless mid-price fills for raw signal comparison."""
    return Fill(quantity, mid, mid, mid, 0, 0, 0, liquidity="mid_upper_bound")
