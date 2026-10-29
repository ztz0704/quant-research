"""Resting limits eligible only at later observations with contra flow or quote-through."""

from dataclasses import dataclass

import numpy as np

from market_microstructure.costs.fees import transaction_fee
from market_microstructure.execution.fills import FillParameters, fill_probability
from market_microstructure.execution.latency import Order
from market_microstructure.execution.taker import Fill


@dataclass
class LiveLimit:
    """A fixed limit price and approximate queue ahead at arrival."""

    order: Order
    price: float
    quantity: float
    queue: float
    placed_event: int


def maker_attempt(live: LiveLimit, event: dict, local_event: int, parameters: FillParameters,
                  fee_bps: float, rng: np.random.Generator, max_participation: float) -> tuple[Fill | None, dict]:
    """No same-event fills; no future markout affects the probability or eligibility."""
    side = np.sign(live.quantity)
    bid, ask = event["bid_price_1"], event["ask_price_1"]
    mid = (bid + ask) / 2
    volume = float(event.get("trade_size", 0))
    signed = float(event.get("signed_trade_size", volume * event.get("trade_side", 0)))
    trade_price = event.get("trade_price", np.nan)
    contra = side * signed < 0 and np.isfinite(trade_price) and side * (trade_price - live.price) <= 0
    through = ask <= live.price if side > 0 else bid >= live.price
    eligible = local_event > live.placed_event and (contra or through)
    depth = event["bid_size_1"] + event["ask_size_1"]
    imbalance = (event["bid_size_1"] - event["ask_size_1"]) / depth if depth else 0
    probability = fill_probability(side, live.queue, volume, imbalance,
                                   event.get("trade_intensity", 0), parameters) if eligible else 0.0
    diagnostics = {"fill_probability": probability, "eligible": eligible, "queue_ahead": live.queue,
                   "quote_through": through, "limit_price": live.price}
    if not eligible or rng.random() >= probability:
        return None, diagnostics
    capacity = volume * max_participation if contra else abs(live.quantity)
    quantity = side * min(abs(live.quantity), capacity)
    if quantity == 0:
        return None, diagnostics
    fill = Fill(quantity, live.price, mid, live.price, quantity * (live.price - mid),
                transaction_fee(quantity, live.price, fee_bps), 0, liquidity="maker")
    return fill, diagnostics
