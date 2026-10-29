"""Conservative main-board cash-stock constraints: lots, T+1, no shorting/leverage."""

import math

import pandas as pd

from market_microstructure.backtest.accounting import Portfolio


def constrained_quantity(quantity: float, asset: str, timestamp: pd.Timestamp,
                         portfolio: Portfolio, lot_size: int, ask: float,
                         commission_bps: float, minimum_commission: float) -> float:
    """Newly purchased shares cannot be sold in the same China trading date."""
    if lot_size < 1:
        raise ValueError("Positive board-lot size required")
    if quantity > 0:
        affordable = max(0, portfolio.cash - minimum_commission) / (ask * (1 + commission_bps / 1e4))
        return math.floor(min(quantity, affordable) / lot_size) * lot_size
    date = timestamp.tz_convert("Asia/Shanghai").date()
    settled = sum(lot.quantity for lot in portfolio.lots[asset]
                  if lot.quantity > 0 and lot.timestamp.tz_convert("Asia/Shanghai").date() < date)
    permitted = min(abs(quantity), settled, max(0, portfolio.positions[asset]))
    return -math.floor(permitted / lot_size) * lot_size
