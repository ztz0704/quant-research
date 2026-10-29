"""Cash ledger and FIFO closed-lot diagnostics, reconciled to marked equity."""

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_microstructure.execution.taker import Fill


@dataclass
class Lot:
    """An open signed inventory lot and allocated entry costs."""

    quantity: float
    price: float
    mid: float
    fee_per_share: float
    timestamp: pd.Timestamp
    event: int


class Portfolio:
    """Each instance owns its mutable state; no global positions or RNG."""

    def __init__(self, initial_cash: float):
        if initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, float] = defaultdict(float)
        self.marks: dict[str, float] = {}
        self.lots: dict[str, deque[Lot]] = defaultdict(deque)
        self.gross_alpha = 0.0
        self.realized_gross_alpha = 0.0
        self.costs = {"spread_cost": 0.0, "fees": 0.0, "slippage": 0.0, "impact": 0.0}
        self.closed_lots: list[dict] = []

    def mark(self, asset: str, mid: float) -> None:
        """Recognize the price move using inventory held before the current event."""
        if asset in self.marks:
            self.gross_alpha += self.positions[asset] * (mid - self.marks[asset])
        self.marks[asset] = mid

    def apply_fill(self, asset: str, fill: Fill, timestamp: pd.Timestamp, event: int) -> None:
        """Update cash and FIFO lots, including correct fee/rebate allocation on exits."""
        quantity = fill.quantity
        if quantity == 0:
            return
        self.cash -= quantity * fill.price + fill.fees
        for key in self.costs:
            self.costs[key] += getattr(fill, key)
        self.positions[asset] += quantity
        fee_per_share = fill.fees / abs(quantity)
        lots = self.lots[asset]
        while lots and quantity * lots[0].quantity < 0:
            lot = lots[0]
            matched = min(abs(quantity), abs(lot.quantity))
            side = np.sign(lot.quantity)
            gross = matched * side * (fill.mid - lot.mid)
            net = matched * (side * (fill.price - lot.price) - lot.fee_per_share - fee_per_share)
            self.realized_gross_alpha += gross
            self.closed_lots.append({"asset": asset, "entry_timestamp": lot.timestamp,
                "exit_timestamp": timestamp, "quantity": matched, "side": side,
                "gross_alpha": gross, "net_pnl": net, "holding_events": event - lot.event,
                "holding_ms": (timestamp - lot.timestamp).total_seconds() * 1000})
            quantity += matched * side
            lot.quantity -= matched * side
            if abs(lot.quantity) < 1e-10:
                lots.popleft()
            if abs(quantity) < 1e-10:
                quantity = 0.0
                break
        if quantity:
            lots.append(Lot(quantity, fill.price, fill.mid, fee_per_share, timestamp, event))

    @property
    def equity(self) -> float:
        """Portfolio value at the latest observable mid for each asset."""
        return self.cash + sum(quantity * self.marks[asset] for asset, quantity in self.positions.items())

    @property
    def unrealized_gross_alpha(self) -> float:
        """Remaining inventory MTM relative to each lot's entry mid; not extra PnL."""
        return sum(lot.quantity * (self.marks[asset] - lot.mid)
                   for asset, lots in self.lots.items() for lot in lots)

    def snapshot(self) -> dict:
        """Expose enough values to reconcile both cash and the complete cost bridge."""
        return {"cash": self.cash, "equity": self.equity,
                "gross_pnl": self.gross_alpha, "net_pnl": self.equity - self.initial_cash,
                "realized_gross_alpha": self.realized_gross_alpha,
                "inventory_mark_to_market": self.unrealized_gross_alpha, **self.costs}
