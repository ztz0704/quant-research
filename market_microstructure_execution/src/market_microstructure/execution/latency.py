"""Asset-local event latency or elapsed UTC time latency."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Order:
    """Pending target inventory, never an unbounded stale delta."""

    order_id: int
    asset: str
    target: float
    decision_timestamp: pd.Timestamp
    decision_event: int
    latency_events: int | None = 1
    latency_ms: int | None = None

    def due(self, timestamp: pd.Timestamp, event: int) -> bool:
        """Return true only when the order has reached the local arrival clock."""
        if (self.latency_events is None) == (self.latency_ms is None):
            raise ValueError("Configure exactly one latency clock")
        if self.latency_ms is not None:
            if self.latency_ms < 0:
                raise ValueError("Latency cannot be negative")
            return timestamp >= self.decision_timestamp + pd.Timedelta(milliseconds=self.latency_ms)
        if self.latency_events < 0:
            raise ValueError("Latency cannot be negative")
        return event >= self.decision_event + self.latency_events
