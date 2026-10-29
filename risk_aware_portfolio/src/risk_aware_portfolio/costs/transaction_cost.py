"""Ex-ante convex cost and ex-post exact cost in fractions of pre-trade NAV."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cvxpy as cp


@dataclass(frozen=True)
class TransactionCost:
    """Impact coefficient multiplies absolute NAV trade to the 3/2 power.

    This is a reduced-form NAV model; it has no participation-rate/ADV capacity claim.
    """

    linear_bps: float = 5.0
    impact_coefficient: float = 0.002
    optimization_impact: bool = True
    trade_scale: float = 0.01
    sell_bps: float = 0.0
    sell_bps_schedule: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(
            [self.linear_bps, self.impact_coefficient, self.sell_bps, self.trade_scale]
        ).all():
            raise ValueError("Cost coefficients must be finite")
        if (
            min(self.linear_bps, self.impact_coefficient, self.sell_bps) < 0
            or self.trade_scale <= 0
        ):
            raise ValueError("Cost coefficients must be nonnegative")
        schedule = tuple(
            (date.fromisoformat(d).isoformat(), float(rate)) for d, rate in self.sell_bps_schedule
        )
        dates = [d for d, _ in schedule]
        if dates != sorted(set(dates)) or any(rate < 0 for _, rate in schedule):
            raise ValueError("Sell-tax schedule needs unique sorted dates and nonnegative rates")
        object.__setattr__(self, "sell_bps_schedule", schedule)

    def at_date(self, asof: object) -> "TransactionCost":
        """Use only the effective sell-tax rate at this execution date."""
        if not self.sell_bps_schedule:
            return self
        cutoff = date.fromisoformat(str(asof)[:10]).isoformat()
        rates = [rate for effective, rate in self.sell_bps_schedule if effective <= cutoff]
        if not rates:
            raise ValueError("No sell-tax rate known at execution date")
        return replace(self, sell_bps=rates[-1], sell_bps_schedule=())

    def realized(self, trade: np.ndarray) -> float:
        """Charge one-way absolute trades, including initial deployment."""
        magnitude = np.abs(np.asarray(trade, dtype=float))
        if not np.isfinite(magnitude).all():
            raise ValueError("Nonfinite trade")
        return float(
            self.linear_bps / 1e4 * magnitude.sum()
            + self.sell_bps / 1e4 * np.maximum(-np.asarray(trade), 0).sum()
            + self.impact_coefficient * np.power(magnitude, 1.5).sum()
        )

    def expression(self, trade: cp.Expression) -> cp.Expression:
        """Return a convex cost expression; fractional power is handled by Clarabel."""
        import cvxpy as cp

        magnitude = cp.abs(trade)
        linear = self.linear_bps / 1e4 * cp.sum(magnitude)
        if self.sell_bps:
            linear += self.sell_bps / 1e4 * cp.sum(cp.pos(-trade))
        if self.optimization_impact and self.impact_coefficient:
            return linear + self.impact_coefficient * self.trade_scale**1.5 * cp.sum(
                cp.power(magnitude / self.trade_scale, 1.5)
            )
        return linear

    def predicted(self, trade: np.ndarray) -> float:
        """Numeric value of the optimization cost, distinct from realized accounting."""
        if self.optimization_impact:
            return self.realized(trade)
        return float(
            self.linear_bps / 1e4 * np.abs(trade).sum()
            + self.sell_bps / 1e4 * np.maximum(-trade, 0).sum()
        )
