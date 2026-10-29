"""Exact discrete Almgren-Chriss objective with explicit price/risk units.

For sells, execution precedes the interval's random innovation and permanent impact:
S_tilde[k] = S[k-1] - half_spread - eta*n[k]/tau;
S[k] = S[k-1] + sigma*sqrt(tau)*z[k] - gamma*n[k].
This timing gives adjusted temporary coefficient eta/tau - gamma/2.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class ExecutionParameters:
    """CNY, shares and seconds; sigma is CNY/sqrt(second), lambda is 1/CNY."""

    order_size: float
    intervals: int
    duration_seconds: float
    sigma: float
    eta: float
    gamma: float = 0.0
    half_spread: float = 0.0

    @property
    def tau(self) -> float:
        """Seconds per interval."""
        return self.duration_seconds / self.intervals

    @property
    def adjusted_impact(self) -> float:
        """Positive discrete quadratic coefficient (CNY/share)."""
        return self.eta / self.tau - self.gamma / 2

    def validate(self) -> None:
        """Reject an indefinite objective, rather than silently clipping schedules."""
        values = (self.order_size, self.duration_seconds, self.sigma, self.eta,
                  self.gamma, self.half_spread)
        if (not np.isfinite(values).all() or min(values) < 0 or self.order_size <= 0
                or self.duration_seconds <= 0 or self.intervals < 1
                or not isinstance(self.intervals, int)):
            raise ValueError("Execution parameters must be finite with positive size/time/intervals")
        if self.adjusted_impact <= 0:
            raise ValueError("Require eta/tau > gamma/2 for a convex discrete objective")


def optimal_inventory(parameters: ExecutionParameters, risk_aversion: float) -> np.ndarray:
    """Solve the positive-definite tridiagonal Euler equations in O(N) storage."""
    parameters.validate()
    if not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError("Risk aversion must be finite and nonnegative")
    count, size = parameters.intervals, parameters.order_size
    if count == 1:
        return np.array([size, 0.0])
    if risk_aversion == 0 or parameters.sigma == 0:
        return np.linspace(size, 0, count + 1)
    a = parameters.adjusted_impact
    b = risk_aversion * parameters.sigma**2 * parameters.tau
    band = np.zeros((3, count - 1))
    band[0, 1:], band[2, :-1] = -a, -a
    band[1] = 2 * a + b
    rhs = np.zeros(count - 1)
    rhs[0] = a * size
    interior = solve_banded((1, 1), band, rhs)
    return np.r_[size, interior, 0.0]


def schedule_costs(inventory: np.ndarray, parameters: ExecutionParameters,
                   risk_aversion: float = 0.0) -> dict[str, float]:
    """Analytic shortfall mean/variance for any nonnegative monotone sell schedule."""
    parameters.validate()
    inventory = np.asarray(inventory, dtype=float)
    if (len(inventory) != parameters.intervals + 1 or not np.isfinite(inventory).all()
            or not np.isclose(inventory[0], parameters.order_size)
            or not np.isclose(inventory[-1], 0) or (np.diff(inventory) > 1e-7).any()
            or (inventory < -1e-7).any() or risk_aversion < 0):
        raise ValueError("Schedule must finish the complete positive order without reversals")
    trades = -np.diff(inventory)
    temporary = parameters.eta / parameters.tau * np.dot(trades, trades)
    permanent = parameters.gamma / 2 * (parameters.order_size**2 - np.dot(trades, trades))
    spread = parameters.half_spread * parameters.order_size
    mean = float(temporary + permanent + spread)
    variance = float(parameters.sigma**2 * parameters.tau * np.dot(inventory[1:], inventory[1:]))
    return {"expected_cost_cny": mean, "cost_variance_cny2": variance,
            "cost_std_cny": np.sqrt(variance), "objective_cny": mean + risk_aversion * variance,
            "temporary_cost_cny": float(temporary), "permanent_cost_cny": float(permanent),
            "spread_cost_cny": float(spread)}


def simulate_shortfall(inventory: np.ndarray, parameters: ExecutionParameters,
                       paths: int = 10000, seed: int = 0) -> np.ndarray:
    """Independent model validation via price paths; never presented as real-market evidence."""
    schedule_costs(inventory, parameters)
    if paths < 2:
        raise ValueError("Need at least two Monte Carlo paths")
    rng = np.random.default_rng(seed)
    price = np.zeros(paths)
    proceeds = np.zeros(paths)
    for quantity in -np.diff(inventory):
        proceeds += quantity * (price - parameters.half_spread - parameters.eta * quantity / parameters.tau)
        price += parameters.sigma * np.sqrt(parameters.tau) * rng.normal(size=paths) - parameters.gamma * quantity
    return -proceeds
