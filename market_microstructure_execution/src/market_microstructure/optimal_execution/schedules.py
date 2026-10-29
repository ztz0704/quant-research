"""Ex-ante comparison schedules and feasible A-share lot rounding."""

import numpy as np


def inventory_from_weights(size: float, weights: np.ndarray) -> np.ndarray:
    """Convert nonnegative forecast weights into a conserving inventory path."""
    weights = np.asarray(weights, dtype=float)
    if (size <= 0 or len(weights) < 1 or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError("Need positive size and nonnegative nonempty weights")
    traded = size * weights / weights.sum()
    inventory = np.r_[size, size - np.cumsum(traded)]
    inventory[-1] = 0.0
    return inventory


def comparison_schedules(size: float, intervals: int,
                         forecast_volume: np.ndarray) -> dict[str, np.ndarray]:
    """VWAP-style uses a forecast known at launch; never the future realized volume."""
    if len(forecast_volume) != intervals:
        raise ValueError("Volume forecast length must match execution intervals")
    return {"TWAP": inventory_from_weights(size, np.ones(intervals)),
            "VWAP_style": inventory_from_weights(size, forecast_volume),
            "front_loaded": inventory_from_weights(size, np.arange(intervals, 0, -1)),
            "back_loaded": inventory_from_weights(size, np.arange(1, intervals + 1))}


def round_lots(inventory: np.ndarray, lot_size: int = 100) -> np.ndarray:
    """Largest remainder rounds planned child orders; this is not an integer optimizer."""
    inventory = np.asarray(inventory, dtype=float)
    if lot_size <= 0 or not np.isclose(inventory[-1], 0) or (np.diff(inventory) > 1e-7).any():
        raise ValueError("Invalid inventory path or lot size")
    lots = inventory[0] / lot_size
    if not np.isclose(lots, round(lots)):
        raise ValueError("Parent size must be divisible by the lot size")
    fractional = -np.diff(inventory) / lot_size
    children = np.floor(fractional).astype(int)
    remainder = int(round(lots)) - children.sum()
    priority = np.argsort(-(fractional - children), kind="stable")
    children[priority[:remainder]] += 1
    return np.r_[inventory[0], inventory[0] - np.cumsum(children * lot_size)]
