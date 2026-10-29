"""Threshold policy is fixed before observing the test sample."""

import numpy as np


def target_inventory(signal: float, threshold: float, max_inventory: float) -> float:
    """Long above +threshold, short below -threshold, flat in the neutral region."""
    if not 0 <= threshold < 1 or max_inventory <= 0:
        raise ValueError("Invalid signal threshold or inventory cap")
    if not np.isfinite(signal):
        return 0.0
    if signal > threshold:
        return max_inventory
    if signal < -threshold:
        return -max_inventory
    return 0.0


def order_quantity(target: float, position: float, trade_size: float, max_inventory: float) -> float:
    """Recompute pending target at arrival so stale orders cannot breach inventory limits."""
    if trade_size <= 0 or max_inventory <= 0:
        raise ValueError("Invalid order/inventory scale")
    target = float(np.clip(target, -max_inventory, max_inventory))
    return float(np.clip(target - position, -trade_size, trade_size))
