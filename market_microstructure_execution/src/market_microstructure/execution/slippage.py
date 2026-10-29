"""Direction-independent adverse slippage magnitude, in price units per share."""


def slippage_per_share(price: float, quantity: float, depth: float,
                       fixed_bps: float = 0.0, depth_coefficient_bps: float = 0.0) -> float:
    """Apply the sign at execution: add for buys and subtract for sells."""
    if fixed_bps < 0 or depth_coefficient_bps < 0:
        raise ValueError("Adverse slippage coefficients must be nonnegative")
    if quantity == 0:
        return 0.0
    if depth <= 0:
        raise ValueError("Cannot execute against empty depth")
    return price * (fixed_bps + depth_coefficient_bps * abs(quantity) / depth) / 1e4
