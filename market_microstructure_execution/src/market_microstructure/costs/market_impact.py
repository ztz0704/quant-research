"""Temporary nonlinear impact; exogenous replay does not feed permanent impact into quotes."""


def temporary_impact(price: float, quantity: float, depth: float,
                     coefficient_bps: float = 0.0, exponent: float = 0.5) -> float:
    """Return adverse price units/share; currency cost is |Q| times this magnitude."""
    if coefficient_bps < 0 or exponent <= 0:
        raise ValueError("Invalid impact parameters")
    if quantity == 0:
        return 0.0
    if depth <= 0:
        raise ValueError("Cannot compute impact for empty depth")
    return price * coefficient_bps / 1e4 * (abs(quantity) / depth) ** exponent
