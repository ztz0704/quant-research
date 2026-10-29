"""Fees charged on absolute executed notional."""


def transaction_fee(quantity: float, price: float, fee_bps: float) -> float:
    """Positive is a fee; negative is a rebate; zero quantity costs zero."""
    if price <= 0:
        raise ValueError("Fee price must be positive")
    return abs(quantity) * price * fee_bps / 1e4
