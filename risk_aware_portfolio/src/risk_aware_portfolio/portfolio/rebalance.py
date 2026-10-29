"""Trading-date calendar schedules; no decisions derived from future returns."""

import pandas as pd


def is_rebalance(date: pd.Timestamp, previous_date: pd.Timestamp | None, frequency: str) -> bool:
    """Rebalance at first available session of each calendar week/month."""
    if previous_date is None or frequency == "daily":
        return True
    if frequency == "weekly":
        return date.to_period("W") != previous_date.to_period("W")
    if frequency == "monthly":
        return date.to_period("M") != previous_date.to_period("M")
    raise ValueError(f"Unknown frequency: {frequency}")
