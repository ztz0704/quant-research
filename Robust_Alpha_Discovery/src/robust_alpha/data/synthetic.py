"""Deterministic synthetic OHLCV data for pipeline validation only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_panel(
    *,
    start_date: str = "2018-01-02",
    n_days: int = 900,
    n_assets: int = 60,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a realistic-shaped panel without claiming real-market inference.

    The generator deliberately includes weak serial structure, staggered universe membership,
    changing liquidity, and occasional non-tradable dates. It is a software test fixture, not a
    calibrated market simulator.
    """

    if n_days < 130:
        raise ValueError("n_days must be at least 130 to support the longest V1 lookback.")
    if n_assets < 10:
        raise ValueError("n_assets must be at least 10 for cross-sectional evaluation.")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    assets = np.array([f"S{index:04d}" for index in range(n_assets)])

    market = rng.normal(0.00015, 0.008, size=n_days)
    market_vol = np.where(np.arange(n_days) < n_days // 2, 1.0, 1.35)
    market *= market_vol
    asset_beta = rng.normal(1.0, 0.15, size=n_assets)
    asset_drift = rng.normal(0.00005, 0.00012, size=n_assets)
    innovations = rng.normal(0.0, 0.013, size=(n_days, n_assets))

    log_returns = np.zeros((n_days, n_assets), dtype=float)
    log_returns[0] = asset_beta * market[0] + asset_drift + innovations[0]
    for day in range(1, n_days):
        # Positive persistence makes the synthetic fixture capable of exercising OOS selection.
        log_returns[day] = (
            asset_beta * market[day]
            + asset_drift
            + 0.18 * log_returns[day - 1]
            + innovations[day]
        )

    base_prices = rng.uniform(8.0, 80.0, size=n_assets)
    open_prices = base_prices * np.exp(np.cumsum(log_returns, axis=0))
    intraday = rng.normal(0.0, 0.004, size=(n_days, n_assets))
    close_prices = open_prices * np.exp(intraday)
    range_width = np.abs(rng.normal(0.006, 0.003, size=(n_days, n_assets)))
    high_prices = np.maximum(open_prices, close_prices) * np.exp(range_width)
    low_prices = np.minimum(open_prices, close_prices) * np.exp(-range_width)

    base_volume = rng.lognormal(mean=13.2, sigma=0.7, size=n_assets)
    volume_shock = rng.lognormal(mean=0.0, sigma=0.35, size=(n_days, n_assets))
    volume = base_volume * volume_shock * (1.0 + 8.0 * np.abs(log_returns))
    amount = volume * close_prices

    entry_day = rng.integers(0, min(60, n_days // 8) + 1, size=n_assets)
    exit_day = np.full(n_assets, n_days, dtype=int)
    delisted = rng.random(n_assets) < 0.08
    if delisted.any():
        exit_day[delisted] = rng.integers(int(n_days * 0.75), n_days, size=delisted.sum())
    day_index = np.arange(n_days)[:, None]
    universe_member = (day_index >= entry_day[None, :]) & (day_index < exit_day[None, :])
    random_halt = rng.random((n_days, n_assets)) < 0.012
    tradable = universe_member & ~random_halt

    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), n_assets),
            "asset": np.tile(assets, n_days),
            "open": open_prices.ravel(),
            "high": high_prices.ravel(),
            "low": low_prices.ravel(),
            "close": close_prices.ravel(),
            "volume": volume.ravel(),
            "amount": amount.ravel(),
            "universe_member": universe_member.ravel(),
            "tradable": tradable.ravel(),
        }
    )
    return frame.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)

