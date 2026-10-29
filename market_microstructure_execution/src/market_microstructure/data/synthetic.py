"""Synthetic snapshots with a known causal imbalance-to-price coefficient."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter


@dataclass(frozen=True)
class SyntheticConfig:
    """Price parameters are currency/share; depth and trades are shares."""

    events: int = 6000
    assets: tuple[str, ...] = ("SYN_A", "SYN_B")
    levels: int = 5
    start: str = "2026-01-05T09:30:00Z"
    event_interval_ms: int = 100
    initial_mid: float = 100.0
    tick_size: float = 0.01
    spread_ticks: float = 2.0
    beta: float = 0.003
    sigma: float = 0.006
    imbalance_persistence: float = 0.75
    base_depth: float = 1000.0
    trade_probability: float = 0.5
    trade_size_mean: float = 50.0
    regime_events: int = 750
    high_vol_multiplier: float = 2.0


def generate_orderbook(config: SyntheticConfig, seed: int = 20261003) -> pd.DataFrame:
    """Generate T snapshots per asset; future innovations never enter today's book.

    mid[t+1]-mid[t] = beta * imbalance[t] + sigma[t+1] * epsilon[t+1].
    Continuous prices deliberately avoid imposing a tick-rounding bias on beta recovery.
    """
    c = config
    if c.events < 2 or not 1 <= c.levels <= 5 or not c.assets:
        raise ValueError("Require events >= 2, assets, and 1..5 levels")
    if len(set(c.assets)) != len(c.assets) or any(not str(a).strip() for a in c.assets):
        raise ValueError("Assets must be unique and nonempty")
    if not 0 <= c.imbalance_persistence < 1 or not 0 <= c.trade_probability <= 1:
        raise ValueError("Invalid persistence or trade probability")
    positive = (c.event_interval_ms, c.initial_mid, c.tick_size, c.spread_ticks,
                c.base_depth, c.trade_size_mean, c.regime_events, c.high_vol_multiplier)
    if min(positive) <= 0 or c.sigma < 0:
        raise ValueError("Invalid synthetic scale")
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range(c.start, periods=c.events, freq=f"{c.event_interval_ms}ms")
    timestamp = timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")
    frames = []
    for asset_number, asset in enumerate(c.assets):
        latent = lfilter([np.sqrt(1 - c.imbalance_persistence**2)],
                         [1, -c.imbalance_persistence], rng.normal(size=c.events))
        imbalance = np.tanh(latent)
        regime = (np.arange(c.events) // c.regime_events) % 2
        sigma = c.sigma * np.where(regime == 1, c.high_vol_multiplier, 1.0)
        change = c.beta * imbalance[:-1] + sigma[1:] * rng.normal(size=c.events - 1)
        mid = c.initial_mid * (1 + 0.25 * asset_number) + np.r_[0.0, np.cumsum(change)]
        spread = c.tick_size * c.spread_ticks * (1 + 0.5 * regime)
        depth = c.base_depth * rng.lognormal(0, 0.2, c.events) / (1 + 0.3 * regime)
        frame = pd.DataFrame({"timestamp": timestamp, "asset": asset})
        for level in range(1, c.levels + 1):
            frame[f"bid_price_{level}"] = mid - spread / 2 - (level - 1) * c.tick_size
            frame[f"ask_price_{level}"] = mid + spread / 2 + (level - 1) * c.tick_size
            frame[f"bid_size_{level}"] = depth * (1 + imbalance) / 2 * level**0.3
            frame[f"ask_size_{level}"] = depth * (1 - imbalance) / 2 * level**0.3
        arrivals = rng.random(c.events) < c.trade_probability
        side = np.where(rng.random(c.events) < (1 + imbalance) / 2, 1, -1)
        frame["trade_size"] = np.where(arrivals, rng.exponential(c.trade_size_mean, c.events), 0)
        frame["trade_side"] = np.where(arrivals, side, 0)
        frame["trade_price"] = np.where(arrivals, mid + side * spread / 2, np.nan)
        # Truth columns are diagnostic metadata and must never be model features.
        frame["synthetic_regime"] = regime
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "asset"])
    if (result["bid_price_1"] <= 0).any():
        raise ValueError("Synthetic path produced nonpositive prices; reduce noise or length")
    return result.reset_index(drop=True)
