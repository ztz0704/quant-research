"""Vectorized book formulas with explicit empty-depth behavior."""

import numpy as np
import pandas as pd


def book_features(frame: pd.DataFrame, levels: int = 1, weighting: str = "equal") -> pd.DataFrame:
    """Compute prices, depth, imbalance and size-weighted quote (microprice).

    With zero total depth, imbalance is neutral and microprice defaults to mid.
    Distance-decay weights are 1 / (1 + distance from best / best spread).
    """
    if not 1 <= levels <= 5 or weighting not in {"equal", "distance_decay"}:
        raise ValueError("Invalid book feature configuration")
    bid, ask = frame.bid_price_1.to_numpy(), frame.ask_price_1.to_numpy()
    bs, az = frame.bid_size_1.to_numpy(), frame.ask_size_1.to_numpy()
    mid, spread, depth = (bid + ask) / 2, ask - bid, bs + az
    imbalance = np.divide(bs - az, depth, out=np.zeros_like(depth), where=depth > 0)
    microprice = np.divide(az * bid + bs * ask, depth, out=mid.copy(), where=depth > 0)
    bid_depth, ask_depth = np.zeros(len(frame)), np.zeros(len(frame))
    weighted_bid, weighted_ask = np.zeros(len(frame)), np.zeros(len(frame))
    for level in range(1, levels + 1):
        bp, ap = frame[f"bid_price_{level}"].to_numpy(), frame[f"ask_price_{level}"].to_numpy()
        bz, a_size = frame[f"bid_size_{level}"].to_numpy(), frame[f"ask_size_{level}"].to_numpy()
        bw = 1.0 if weighting == "equal" else 1 / (1 + (bid - bp) / spread)
        aw = 1.0 if weighting == "equal" else 1 / (1 + (ap - ask) / spread)
        bid_depth += bz
        ask_depth += a_size
        weighted_bid += bz * bw
        weighted_ask += a_size * aw
    total = weighted_bid + weighted_ask
    ik = np.divide(weighted_bid - weighted_ask, total, out=np.zeros_like(total), where=total > 0)
    return pd.DataFrame({
        "mid": mid, "spread": spread, "spread_pct": spread / mid,
        "spread_bps": spread / mid * 1e4, "imbalance_1": imbalance,
        "imbalance_k": ik, "microprice": microprice,
        "microprice_deviation": microprice - mid,
        "microprice_deviation_bps": (microprice - mid) / mid * 1e4,
        "bid_depth": bid_depth, "ask_depth": ask_depth,
        "total_depth": bid_depth + ask_depth, "log_depth": np.log1p(bid_depth + ask_depth),
    }, index=frame.index)
