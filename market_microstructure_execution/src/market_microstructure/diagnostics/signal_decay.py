"""OOS-only horizon decay and per-asset rolling correlations."""

import numpy as np
import pandas as pd

from market_microstructure.labels.labeling import build_labels
from market_microstructure.models.evaluation import correlation


def signal_decay(book: pd.DataFrame, predictions: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """IC uses the same OOS timestamp set; label boundaries remain in the source book."""
    rows = []
    for horizon in horizons:
        labeled = book[["timestamp", "asset"]].join(build_labels(book, horizon_events=horizon))
        merged = predictions[["timestamp", "asset", "signal"]].merge(labeled, on=["timestamp", "asset"])
        for asset, group in merged.groupby("asset"):
            rows.append({"asset": asset, "horizon_events": horizon,
                         "ic": correlation(group.signal, group.future_return_bps),
                         "rows": int(group.future_return.notna().sum())})
    return pd.DataFrame(rows)


def decay_half_life(decay: pd.DataFrame) -> pd.DataFrame:
    """First sampled horizon at <= half peak |IC| after peak; censored if never reached.

    A descriptive IC half-life is not an exponential decay estimate.
    """
    rows = []
    for asset, group in decay.groupby("asset"):
        group = group.sort_values("horizon_events").dropna(subset=["ic"])
        if group.empty:
            continue
        peak = group.loc[group.ic.abs().idxmax()]
        later = group.loc[(group.horizon_events > peak.horizon_events) & (group.ic.abs() <= abs(peak.ic) / 2)]
        rows.append({"asset": asset, "peak_horizon_events": peak.horizon_events, "peak_ic": peak.ic,
                     "half_life_events": later.horizon_events.iloc[0] if len(later) else np.nan,
                     "censored": later.empty})
    return pd.DataFrame(rows)


def rolling_signal_ic(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing descriptive IC; future labels are for evaluation, not signal generation."""
    parts = []
    for asset, group in frame.groupby("asset"):
        part = group[["timestamp"]].copy()
        part["rolling_ic"] = group.signal.rolling(window, min_periods=max(3, window // 2)).corr(group.future_return_bps)
        part["asset"] = asset
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
