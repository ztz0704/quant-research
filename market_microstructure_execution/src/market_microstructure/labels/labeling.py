"""Event/time horizons and three-class/binary mid-price targets."""

import numpy as np
import pandas as pd


def build_labels(
    frame: pd.DataFrame, horizon_events: int | None = 10, horizon_ms: int | None = None,
    threshold_bps: float = 0.5, classification: str = "three_class",
) -> pd.DataFrame:
    """Label by asset; time horizon uses first observation at or after t + delta.

    Binary mode excludes neutral movements via NaN (does not relabel them as down).
    Missing future observations stay NaN with NaT label_end; never truncate silently.
    """
    if (horizon_events is None) == (horizon_ms is None):
        raise ValueError("Configure exactly one event or time horizon")
    horizon = horizon_ms if horizon_ms is not None else horizon_events
    if horizon <= 0 or threshold_bps < 0 or classification not in {"binary", "three_class"}:
        raise ValueError("Invalid labeling settings")
    frames = []
    group_keys = ["asset", "session_id"] if "session_id" in frame else ["asset"]
    for _, group in frame.groupby(group_keys, sort=False):
        timestamps = pd.DatetimeIndex(group.timestamp)
        if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
            raise ValueError("Labels require strictly ordered per-asset timestamps")
        mid = (group.bid_price_1.to_numpy() + group.ask_price_1.to_numpy()) / 2
        if horizon_ms is None:
            positions = np.arange(len(group)) + horizon_events
        else:
            clock_ns = timestamps.as_unit("ns").asi8
            positions = np.searchsorted(clock_ns, clock_ns + int(horizon_ms) * 1_000_000)
        valid = positions < len(group)
        future = np.full(len(group), np.nan)
        future[valid] = mid[positions[valid]]
        end = pd.Series(pd.NaT, index=group.index, dtype="datetime64[ns, UTC]")
        end.loc[valid] = timestamps[positions[valid]]
        returns = (future - mid) / mid
        direction = np.where(returns * 1e4 > threshold_bps, 1.0,
                             np.where(returns * 1e4 < -threshold_bps, -1.0, 0.0))
        direction[~valid] = np.nan
        if classification == "binary":
            direction = np.where(direction == 0, np.nan, np.where(direction == -1, 0, direction))
        frames.append(pd.DataFrame({
            "future_return": returns, "future_return_bps": returns * 1e4,
            "future_price_change": future - mid, "direction": direction, "label_end": end,
        }, index=group.index))
    return pd.concat(frames).sort_index()
