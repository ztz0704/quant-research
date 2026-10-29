"""Whitelisted causal features; labels and synthetic truth never enter model inputs."""

import numpy as np
import pandas as pd

from market_microstructure.data.validation import infer_levels, validate_orderbook
from market_microstructure.features.orderbook import book_features
from market_microstructure.features.volatility import realized_volatility

BASE_FEATURES = (
    "imbalance_1", "imbalance_k", "spread_bps", "microprice_deviation_bps", "log_depth",
    "return_1_bps", "realized_vol_bps", "spread_change_bps", "depth_change_pct",
    "imbalance_change", "order_flow_imbalance",
)
TRADE_FEATURES = ("trade_intensity", "signed_volume_imbalance")


def _order_flow(frame: pd.DataFrame) -> pd.Series:
    """Cont-style best-quote OFI contribution, using current and previous quotes."""
    bid, ask, bz, az = (frame[column] for column in (
        "bid_price_1", "ask_price_1", "bid_size_1", "ask_size_1",
    ))
    contribution = (
        (bid >= bid.shift()).astype(float) * bz
        - (bid <= bid.shift()).astype(float) * bz.shift()
        - (ask <= ask.shift()).astype(float) * az
        + (ask >= ask.shift()).astype(float) * az.shift()
    )
    return contribution


def _trade_direction(frame: pd.DataFrame) -> pd.Series:
    """Use provided side, otherwise causal tick rule on last observed trade prices."""
    if "trade_side" in frame:
        return frame.trade_side.astype(float)
    if "trade_price" not in frame:
        return pd.Series(np.nan, index=frame.index)
    active_price = frame.trade_price.where(frame.trade_size > 0).ffill()
    direction = np.sign(active_price.diff()).replace(0, np.nan).ffill().fillna(0)
    return direction.where(frame.trade_size > 0, 0)


def build_features(
    book: pd.DataFrame, levels: int = 1, rolling_window: int = 50,
    min_periods: int = 10, weighting: str = "equal",
) -> pd.DataFrame:
    """Preserve every event, including honest NaN warmup rows; only trailing windows."""
    book, _ = validate_orderbook(book)
    if levels > infer_levels(book) or not 1 <= min_periods <= rolling_window:
        raise ValueError("Unavailable book levels or invalid rolling window")
    frames = []
    group_keys = ["asset", "session_id"] if "session_id" in book else ["asset"]
    for _, frame in book.groupby(group_keys, sort=False):
        result = pd.concat([frame, book_features(frame, levels, weighting)], axis=1)
        returns = result.mid.pct_change(fill_method=None)
        result["return_1_bps"] = returns * 1e4
        result["realized_vol_bps"] = realized_volatility(returns, rolling_window, min_periods) * 1e4
        result["spread_change_bps"] = result.spread_bps.diff()
        result["depth_change_pct"] = result.total_depth.pct_change(fill_method=None)
        result["imbalance_change"] = result.imbalance_1.diff()
        result["order_flow_imbalance"] = (
            _order_flow(frame).rolling(rolling_window, min_periods=min_periods).sum()
            / result.total_depth.replace(0, np.nan)
        )
        if "trade_size" in frame:
            active = (frame.trade_size > 0).astype(float)
            result["trade_intensity"] = active.rolling(rolling_window, min_periods=min_periods).mean()
            signed_volume = frame.signed_trade_size if "signed_trade_size" in frame else frame.trade_size * _trade_direction(frame)
            volume = frame.trade_size.rolling(rolling_window, min_periods=min_periods).sum()
            signed = signed_volume.rolling(rolling_window, min_periods=min_periods).sum()
            result["signed_volume_imbalance"] = (signed / volume.replace(0, np.nan)).where(volume > 0, 0)
        frames.append(result)
    output = pd.concat(frames).sort_values(["timestamp", "asset"]).reset_index(drop=True)
    columns = list(BASE_FEATURES) + [name for name in TRADE_FEATURES if name in output]
    output[columns] = output[columns].replace([np.inf, -np.inf], np.nan)
    output.attrs["feature_columns"] = columns
    return output


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Never infer predictors from arbitrary numeric input columns."""
    return [name for name in (*BASE_FEATURES, *TRADE_FEATURES) if name in frame]
