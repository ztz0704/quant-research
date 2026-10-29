"""Causal as-of snapshot sampling, with original observation timestamps."""

import pandas as pd

from market_microstructure.data.validation import validate_orderbook


def resample_book(book: pd.DataFrame, frequency: str = "100ms") -> pd.DataFrame:
    """Sample last known quotes on a regular grid; do not duplicate trade flow.

    The first grid point is at/after the first observation, the last is at/before the
    last observation. Sampled trades are unavailable: use event data for flow studies.
    """
    book, _ = validate_orderbook(book)
    frames = []
    for asset, frame in book.groupby("asset", sort=False):
        grid = pd.DataFrame({"timestamp": pd.date_range(
            frame.timestamp.iloc[0].ceil(frequency), frame.timestamp.iloc[-1].floor(frequency),
            freq=frequency,
        )})
        quotes = frame.drop(columns=[c for c in frame if c.startswith("trade_")
                                     or c == "signed_trade_size"])
        quotes = quotes.rename(columns={"timestamp": "observed_timestamp"})
        result = pd.merge_asof(grid, quotes, left_on="timestamp", right_on="observed_timestamp",
                               direction="backward")
        result["asset"] = asset
        frames.append(result)
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "asset"]).reset_index(drop=True)
