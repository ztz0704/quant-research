"""Maker markouts use post-fill labels only as diagnostics."""

import numpy as np
import pandas as pd

from market_microstructure.backtest.engine import BacktestResult
from market_microstructure.labels.labeling import build_labels


def maker_markouts(book: pd.DataFrame, result: BacktestResult, horizon_events: int) -> pd.DataFrame:
    """Signed mid-price markout: negative denotes adverse selection for either side."""
    fills = result.fills.loc[result.fills.liquidity == "maker"].copy()
    labels = book[["timestamp", "asset"]].join(build_labels(book, horizon_events=horizon_events))
    output = fills.merge(labels, on=["timestamp", "asset"], how="left", validate="many_to_one")
    output["signed_markout_bps"] = np.sign(output.quantity) * output.future_return_bps
    output["spread_capture"] = -output.spread_cost
    output["post_fill_mid_pnl"] = output.quantity * output.future_price_change
    return output
