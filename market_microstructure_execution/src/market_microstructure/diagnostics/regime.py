"""Regime cutoffs fit historical training data; PnL attributed by event cash bridge."""

import numpy as np
import pandas as pd

from market_microstructure.backtest.engine import BacktestResult
from market_microstructure.models.evaluation import correlation


def regime_analysis(history: pd.DataFrame, book: pd.DataFrame, signals: pd.DataFrame,
                    result: BacktestResult) -> pd.DataFrame:
    """Six marginal regimes, never an OOS quantile threshold fed into the strategy."""
    cutoffs = history.groupby("asset")[["realized_vol_bps", "spread_bps", "total_depth"]].median()
    sample = book.merge(signals[["timestamp", "asset", "signal", "direction", "class_prediction"]]
                        if "class_prediction" in signals else signals[["timestamp", "asset", "signal"]],
                        on=["timestamp", "asset"], suffixes=("", "_prediction"))
    events = result.events.copy()
    events["incremental_pnl"] = events.net_pnl.diff().fillna(events.net_pnl.iloc[0])
    events["incremental_spread"] = events.spread_cost.diff().fillna(events.spread_cost.iloc[0])
    sample = sample.merge(events[["timestamp", "asset", "incremental_pnl", "incremental_spread"]],
                          on=["timestamp", "asset"])
    rows = []
    dimensions = (("realized_vol_bps", "low_volatility", "high_volatility"),
                  ("spread_bps", "tight_spread", "wide_spread"),
                  ("total_depth", "low_liquidity", "high_liquidity"))
    for asset, group in sample.groupby("asset"):
        for column, low_name, high_name in dimensions:
            for mask, name in ((group[column] <= cutoffs.loc[asset, column], low_name),
                               (group[column] > cutoffs.loc[asset, column], high_name)):
                regime = group.loc[mask]
                regime_keys = pd.MultiIndex.from_frame(regime[["timestamp", "asset"]])
                fills = result.fills.loc[pd.MultiIndex.from_frame(result.fills[["timestamp", "asset"]]).isin(regime_keys)]
                order_events = result.orders.loc[pd.MultiIndex.from_frame(result.orders[["timestamp", "asset"]]).isin(regime_keys)] if len(result.orders) else result.orders
                arrival_ids = (order_events.loc[order_events.status.eq("maker_arrived"), "order_id"]
                               if len(order_events) else pd.Series(dtype=int))
                filled_ids = (result.fills.loc[result.fills.liquidity.eq("maker"), "order_id"]
                              if len(result.fills) else pd.Series(dtype=int))
                # Fill rate follows the arrival cohort, even if filled in a later regime.
                cohort_fill_rate = arrival_ids.isin(filled_ids).mean() if len(arrival_ids) else np.nan
                accuracy = np.nan
                if "class_prediction" in regime:
                    valid = regime.direction.notna()
                    accuracy = (regime.loc[valid, "class_prediction"] == regime.loc[valid, "direction"]).mean()
                rows.append({"asset": asset, "regime": name, "rows": len(regime),
                    "training_cutoff": cutoffs.loc[asset, column],
                    "signal_ic": correlation(regime.signal, regime.future_return_bps),
                    "prediction_accuracy": accuracy, "net_pnl": regime.incremental_pnl.sum(),
                    "spread_cost": regime.incremental_spread.sum(), "fill_count": len(fills),
                    "maker_order_fill_rate": cohort_fill_rate})
    return pd.DataFrame(rows)
