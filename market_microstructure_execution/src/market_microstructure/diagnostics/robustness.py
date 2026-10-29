"""Conditional execution stresses hold OOS signals fixed except explicit added noise."""

from dataclasses import replace

import numpy as np
import pandas as pd

from market_microstructure.backtest.engine import BacktestConfig, run_backtest
from market_microstructure.backtest.metrics import performance_metrics


def robustness_suite(book: pd.DataFrame, signals: pd.DataFrame, config: dict) -> pd.DataFrame:
    """No model refit on stressed test prices; no final-fold model backfills earlier signals."""
    base = BacktestConfig.from_config(config, "taker")
    settings = config["robustness"]
    cases = [("baseline", "baseline", 1.0, book, signals, base)]
    for latency in settings["latencies"]:
        cases.append((f"latency_{latency}", "latency_events", latency, book, signals,
                      replace(base, latency_events=latency, latency_ms=None)))
    for multiplier in settings["spread_multipliers"]:
        changed = book.copy()
        mid = (book.bid_price_1 + book.ask_price_1) / 2
        for column in book:
            if column.startswith(("bid_price_", "ask_price_")):
                changed[column] = mid + (book[column] - mid) * multiplier
        cases.append((f"spread_{multiplier}", "spread_multiplier", multiplier, changed, signals, base))
    for multiplier in settings["fee_multipliers"]:
        cases.append((f"fees_{multiplier}", "fee_multiplier", multiplier, book, signals,
                      replace(base, taker_fee_bps=base.taker_fee_bps * multiplier,
                              minimum_commission=base.minimum_commission * multiplier)))
    for multiplier in settings["depth_multipliers"]:
        changed = book.copy()
        for column in book:
            if column.startswith(("bid_size_", "ask_size_")):
                changed[column] *= multiplier
        cases.append((f"depth_{multiplier}", "depth_multiplier", multiplier, changed, signals, base))
    noisy = signals.copy()
    rng = np.random.default_rng(config["seed"])
    noisy["signal"] = np.clip(signals.signal + rng.normal(0, settings["signal_noise_std"], len(signals)), -1, 1)
    cases.append(("signal_noise", "signal_noise_std", settings["signal_noise_std"], book, noisy, base))
    changed = book.copy()
    old_mid = (book.bid_price_1 + book.ask_price_1) / 2
    reference = old_mid.groupby(book.asset).transform("first")
    new_mid = reference + settings["volatility_multiplier"] * (old_mid - reference)
    for column in book:
        if column.startswith(("bid_price_", "ask_price_")) or column == "trade_price":
            changed[column] += new_mid - old_mid
    changed["realized_vol_bps"] *= settings["volatility_multiplier"]
    cases.append(("volatility_shock", "volatility_multiplier", settings["volatility_multiplier"], changed, signals, base))
    rows = []
    for name, dimension, parameter, quotes, scores, assumptions in cases:
        result = run_backtest(quotes, scores, assumptions)
        rows.append({"scenario": name, "dimension": dimension, "parameter": parameter,
                     "interpretation": "conditional execution sensitivity; fixed OOS signals",
                     **performance_metrics(result)})
    return pd.DataFrame(rows)
