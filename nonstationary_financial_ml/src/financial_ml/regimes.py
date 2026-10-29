"""Causal regime thresholds fitted only on the first training calendar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_ml.evaluation import load_research
from financial_ml.metrics import daily_ic
from financial_ml.temporal import walk_forward
from financial_ml.utils import output, write_json


def regime_dates(panel: pd.DataFrame, training_end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Returns through t-1 classify the regime at decision t."""
    market = panel.groupby("date", sort=True)["return"].agg(["mean", "std"])
    market["market_volatility"] = market["mean"].shift(1).rolling(20).std()
    market["market_trend"] = market["mean"].shift(1).rolling(20).sum()
    market["dispersion"] = market["std"].shift(1)
    training = market.loc[market.index < training_end]
    thresholds = {key: float(training[key].median()) for key in ("market_volatility", "dispersion")}
    market["volatility_regime"] = np.where(market.market_volatility >= thresholds["market_volatility"], "high_vol", "low_vol")
    market["trend_regime"] = np.where(market.market_trend >= 0, "bull_like", "bear_like")
    market["dispersion_regime"] = np.where(market.dispersion >= thresholds["dispersion"], "high_dispersion", "low_dispersion")
    for key, value in [("volatility_regime", "market_volatility"), ("trend_regime", "market_trend"), ("dispersion_regime", "dispersion")]:
        market.loc[market[value].isna(), key] = "unavailable"
    return market.reset_index(), thresholds


def study(cfg: dict) -> None:
    panel, predictions, _ = load_research(cfg)
    first = walk_forward(panel, cfg["validation"])[0]
    regimes, thresholds = regime_dates(panel, first.train_end)
    destination = output(cfg)
    regimes.to_csv(destination / "tables/regime_dates.csv", index=False)
    write_json({"training_end_exclusive": str(first.train_end), "thresholds": thresholds}, destination / "tables/regime_manifest.json")
    rows = []
    merged = predictions.merge(regimes, on="date", validate="many_to_one")
    for model, group in merged.groupby("model", sort=True):
        for dimension in ("volatility_regime", "trend_regime", "dispersion_regime"):
            for regime, subset in group.groupby(dimension, sort=True):
                ic = daily_ic(subset, "raw_prediction")
                rows.append({"model": model, "dimension": dimension, "regime": regime,
                             "mean_ic": ic.ic.mean(), "mean_rank_ic": ic.rank_ic.mean(), "dates": len(ic), "rows": len(subset)})
    pd.DataFrame(rows).to_csv(destination / "tables/regime_summary.csv", index=False)
