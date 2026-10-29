"""Backward-looking return features and explicitly lagged supplier exposures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_ml.preprocessing import transform

BASE_FEATURES = ["size", "momentum", "volatility", "liquidity"]
RETURN_FEATURES = ["rev_1", "rev_5", "mom_20", "mom_60", "vol_20", "vol_60"]


def build_features(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Features at decision date t depend on returns through t-lag, never y_t."""
    lag = int(cfg["features"]["additional_return_lag"])
    if lag < 1:
        raise ValueError("This close-to-close execution convention requires at least one lag")
    result = panel.copy()
    by_asset = result.groupby("asset", sort=False)["return"]
    names = RETURN_FEATURES.copy()
    for name in BASE_FEATURES:
        if name in result:
            names.append(name)
    if "size" not in result:
        result["size"] = result.groupby("asset", sort=False)["market_cap"].shift(lag).map(np.log)
        names.append("size")
    for name in ("x", "x_correlated"):
        if name in result:
            if not result.asset.str.startswith("SYN").all():
                raise ValueError("Controlled synthetic drivers cannot silently become real supplier features")
            # Synthetic x_t is explicitly observable at the decision instant and
            # generates r_(t+1); unlike trailing close-return features it needs no extra lag.
            names.append(name)
    if any(name in names for name in BASE_FEATURES if name != "size") and "feature_asof" not in result:
        raise ValueError("Supplier momentum/volatility/liquidity require a verified feature_asof field")
    result["rev_1"] = -by_asset.shift(lag)
    for window, name in [(5, "rev_5"), (20, "mom_20"), (60, "mom_60")]:
        values = by_asset.transform(lambda s: np.expm1(np.log1p(s).shift(lag).rolling(window).sum()))
        result[name] = -values if name.startswith("rev") else values
    for window in (20, 60):
        result[f"vol_{window}"] = by_asset.transform(lambda s: s.shift(lag).rolling(window).std(ddof=1))
    result["return_feature_asof"] = result.groupby("asset", sort=False)["date"].shift(lag)
    tradable = result["tradable"] if "tradable" in result else pd.Series(True, index=result.index)
    result["eligible"] = tradable & result[names].notna().all(axis=1)
    minimum = int(cfg["features"].get("min_cross_section", 3))
    if minimum < 3:
        raise ValueError("min_cross_section must permit at least three stocks for IC")
    # Count only current information. A future halt/target missingness must not
    # shrink today's prediction universe or date-local normalization sample.
    current_count = result.groupby("date", sort=False)["eligible"].transform("sum")
    result["eligible"] &= current_count.ge(minimum)
    for name in names:
        result[f"raw__{name}"] = result[name]
    return transform(result, names, cfg["features"]), names
