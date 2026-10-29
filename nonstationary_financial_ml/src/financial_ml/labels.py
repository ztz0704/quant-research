"""Market-calendar forward labels with explicit outcome availability."""

import numpy as np
import pandas as pd


def add_labels(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Label t is compounded return on sessions t+1..t+h; eligibility is untouched."""
    if horizon < 1:
        raise ValueError("Forward horizon must be positive")
    result = frame.copy()
    groups = result.groupby("asset", sort=False)
    future_logs = [np.log1p(groups["return"].shift(-offset)) for offset in range(1, horizon + 1)]
    result["target"] = np.expm1(pd.concat(future_logs, axis=1).sum(axis=1, min_count=horizon))
    result["label_end"] = groups["date"].shift(-horizon)
    valid = result["target"].notna() & result["label_end"].notna()
    if "tradable" in result:
        valid &= result["tradable"] & groups["tradable"].shift(-horizon).eq(True)
    result["target_valid"] = valid
    result.loc[~valid, "target"] = np.nan
    result["direction"] = np.sign(result["target"])
    return result
