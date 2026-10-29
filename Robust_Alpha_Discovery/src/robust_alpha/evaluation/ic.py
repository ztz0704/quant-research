"""Daily cross-sectional Spearman information coefficients."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def daily_spearman_ic(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    *,
    target_column: str = "target_o2o",
    min_assets: int = 20,
) -> pd.DataFrame:
    """Calculate date-by-date rank correlation for every signal."""

    if min_assets < 3:
        raise ValueError("min_assets must be at least 3.")
    required = ["date", target_column, *signal_columns]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Missing IC inputs: {missing}")

    rows: list[dict[str, object]] = []
    for date, cross_section in frame.groupby("date", sort=True, observed=True):
        for signal in signal_columns:
            valid = cross_section[[signal, target_column]].dropna()
            n_assets = len(valid)
            coefficient = np.nan
            if n_assets >= min_assets and valid[signal].nunique() > 1:
                coefficient = stats.spearmanr(
                    valid[signal].to_numpy(), valid[target_column].to_numpy()
                ).statistic
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "signal": signal,
                    "ic": coefficient,
                    "n_assets": n_assets,
                }
            )
    return pd.DataFrame(rows)


def summarize_ic(
    daily_ic: pd.DataFrame, *, annualization: int = 252, hac_maxlags: int = 5
) -> pd.DataFrame:
    """Summarize predictive strength without suppressing missing or negative results."""

    required = {"signal", "ic"}
    if not required.issubset(daily_ic):
        raise KeyError(f"daily_ic must contain {sorted(required)}")

    rows: list[dict[str, object]] = []
    for signal, group in daily_ic.groupby("signal", sort=True, observed=True):
        values = group["ic"].dropna().astype(float)
        count = len(values)
        standard_deviation = values.std(ddof=1) if count > 1 else np.nan
        mean = values.mean() if count else np.nan
        standard_error = standard_deviation / np.sqrt(count) if count > 1 else np.nan
        t_stat = mean / standard_error if standard_error and standard_error > 0 else np.nan
        pvalue = 2.0 * stats.t.sf(abs(t_stat), count - 1) if count > 1 else np.nan
        hac_standard_error = np.nan
        hac_t = np.nan
        hac_p = np.nan
        if count > max(3, hac_maxlags + 1):
            model = sm.OLS(values.to_numpy(), np.ones((count, 1))).fit(
                cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
            )
            hac_standard_error = float(model.bse[0])
            hac_t = float(model.tvalues[0])
            hac_p = float(model.pvalues[0])
        rows.append(
            {
                "signal": signal,
                "mean_ic": mean,
                "median_ic": values.median() if count else np.nan,
                "ic_std": standard_deviation,
                "icir": (
                    mean / standard_deviation * np.sqrt(annualization)
                    if standard_deviation and standard_deviation > 0
                    else np.nan
                ),
                "ic_hit_rate": (values > 0).mean() if count else np.nan,
                "ic_t_stat": t_stat,
                "ic_pvalue_two_sided": pvalue,
                "ic_hac_standard_error": hac_standard_error,
                "ic_hac_t_stat": hac_t,
                "ic_hac_pvalue_two_sided": hac_p,
                "n_dates": count,
            }
        )
    return pd.DataFrame(rows).sort_values("signal").reset_index(drop=True)
