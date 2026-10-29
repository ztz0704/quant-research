"""Daily cross-sectional univariate factor returns and HAC inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def daily_factor_returns(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    *,
    target_column: str = "target_o2o",
    min_assets: int = 20,
) -> pd.DataFrame:
    """Estimate ``return = intercept + beta * signal`` in every date cross-section."""

    rows: list[dict[str, object]] = []
    for date, cross_section in frame.groupby("date", sort=True, observed=True):
        for signal in signal_columns:
            valid = cross_section[[signal, target_column]].dropna()
            beta = np.nan
            intercept = np.nan
            residual_std = np.nan
            if len(valid) >= min_assets and valid[signal].nunique() > 1:
                x = valid[signal].to_numpy(dtype=float)
                y = valid[target_column].to_numpy(dtype=float)
                design = np.column_stack([np.ones(len(x)), x])
                coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
                intercept, beta = coefficients
                residual = y - design @ coefficients
                residual_std = residual.std(ddof=2) if len(residual) > 2 else np.nan
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "signal": signal,
                    "factor_return": beta,
                    "intercept": intercept,
                    "residual_std": residual_std,
                    "n_assets": len(valid),
                }
            )
    return pd.DataFrame(rows)


def factor_return_summary(
    daily_returns: pd.DataFrame,
    *,
    hac_maxlags: int = 5,
) -> pd.DataFrame:
    """Report naive and Newey-West/HAC inference for mean factor returns."""

    rows: list[dict[str, object]] = []
    for signal, group in daily_returns.groupby("signal", sort=True, observed=True):
        values = group["factor_return"].dropna().to_numpy(dtype=float)
        count = len(values)
        mean = values.mean() if count else np.nan
        standard_deviation = values.std(ddof=1) if count > 1 else np.nan
        naive_standard_error = standard_deviation / np.sqrt(count) if count > 1 else np.nan
        naive_t = (
            mean / naive_standard_error
            if naive_standard_error and naive_standard_error > 0
            else np.nan
        )
        naive_p = 2.0 * stats.t.sf(abs(naive_t), count - 1) if count > 1 else np.nan

        hac_t = np.nan
        hac_p = np.nan
        hac_standard_error = np.nan
        if count > max(3, hac_maxlags + 1):
            model = sm.OLS(values, np.ones((count, 1))).fit(
                cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
            )
            hac_standard_error = float(model.bse[0])
            hac_t = float(model.tvalues[0])
            hac_p = float(model.pvalues[0])

        rows.append(
            {
                "signal": signal,
                "mean_factor_return": mean,
                "factor_return_std": standard_deviation,
                "naive_standard_error": naive_standard_error,
                "naive_t_stat": naive_t,
                "naive_pvalue_two_sided": naive_p,
                "hac_standard_error": hac_standard_error,
                "hac_t_stat": hac_t,
                "hac_pvalue_two_sided": hac_p,
                "n_dates": count,
            }
        )
    return pd.DataFrame(rows).sort_values("signal").reset_index(drop=True)


def factor_returns_wide(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Pivot long daily factor returns to a joint date-by-signal matrix."""

    return daily_returns.pivot(index="date", columns="signal", values="factor_return").sort_index()

