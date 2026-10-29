"""Cross-sectional metrics and serial-dependence-aware mean inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def correlation(x, y, rank: bool = True) -> float:
    """Constant predictions have undefined IC, not fabricated predictive skill."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(spearmanr(x, y).statistic if rank else pearsonr(x, y).statistic)


def daily_ic(frame: pd.DataFrame, prediction: str = "prediction") -> pd.DataFrame:
    """One observation per market date, rather than treating pooled stocks as iid."""
    valid = frame.loc[frame["target"].notna() & frame[prediction].notna()]
    rows = []
    for date, group in valid.groupby("date", sort=True):
        rows.append({"date": date, "ic": correlation(group[prediction], group.target, False),
                     "rank_ic": correlation(group[prediction], group.target), "assets": len(group)})
    return pd.DataFrame(rows, columns=["date", "ic", "rank_ic", "assets"])


def hac_mean(values, maxlags: int = 5) -> dict:
    """Estimate the mean with Newey-West covariance; small samples stay explicit."""
    sample = pd.Series(values).dropna().to_numpy(float)
    if len(sample) < 10 or np.std(sample) < 1e-12:
        return {"n_dates": len(sample), "mean": float(np.mean(sample)) if len(sample) else np.nan,
                "hac_se": np.nan, "hac_t": np.nan, "hac_p": np.nan}
    fit = sm.OLS(sample, np.ones((len(sample), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": min(maxlags, len(sample) - 1)})
    return {"n_dates": len(sample), "mean": float(fit.params[0]), "hac_se": float(fit.bse[0]),
            "hac_t": float(fit.tvalues[0]), "hac_p": float(fit.pvalues[0])}


def prediction_metrics(frame: pd.DataFrame, column: str, maxlags: int = 5) -> dict:
    """Pooled error metrics plus unannualized date-local IC dispersion."""
    valid = frame.loc[frame.target.notna() & frame[column].notna()]
    ic = daily_ic(valid, column)
    result = hac_mean(ic.rank_ic, maxlags)
    result.update(mse=mean_squared_error(valid.target, valid[column]), mae=mean_absolute_error(valid.target, valid[column]),
                  r2=r2_score(valid.target, valid[column]), mean_ic=ic.ic.mean(), mean_rank_ic=ic.rank_ic.mean(),
                  rank_ic_std=ic.rank_ic.std(ddof=1), rank_icir=ic.rank_ic.mean() / ic.rank_ic.std(ddof=1) if ic.rank_ic.std(ddof=1) > 0 else np.nan,
                  positive_rank_ic_ratio=ic.rank_ic.gt(0).mean() if ic.rank_ic.notna().any() else np.nan,
                  pooled_pearson=correlation(valid[column], valid.target, False),
                  pooled_spearman=correlation(valid[column], valid.target), rows=len(valid))
    return result
