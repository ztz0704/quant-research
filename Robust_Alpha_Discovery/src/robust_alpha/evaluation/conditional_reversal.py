"""Frozen temporary-price-pressure hypothesis, no candidate mining or sign flip."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests

TERMS = ["rev", "volshock", "rev_x_volshock"]
CONTROLS = ["size", "momentum_20", "volatility_20", "illiquidity_20"]


def calendar_hac(values: pd.Series, lags: int = 5) -> dict:
    """Intercept-only HAC retaining market-date gaps rather than compressing them."""
    if lags < 0:
        raise ValueError("HAC lags must be nonnegative")
    values = values.astype(float)
    n = int(values.notna().sum())
    mean = float(values.mean()) if n else np.nan
    result = {
        "n_dates": n,
        "missing_dates": int(values.isna().sum()),
        "mean": mean,
        "hac_se": np.nan,
        "hac_p": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
    }
    if n < 10:
        return result
    score = (values - mean).fillna(0).to_numpy()
    variance = float(score @ score)
    for lag in range(1, min(lags, len(score) - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(score[lag:] @ score[:-lag])
    se = np.sqrt(max(variance, 0)) / n
    result.update(
        hac_se=float(se),
        ci_low=float(mean - 1.96 * se),
        ci_high=float(mean + 1.96 * se),
        hac_p=float(2 * norm.sf(abs(mean / se))) if se > 1e-15 else np.nan,
    )
    return result


def _zscore(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=0)
    return (values - values.mean()) / sd if sd > 1e-12 else values * 0


def build_pressure_features(
    panel: pd.DataFrame, min_assets: int = 20, min_amount: float = 0
) -> tuple[pd.DataFrame, list[str]]:
    """REV=-log(Ct/Ct−5), volume surprise relative to previous 20 sessions.

    Rolling windows run over market-grid rows. A suspension/gap stays NaN; we
    do not jump to the next quotation. Float-cap proxy is explicitly identified
    for historical integration when total cap is absent.
    """
    result = panel.sort_values(["asset", "date"]).copy()
    logs = np.log(result.research_close)
    groups = result.asset
    returns = logs.groupby(groups).diff()
    result["rev"] = -(logs - logs.groupby(groups).shift(5))
    volume_mean = result.volume.groupby(groups).transform(lambda s: s.shift(1).rolling(20).mean())
    volume_ratio = (result.volume / volume_mean).where(result.volume.gt(0) & volume_mean.gt(0))
    result["volshock"] = np.log(volume_ratio).replace([np.inf, -np.inf], np.nan)
    result["momentum_20"] = logs - logs.groupby(groups).shift(20)
    result["volatility_20"] = returns.groupby(groups).transform(lambda s: s.rolling(20).std())
    amihud = returns.abs() / result.amount.where(result.amount.gt(0))
    result["illiquidity_20"] = amihud.groupby(groups).transform(lambda s: s.rolling(20).mean())
    result["amount_20"] = result.amount.groupby(groups).transform(lambda s: s.rolling(20).mean())
    cap = result.total_market_cap.where(result.total_market_cap.notna(), result.float_market_cap)
    result["cap_kind"] = np.where(result.total_market_cap.notna(), "total", "float_proxy")
    result["size"] = np.log(cap)
    base = ["rev", "volshock", *CONTROLS]
    result["eligible"] &= result[base].notna().all(axis=1) & result.industry.notna()
    result["eligible"] &= result.amount_20.ge(min_amount)
    counts = result.groupby("date").eligible.transform("sum")
    result["eligible"] &= counts.ge(min_assets)
    for name in base:
        result[f"raw__{name}"] = result[name]
        result[name] = result[name].where(result.eligible).groupby(result.date).transform(_zscore)
    result["rev_x_volshock"] = result.rev * result.volshock
    # Descriptive groups only, boundaries calculated within the current cross section.
    for name, source in (
        ("liquidity_group", "raw__illiquidity_20"),
        ("volatility_group", "raw__volatility_20"),
    ):
        ranks = result[source].where(result.eligible).groupby(result.date).rank(pct=True)
        result[name] = pd.cut(ranks, [0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"])
    result["feature_asof"] = result.decision_time
    return result.sort_values(["date", "asset"]).reset_index(drop=True), TERMS + CONTROLS


def conditional_research(panel: pd.DataFrame, min_assets: int = 20, lags: int = 5) -> dict:
    """Daily joint regression with industry/size/styles; time-series inference only."""
    records, ic_rows, group_rows = [], [], []
    for date, current in panel.groupby("date", sort=True):
        sample = current.loc[current.eligible & current.target_valid].copy()
        record = {
            "date": date,
            "eligible_assets": int(current.eligible.sum()),
            "valid_assets": len(sample),
            "status": "insufficient_rows",
            **{term: np.nan for term in TERMS},
        }
        if len(sample) >= min_assets:
            industry = pd.get_dummies(sample.industry, drop_first=True, dtype=float)
            x = np.column_stack([np.ones(len(sample)), sample[TERMS + CONTROLS], industry])
            coefficient, _, rank, _ = np.linalg.lstsq(x, sample.target, rcond=None)
            record["design_rank"], record["design_columns"] = rank, x.shape[1]
            record["design_condition_number"] = float(np.linalg.cond(x))
            record["rev_interaction_correlation"] = float(sample.rev.corr(sample.rev_x_volshock))
            if rank == x.shape[1] and len(sample) > rank + 2:
                record.update(status="ok", **dict(zip(TERMS, coefficient[1:4], strict=True)))
            else:
                record["status"] = "unidentified_design"
        records.append(record)
        for term in TERMS:
            valid = sample[[term, "target"]].dropna()
            rank_ic = (
                float(spearmanr(valid[term], valid.target).statistic)
                if len(valid) >= 3 and valid[term].std() > 1e-12
                else np.nan
            )
            ic_rows.append({"date": date, "term": term, "rank_ic": rank_ic, "assets": len(valid)})
        for dimension in ("liquidity_group", "volatility_group"):
            for group, subset in sample.groupby(dimension, observed=True):
                value = (
                    float(spearmanr(subset.rev, subset.target).statistic)
                    if len(subset) >= 3 and subset.rev.std() > 1e-12
                    else np.nan
                )
                group_rows.append(
                    {
                        "date": date,
                        "dimension": dimension,
                        "group": group,
                        "rank_ic": value,
                        "assets": len(subset),
                    }
                )
    daily = pd.DataFrame(records)
    coefficients = pd.DataFrame(
        [{"term": term, **calendar_hac(daily[term], lags)} for term in TERMS]
    )
    coefficients["holm_p"] = np.nan
    finite = coefficients.hac_p.notna()
    if finite.any():
        # Missing/unidentified terms remain in the declared three-test family.
        p = coefficients.hac_p.fillna(1).to_numpy()
        coefficients.loc[finite, "holm_p"] = multipletests(p, method="holm")[1][finite]
    ic = pd.DataFrame(ic_rows)
    summary = []
    for term, group in ic.groupby("term"):
        estimate = calendar_hac(group.rank_ic, lags)
        sd = group.rank_ic.std()
        summary.append(
            {
                "term": term,
                **estimate,
                "icir_annualized": estimate["mean"] / sd * np.sqrt(252) if sd > 0 else np.nan,
            }
        )
    return {
        "cross_sectional_regression_daily": daily,
        "factor_hac_summary": coefficients,
        "factor_daily_ic": ic,
        "factor_ic_summary": pd.DataFrame(summary),
        "factor_state_diagnostics": pd.DataFrame(group_rows),
    }
