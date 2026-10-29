"""Fixed exploratory diagnostics, with calendar-paired estimands and coverage."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from robust_alpha.evaluation.conditional_reversal import calendar_hac
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from financial_ml.metrics import correlation
from financial_ml.models import make_model
from financial_ml.research_v2 import calendar_block_ci, fit_calibration, matched_predictions


def paired_estimate(values: pd.Series, cfg: dict, estimand: str) -> dict:
    low, high = calendar_block_ci(
        values,
        cfg["statistics"]["bootstrap_replications"],
        cfg["statistics"]["block_days"],
        cfg["seed"],
    )
    return {
        "estimand": estimand,
        **calendar_hac(values, cfg["statistics"]["hac_lags"]),
        "expected_dates": len(values),
        "coverage": float(values.notna().mean()),
        "block_ci_low": low,
        "block_ci_high": high,
        "evidence": "exploratory_observed_pairs_only",
        "full_window_observed": bool(values.notna().all()),
        "confirmatory": False,
    }


def model_diagnostics(
    archive: pd.DataFrame,
    panel: pd.DataFrame,
    fits: pd.DataFrame,
    models: dict,
    features: list[str],
    cfg: dict,
) -> dict:
    matched, _ = matched_predictions(archive)
    dates = pd.DatetimeIndex(archive.attrs["oos_calendar"])
    daily, grouped, importance, classifications = [], [], [], []
    # State thresholds are fixed from history strictly before first OOS decision.
    past = panel.loc[panel.date.lt(dates[0]) & panel.eligible]
    state_threshold = float(past.groupby("date").raw__volatility_20.median().median())
    current_vol = panel.loc[panel.eligible].groupby("date").raw__volatility_20.median()
    for date, section in matched.groupby("date"):
        valid = section.loc[section.target_valid & section.target.notna()].copy()
        size_rank = section.loc[section.model.eq("ridge")].set_index("asset")["size"].rank(pct=True)
        size_groups = pd.cut(size_rank, [0, 1 / 3, 2 / 3, 1], labels=["small", "mid", "large"])
        classifications.extend(
            {
                "date": date,
                "asset": asset,
                "size_group": str(group),
                "classification_asof": section.feature_asof.max(),
            }
            for asset, group in size_groups.items()
        )
        valid["size_group"] = valid.asset.map(
            pd.cut(size_rank, [0, 1 / 3, 2 / 3, 1], labels=["small", "mid", "large"])
        )
        for model, day in valid.groupby("model"):
            daily.append(
                {
                    "date": date,
                    "model": model,
                    "mse_return": float(np.mean((day.expected_return - day.target) ** 2)),
                    "mae_return": float(np.mean(np.abs(day.expected_return - day.target))),
                    "mean_forecast": day.expected_return.mean(),
                    "mean_target": day.target.mean(),
                    "assets": len(day),
                    "model_age_sessions": day.model_age_sessions.iloc[0],
                    "volatility_state": "high"
                    if current_vol.get(date, np.nan) > state_threshold
                    else "low",
                }
            )
        for dimension in ("industry", "size_group", "liquidity_group"):
            for (group, model), day in valid.groupby([dimension, "model"], observed=True):
                grouped.append(
                    {
                        "date": date,
                        "dimension": dimension,
                        "group": str(group),
                        "model": model,
                        "rank_ic": correlation(day.score, day.target),
                        "mse_return": float(np.mean((day.expected_return - day.target) ** 2)),
                        "assets": len(day),
                    }
                )
    daily = pd.DataFrame(daily)
    estimates, differences = [], []
    for metric in ("mse_return", "mae_return"):
        pivot = daily.pivot(index="date", columns="model", values=metric)
        values = (pivot.boosting - pivot.ridge).reindex(dates)
        estimates.append(paired_estimate(values, cfg, f"HGB_minus_Ridge_{metric}"))
        differences.append(
            pd.DataFrame({"date": dates, "metric": metric, "difference": values.to_numpy()})
        )
    states = []
    state_map = daily.loc[daily.model.eq("ridge")].set_index("date").volatility_state
    for state in ("low", "high"):
        pivot = daily.pivot(index="date", columns="model", values="mse_return")
        values = (
            (pivot.boosting - pivot.ridge).reindex(dates).where(state_map.reindex(dates).eq(state))
        )
        states.append(
            {
                "state": state,
                "state_threshold_past_daily_median_volatility": state_threshold,
                **paired_estimate(values, cfg, "HGB_minus_Ridge_MSE_conditional_state"),
            }
        )
    age = daily.assign(
        age_bin=pd.cut(
            daily.model_age_sessions, [0, 35, 49, 10000], labels=["early", "middle", "late"]
        )
    )
    age_summary = (
        age.groupby(["model", "age_bin"], observed=True)
        .agg(
            dates=("date", "nunique"),
            mse_return=("mse_return", "mean"),
            mae_return=("mae_return", "mean"),
        )
        .reset_index()
    )
    # Same deployed estimator, date-local permutation; no re-selection from importance.
    with threadpool_limits(limits=1):
        for (fold, model), section in matched.groupby(["oos_fold", "model"]):
            section = (
                section.loc[section.target_valid & section.target.notna()]
                .copy()
                .reset_index(drop=True)
            )
            estimator = models[f"fold_{fold}_{model}"]["estimator"]
            original = section.score.to_numpy()
            base = np.nanmean(
                [
                    correlation(original[idx], section.loc[idx, "target"])
                    for idx in section.groupby("date").groups.values()
                ]
            )
            for feature in features:
                for repeat in range(3):
                    rng = np.random.default_rng(
                        cfg["diagnostics"]["permutation_seed"] + repeat + 31 * fold
                    )
                    perturbed = section[features].copy()
                    for idx in section.groupby("date").groups.values():
                        perturbed.loc[idx, feature] = rng.permutation(
                            perturbed.loc[idx, feature].to_numpy()
                        )
                    scores = estimator.predict(perturbed.to_numpy())
                    value = np.nanmean(
                        [
                            correlation(scores[idx], section.loc[idx, "target"])
                            for idx in section.groupby("date").groups.values()
                        ]
                    )
                    importance.append(
                        {
                            "fold": fold,
                            "model": model,
                            "feature": feature,
                            "repeat": repeat,
                            "oos_rank_ic_drop": base - value,
                            "observed_label_rows": len(section),
                        }
                    )
    importance = pd.DataFrame(importance)
    fold_importance = (
        importance.groupby(["model", "fold", "feature"]).oos_rank_ic_drop.mean().reset_index()
    )
    fold_importance["importance_rank"] = fold_importance.groupby(
        ["model", "fold"]
    ).oos_rank_ic_drop.rank(ascending=False)
    stability = (
        fold_importance.groupby(["model", "feature"])
        .agg(
            mean_ic_drop=("oos_rank_ic_drop", "mean"),
            std_ic_drop=("oos_rank_ic_drop", "std"),
            mean_rank=("importance_rank", "mean"),
            std_rank=("importance_rank", "std"),
            folds=("fold", "nunique"),
        )
        .reset_index()
    )
    cal = fits[
        [
            "number",
            "model",
            "calibration_intercept",
            "calibration_slope",
            "calibration_dates",
            "calibration_max_label_available",
            "calibration_fit_cutoff",
        ]
    ].copy()
    cal["negative_slope"] = cal.calibration_slope.lt(0)
    error_inference = pd.DataFrame(estimates)
    error_inference["holm_p_two_error_metrics"] = multipletests(
        error_inference.hac_p.fillna(1), method="holm"
    )[1]
    error_inference.loc[error_inference.hac_p.isna(), "holm_p_two_error_metrics"] = np.nan
    return {
        "model_return_error_daily": daily,
        "model_error_paired_inference": error_inference,
        "model_error_paired_daily": pd.concat(differences),
        "model_group_diagnostics": pd.DataFrame(grouped),
        "model_size_group_classifications": pd.DataFrame(classifications),
        "model_state_inference": pd.DataFrame(states),
        "model_age_diagnostics": age_summary,
        "calibration_stability": cal,
        "oos_permutation_importance": importance,
        "oos_importance_stability": stability,
    }


@threadpool_limits.wrap(limits=1)
def window_cadence_diagnostics(
    panel: pd.DataFrame, features: list[str], archive: pd.DataFrame, fits: pd.DataFrame, cfg: dict
) -> dict:
    """One-axis fixed sensitivities, common dates, first-fold frozen parameters.

    Each refit has its own past-only held-out calibration slice. All attempts,
    negative slopes and insufficient dates remain. This is an exploratory new
    schedule, not an amendment to the primary V2 hypothesis or model selection.
    """
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    primary = pd.DatetimeIndex(archive.attrs["oos_calendar"])
    first_eligible = calendar.get_indexer([panel.loc[panel.eligible, "date"].min()])[0]
    earliest = first_eligible + max(cfg["diagnostics"]["window_days"]) + 24
    common = primary[calendar.get_indexer(primary) >= earliest]
    variants = sorted(
        {(w, 42) for w in cfg["diagnostics"]["window_days"]}
        | {(126, step) for step in cfg["diagnostics"]["retrain_sessions"]}
    )
    parameters = {
        model: json.loads(
            fits.loc[(fits.number == fits.number.min()) & fits.model.eq(model), "parameters"].iloc[
                0
            ]
        )
        for model in ("ridge", "boosting")
    }
    results, attempts = [], []
    for window, cadence in variants:
        for model in ("ridge", "boosting"):
            estimator, mapping = None, None
            for index, date in enumerate(common):
                if index % cadence == 0:
                    pos = calendar.get_indexer([date])[0]
                    cal_start, cal_end = calendar[pos - 22], calendar[pos - 2]
                    train_start = calendar[pos - 22 - window]
                    train = panel.loc[
                        panel.date.ge(train_start)
                        & panel.date.lt(cal_start)
                        & panel.eligible
                        & panel.target_valid
                        & panel.label_available_at.lt(cal_start)
                    ]
                    calibration = panel.loc[
                        panel.date.ge(cal_start)
                        & panel.date.lt(cal_end)
                        & panel.eligible
                        & panel.target_valid
                        & panel.label_available_at.lt(date)
                    ]
                    record = {
                        "date": date,
                        "model": model,
                        "window": window,
                        "cadence": cadence,
                        "train_rows": len(train),
                        "calibration_rows": len(calibration),
                    }
                    try:
                        if len(train) < cfg["validation"]["min_train_rows"]:
                            raise ValueError("Insufficient fixed-window training rows")
                        estimator = make_model(model, parameters[model], cfg["seed"])
                        estimator.fit(train[features].to_numpy(), train.target.to_numpy())
                        calibration = calibration.assign(
                            score=estimator.predict(calibration[features].to_numpy())
                        )
                        mapping = fit_calibration(calibration, date)
                        record.update(
                            status="fit",
                            slope=mapping.slope,
                            max_fit_available=train.label_available_at.max(),
                            max_calibration_available=mapping.max_label_available,
                        )
                    except ValueError as error:
                        estimator, mapping = None, None
                        record.update(status="BLOCKED", reason=str(error))
                    attempts.append(record)
                current = panel.loc[panel.date.eq(date) & panel.eligible & panel.target_valid]
                record = {
                    "date": date,
                    "model": model,
                    "window": window,
                    "cadence": cadence,
                    "rank_ic": np.nan,
                    "mse_return": np.nan,
                    "assets": len(current),
                }
                if estimator is not None and len(current) >= 3:
                    score = estimator.predict(current[features].to_numpy())
                    record.update(
                        rank_ic=correlation(score, current.target),
                        mse_return=float(np.mean((mapping.predict(score) - current.target) ** 2)),
                    )
                results.append(record)
    results = pd.DataFrame(
        results, columns=["date", "model", "window", "cadence", "rank_ic", "mse_return", "assets"]
    )
    summary = []
    for (window, cadence), part in results.groupby(["window", "cadence"]):
        pivot = part.pivot(index="date", columns="model", values="rank_ic")
        summary.append(
            {
                "window": window,
                "cadence": cadence,
                **paired_estimate(
                    (pivot.boosting - pivot.ridge).reindex(common),
                    cfg,
                    "sensitivity_HGB_minus_Ridge_RankIC",
                ),
            }
        )
    return {
        "window_cadence_daily": results,
        "window_cadence_fit_attempts": pd.DataFrame(attempts),
        "window_cadence_paired_inference": pd.DataFrame(summary),
    }


def portfolio_inference(periods: pd.DataFrame, calibration: pd.DataFrame, cfg: dict) -> dict:
    calendar = pd.DatetimeIndex(sorted(periods.date.unique()))
    estimates, paired = [], []
    for bps in cfg["portfolio"]["cost_bps_per_full_l1"]:
        part = periods.loc[periods.cost_bps_per_full_l1.eq(bps)]
        pivot = part.pivot(index="date", columns="strategy", values="net_return")
        for first, second in (
            ("C_HGB", "B_Ridge"),
            ("E_HGB_risk_cost", "R_Ridge_risk_cost"),
            ("D_HGB_risk", "C_HGB"),
            ("E_HGB_risk_cost", "D_HGB_risk"),
        ):
            values = (pivot[first] - pivot[second]).reindex(calendar)
            comparison = f"{first}_minus_{second}"
            estimates.append(
                {
                    "cost_bps_per_full_l1": bps,
                    "comparison": comparison,
                    **paired_estimate(values, cfg, "net_return_observed_common_dates"),
                }
            )
            paired.append(
                pd.DataFrame(
                    {
                        "date": calendar,
                        "cost_bps_per_full_l1": bps,
                        "comparison": comparison,
                        "difference": values.to_numpy(),
                    }
                )
            )
    summary = pd.DataFrame(estimates)
    summary["holm_p_exploratory_family"] = multipletests(summary.hac_p.fillna(1), method="holm")[1]
    summary.loc[summary.hac_p.isna(), "holm_p_exploratory_family"] = np.nan
    risk_rows = []
    for (strategy, bps), part in calibration.groupby(["strategy", "cost_bps_per_full_l1"]):
        past_pred = part.predicted_variance.mean()
        actual_var = part.realized_gross_return.var(ddof=1)
        # Descriptive unconditional variance ratio, no forecast calibration claim.
        risk_rows.append(
            {
                "strategy": strategy,
                "cost_bps_per_full_l1": bps,
                "valid_dates": len(part),
                "expected_dates": len(calendar),
                "coverage": len(part) / len(calendar),
                "mean_predicted_variance": past_pred,
                "realized_sample_variance": actual_var,
                "variance_ratio": actual_var / past_pred if past_pred > 0 else np.nan,
                "full_window": len(part) == len(calendar),
            }
        )
    return {
        "portfolio_paired_inference": summary,
        "portfolio_paired_daily": pd.concat(paired),
        "risk_variance_calibration": pd.DataFrame(risk_rows),
    }
