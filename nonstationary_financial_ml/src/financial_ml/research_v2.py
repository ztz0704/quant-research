"""Strict incremental prediction on the shared dynamic A-share contract.

The estimator is fit before the calibration slice and is not refit afterwards.
Calibration therefore sees genuinely held-out historical scores from the SAME
estimator deployed into the test fold. Validation tuning and calibration slices
are separated by the declared embargo; no global IC-dependent score scaling.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from robust_alpha.data.research_contract import RETURN_CONVENTION, identity
from robust_alpha.evaluation.conditional_reversal import calendar_hac
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from financial_ml.metrics import correlation
from financial_ml.models import make_model
from financial_ml.preprocessing import residualize
from financial_ml.temporal import partitions, walk_forward


@dataclass(frozen=True)
class ReturnCalibration:
    intercept: float
    slope: float
    fit_cutoff: pd.Timestamp
    max_label_available: pd.Timestamp
    rows: int
    dates: int

    def predict(self, score) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(score, dtype=float)


def fit_calibration(sample: pd.DataFrame, cutoff: pd.Timestamp) -> ReturnCalibration:
    """Equal weight per date, simple return units; no clipping/sign reversal."""
    if sample.empty or not sample.label_available_at.lt(cutoff).all():
        raise ValueError("Calibration labels have not matured before cutoff")
    if not np.isfinite(sample[["score", "target"]]).all().all():
        raise ValueError("Calibration needs finite historical scores and labels")
    if sample.date.nunique() < 10 or sample.score.std() <= 1e-12:
        raise ValueError("Insufficient calibration dates or constant historical scores")
    root = np.sqrt(1 / sample.groupby("date").score.transform("size").to_numpy())
    design = np.column_stack([np.ones(len(sample)), sample.score.to_numpy()])
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * root[:, None], sample.target * root, rcond=None
    )
    if rank != 2:
        raise ValueError("Unidentified score-to-return calibration")
    return ReturnCalibration(
        float(coefficients[0]),
        float(coefficients[1]),
        cutoff,
        sample.label_available_at.max(),
        len(sample),
        sample.date.nunique(),
    )


def _assert_maturity(sample: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    if sample.empty or not sample.label_available_at.lt(cutoff).all():
        raise ValueError("Fit labels cross the legal maturity boundary")


def _sample_identity(frame: pd.DataFrame, columns: list[str]) -> str:
    values = pd.util.hash_pandas_object(frame[columns].reset_index(drop=True), index=False)
    return hashlib.sha256(values.to_numpy().tobytes()).hexdigest()


@threadpool_limits.wrap(limits=1)
def train_incremental(
    panel: pd.DataFrame, features: list[str], cfg: dict, *, fold_numbers: set[int] | None = None
) -> tuple:
    """Freeze folds/models, tune in first validation half, calibrate in second."""
    if cfg["models"]["enabled"] != ["ridge", "boosting"]:
        raise ValueError("V2 primary comparison is frozen to Ridge and Histogram Boosting")
    predictions, fitted, tuning, artifacts = [], [], [], {}
    expected_test_dates = []
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    settings = cfg["validation"]
    for fold in walk_forward(panel, settings):
        if fold_numbers is not None and fold.number not in fold_numbers:
            continue
        expected_test_dates.extend(dates[(dates >= fold.test_start) & (dates < fold.test_end)])
        train, validation, test = partitions(panel, fold)
        val_dates = dates[(dates >= fold.validation_start) & (dates < fold.validation_end)]
        split = len(val_dates) // 2
        gap = settings["embargo_days"]
        if split - gap < 10 or len(val_dates) - split < 10:
            raise ValueError("Validation needs two >=10-session slices plus embargo")
        calibration_start = val_dates[split]
        tune_end = val_dates[split - gap]
        tune = validation.loc[
            validation.date.lt(tune_end) & validation.label_available_at.lt(calibration_start)
        ]
        calibration = validation.loc[
            validation.date.ge(calibration_start)
            & validation.label_available_at.lt(fold.test_start)
        ]
        train = train.loc[train.label_available_at.lt(fold.validation_start)]
        if len(train) < settings["min_train_rows"] or tune.empty or test.empty:
            raise ValueError(
                f"Fold {fold.number}: insufficient current eligible training/test rows"
            )
        _assert_maturity(train, fold.validation_start)
        _assert_maturity(tune, calibration_start)
        _assert_maturity(calibration, fold.test_start)
        fit = pd.concat([train, tune]).sort_values(["date", "asset"])
        for name in cfg["models"]["enabled"]:
            best, best_metric = None, -np.inf
            for candidate, parameters in enumerate(cfg["models"]["grids"][name]):
                estimator = make_model(name, parameters, cfg["seed"])
                estimator.fit(train[features].to_numpy(), train.target.to_numpy())
                score = estimator.predict(tune[features].to_numpy())
                evaluation = tune[["date", "target"]].assign(score=score)
                metric = np.nanmean(
                    [
                        correlation(group.score, group.target)
                        for _, group in evaluation.groupby("date")
                    ]
                )
                tuning.append(
                    {
                        "fold": fold.number,
                        "model": name,
                        "candidate": candidate,
                        "parameters": json.dumps(parameters, sort_keys=True),
                        "validation_rank_ic": metric,
                    }
                )
                if np.isfinite(metric) and (best is None or metric > best_metric):
                    best, best_metric = parameters, metric
            if best is None:
                raise ValueError(f"No identified validation candidate for {name}; no forced winner")
            estimator = make_model(name, best, cfg["seed"])
            estimator.fit(fit[features].to_numpy(), fit.target.to_numpy())
            historical = calibration.copy()
            historical["score"] = estimator.predict(historical[features].to_numpy())
            mapping = fit_calibration(historical, fold.test_start)
            sample_columns = ["date", "asset", *features, "target", "label_available_at"]
            version = identity(
                {
                    "model": name,
                    "parameters": best,
                    "seed": cfg["seed"],
                    "fit": _sample_identity(fit, sample_columns),
                    "calibration": _sample_identity(historical, sample_columns + ["score"]),
                    "implementation": "research-v2-temporal-calibration-1",
                }
            )
            columns = [
                "date",
                "asset",
                "decision_time",
                "feature_asof",
                "label_start",
                "label_end",
                "label_available_at",
                "entry_time",
                "exit_time",
                "target",
                "target_valid",
                "industry",
                *features,
                "return_convention",
                "return_unit",
                "entry_raw_open",
                "exit_raw_open",
                "entry_buyable",
                "entry_sellable",
                "exit_sellable",
                "liquidity_group",
                "volatility_group",
            ]
            result = test[columns].copy()
            result["score"] = estimator.predict(test[features].to_numpy())
            result["expected_return"] = mapping.predict(result.score)
            result["neutralized_score"] = np.nan
            for _, index in result.groupby("date").groups.items():
                section = result.loc[index]
                result.loc[index, "neutralized_score"] = residualize(
                    section.score.to_numpy(),
                    section,
                    ("size", "momentum_20", "volatility_20", "illiquidity_20"),
                )
            result["oos_fold"], result["model"] = fold.number, name
            result["model_fit_cutoff"] = fold.test_start
            result["estimator_fit_cutoff"] = calibration_start
            result["max_fit_label_available"] = fit.label_available_at.max()
            result["calibration_fit_cutoff"] = mapping.fit_cutoff
            result["max_calibration_label_available"] = mapping.max_label_available
            result["model_version"] = version
            result["model_age_sessions"] = (
                dates.get_indexer(result.date) - dates.get_indexer([calibration_start])[0]
            )
            result["signal_direction"] = "higher_expected_return_is_long"
            predictions.append(result)
            fitted.append(
                {
                    **asdict(fold),
                    "model": name,
                    "model_version": version,
                    "estimator_fit_cutoff": calibration_start,
                    "max_fit_label_available": fit.label_available_at.max(),
                    "fit_rows": len(fit),
                    "parameters": json.dumps(best, sort_keys=True),
                    **{f"calibration_{k}": v for k, v in asdict(mapping).items()},
                }
            )
            artifacts[f"fold_{fold.number}_{name}"] = {
                "estimator": estimator,
                "calibration": mapping,
                "features": features,
                "version": version,
            }
    archive = pd.concat(predictions, ignore_index=True).sort_values(["model", "date", "asset"])
    archive.attrs["oos_calendar"] = [pd.Timestamp(d).isoformat() for d in expected_test_dates]
    validate_archive(archive)
    return archive, pd.DataFrame(fitted), pd.DataFrame(tuning), artifacts


def validate_archive(archive: pd.DataFrame) -> None:
    if archive.duplicated(["model", "date", "asset"]).any():
        raise ValueError("Duplicate OOS identity")
    if not archive.return_convention.eq(RETURN_CONVENTION).all():
        raise ValueError("Incompatible OOS return convention")
    for field in ("feature_asof", "model_fit_cutoff", "calibration_fit_cutoff"):
        if archive[field].isna().any() or archive[field].gt(archive.decision_time).any():
            raise ValueError(f"Invalid prediction timestamp: {field}")
    if not archive.max_fit_label_available.lt(archive.estimator_fit_cutoff).all() or not (
        archive.max_calibration_label_available.lt(archive.calibration_fit_cutoff).all()
    ):
        raise ValueError("Prediction used immature labels")
    if not np.isfinite(archive[["score", "expected_return", "neutralized_score"]]).all().all():
        raise ValueError("Nonfinite OOS forecast")


def matched_predictions(archive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match model support using scores only; outcomes never define decisions."""
    if archive.duplicated(["model", "date", "asset"]).any():
        raise ValueError("Duplicate model prediction")
    coverage = []
    selected = []
    for date, day in archive.groupby("date", sort=True):
        sets = {
            name: set(day.loc[day.model.eq(name) & day.score.notna(), "asset"])
            for name in ("ridge", "boosting")
        }
        common = sets["ridge"] & sets["boosting"]
        coverage.append(
            {
                "date": date,
                "ridge_assets": len(sets["ridge"]),
                "boosting_assets": len(sets["boosting"]),
                "common_assets": len(common),
                "excluded_model_support": len(sets["ridge"] | sets["boosting"]) - len(common),
            }
        )
        section = day.loc[day.asset.isin(common)].copy()
        if not section.empty:
            check = section.groupby("asset")
            for field in (
                "target",
                "label_start",
                "label_end",
                "label_available_at",
                "target_valid",
                "return_unit",
                "return_convention",
                "industry",
                "size",
                "momentum_20",
                "volatility_20",
                "illiquidity_20",
            ):
                if check[field].nunique(dropna=False).gt(1).any():
                    raise ValueError(f"Models disagree on matched outcome metadata: {field}")
            # Recompute date-local neutralization on the identical model support.
            for _, index in section.groupby("model").groups.items():
                part = section.loc[index]
                section.loc[index, "neutralized_score"] = residualize(
                    part.score.to_numpy(),
                    part,
                    ("size", "momentum_20", "volatility_20", "illiquidity_20"),
                )
            selected.append(section)
    if not selected:
        raise ValueError("No common OOS prediction support")
    return pd.concat(selected), pd.DataFrame(coverage)


def calendar_block_ci(values: pd.Series, repetitions: int, block: int, seed: int) -> tuple:
    """Draw full date blocks including missing dates; report their coverage separately."""
    if repetitions < 100 or block < 1:
        raise ValueError("Insufficient bootstrap repetitions or invalid date block")
    sample = values.to_numpy(float)
    if np.isfinite(sample).sum() < max(10, block) or len(sample) < block:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(sample) - block + 1, int(np.ceil(len(sample) / block)))
        draw = np.concatenate([sample[s : s + block] for s in starts])[: len(sample)]
        if np.isfinite(draw).any():
            means.append(np.nanmean(draw))
    return tuple(np.quantile(means, [0.025, 0.975]))


def compare_incremental(archive: pd.DataFrame, cfg: dict) -> dict:
    matched, coverage = matched_predictions(archive)
    expected_dates = pd.DatetimeIndex(archive.attrs.get("oos_calendar", coverage.date))
    coverage = coverage.set_index("date").reindex(expected_dates)
    coverage["no_prediction_date"] = coverage.common_assets.isna()
    for column in ("ridge_assets", "boosting_assets", "common_assets", "excluded_model_support"):
        coverage[column] = coverage[column].fillna(0).astype(int)
    coverage.index.name = "date"
    coverage = coverage.reset_index()
    daily = []
    for date, section in matched.groupby("date", sort=True):
        for column in ("score", "neutralized_score"):
            valid_assets = section.loc[section.target.notna()].asset.unique()
            valid = section.loc[section.asset.isin(valid_assets)]
            for model, group in valid.groupby("model"):
                daily.append(
                    {
                        "date": date,
                        "model": model,
                        "prediction": column,
                        "ic": correlation(group[column], group.target, False),
                        "rank_ic": correlation(group[column], group.target),
                        "assets": len(group),
                    }
                )
    daily = pd.DataFrame(daily)
    estimates, differences = [], []
    calendar = pd.DatetimeIndex(coverage.date)
    for column in ("score", "neutralized_score"):
        matrix = daily.loc[daily.prediction.eq(column)].pivot(index="date", columns="model")
        for metric in ("ic", "rank_ic"):
            diff = (matrix[metric]["boosting"] - matrix[metric]["ridge"]).reindex(calendar)
            low, high = calendar_block_ci(
                diff,
                cfg["statistics"]["bootstrap_replications"],
                cfg["statistics"]["block_days"],
                cfg["seed"],
            )
            estimates.append(
                {
                    "prediction": column,
                    "metric": metric,
                    "comparison": "boosting_minus_ridge",
                    **calendar_hac(diff, cfg["statistics"]["hac_lags"]),
                    "block_ci_low": low,
                    "block_ci_high": high,
                }
            )
            differences.append(
                pd.DataFrame(
                    {
                        "date": calendar,
                        "prediction": column,
                        "metric": metric,
                        "difference": diff.to_numpy(),
                    }
                )
            )
    summary = pd.DataFrame(estimates)
    # Four fixed descriptive comparisons; primary is raw RankIC, all retained.
    summary["holm_p"] = multipletests(summary.hac_p.fillna(1), method="holm")[1]
    summary.loc[summary.hac_p.isna(), "holm_p"] = np.nan
    return {
        "oos_model_daily_ic": daily,
        "oos_model_comparison": summary,
        "oos_paired_daily_difference": pd.concat(differences),
        "oos_model_coverage": coverage,
    }
