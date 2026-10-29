"""Train/evaluate fold models and emit strictly out-of-sample predictions."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from market_microstructure.features.feature_pipeline import feature_columns
from market_microstructure.models.estimators import make_model
from market_microstructure.models.evaluation import classification_metrics, regression_metrics
from market_microstructure.models.validation import WalkForwardConfig, walk_forward_splits


@dataclass
class ResearchResult:
    """OOS artifacts, fold diagnostics and final fitted model per model name."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    audit: pd.DataFrame
    importance: pd.DataFrame
    models: dict[str, Pipeline]
    columns: list[str]


def _predictions(model: Pipeline, name: str, frame: pd.DataFrame, columns: list[str], config: dict) -> pd.DataFrame:
    """Transform model output into a comparable [-1,1] directional score."""
    output = frame[["timestamp", "asset", "future_return_bps", "direction", "label_end"]].copy()
    x = frame[columns]
    if name == "linear":
        output["prediction_bps"] = model.predict(x)
        scale = config["strategy"]["regression_scale_bps"]
        if scale <= 0:
            raise ValueError("Regression signal scale must be positive")
        output["signal"] = np.tanh(output.prediction_bps / scale)
    else:
        probabilities = model.predict_proba(x)
        classes = model.named_steps["model"].classes_
        direction_scores = np.where(classes == 0, -1, classes) if config["labels"]["classification"] == "binary" else classes
        output["signal"] = probabilities @ direction_scores
        output["class_prediction"] = model.predict(x)
        for j, label in enumerate(classes):
            output[f"prob_{int(label)}"] = probabilities[:, j]
    return output


def _metrics(model: Pipeline, name: str, frame: pd.DataFrame, predicted: pd.DataFrame) -> dict:
    """Score only available labels without dropping any economic prediction rows."""
    if name == "linear":
        return regression_metrics(frame.future_return_bps.to_numpy(), predicted.prediction_bps.to_numpy())
    classes = model.named_steps["model"].classes_
    probabilities = predicted[[f"prob_{int(c)}" for c in classes]].to_numpy()
    return classification_metrics(frame.direction.to_numpy(), predicted.class_prediction.to_numpy(),
                                  probabilities, classes)


def _linear_importance(model: Pipeline, train: pd.DataFrame, columns: list[str], lags: int) -> pd.DataFrame:
    """OLS t statistics use HAC covariance for overlapping labels, per asset.

    Per-asset fits avoid treating pooled cross-asset rows as independent observations.
    Joint coefficients can be unstable with collinear book features; report condition numbers.
    """
    rows = []
    for asset, group in train.groupby("asset"):
        transformed = model[:-1].transform(group[columns])
        design = sm.add_constant(transformed, has_constant="add")
        fitted = sm.OLS(group.future_return_bps.to_numpy(), design).fit(
            cov_type="HAC", cov_kwds={"maxlags": min(lags, len(group) - 1)},
        )
        for j, column in enumerate(columns, start=1):
            rows.append({"asset": asset, "feature": column, "coefficient_bps": fitted.params[j],
                         "t_stat_hac": fitted.tvalues[j], "condition_number": fitted.condition_number})
    return pd.DataFrame(rows)


def run_research(frame: pd.DataFrame, config: dict[str, Any],
                 feature_subset: list[str] | None = None) -> ResearchResult:
    """Fit only purged training samples; validation and test transforms remain frozen."""
    available = feature_columns(frame)
    columns = available if feature_subset is None else feature_subset
    if not set(columns).issubset(available) or len(columns) != len(set(columns)):
        raise ValueError("Feature subset must contain unique whitelisted causal features")
    if not columns:
        raise ValueError("No whitelisted features")
    prediction_parts, metric_rows, audits, importance_parts = [], [], [], []
    models = {}
    horizon = config["labels"]["horizon_events"]
    for fold in walk_forward_splits(frame, WalkForwardConfig(**config["validation"])):
        train, validation, test = (frame.iloc[idx] for idx in (fold.train, fold.validation, fold.test))
        if train.empty or validation.empty:
            raise ValueError("Purging leaves an empty training/validation fold")
        for name in config["model"]["compare"]:
            label = "future_return_bps" if name == "linear" else "direction"
            usable = train.loc[train[label].notna()]
            if name != "linear" and usable.direction.nunique() < 2:
                raise ValueError(f"Fold {fold.number}: {name} requires at least two observed classes")
            model = make_model(name, config["model"], config["seed"])
            model.fit(usable[columns], usable[label])
            models[name] = model
            for split, sample in (("validation", validation), ("test", test)):
                predicted = _predictions(model, name, sample, columns, config)
                metrics = _metrics(model, name, sample, predicted)
                metric_rows.append({"fold": fold.number, "model": name, "split": split,
                                    "rows": len(sample), "labeled_rows": int(sample[label].notna().sum()),
                                    **metrics})
                if split == "test":
                    predicted["fold"], predicted["model"] = fold.number, name
                    prediction_parts.append(predicted)
            audits.append({"fold": fold.number, "model": name, "train_rows": len(usable),
                           "train_last_timestamp": usable.timestamp.max(),
                           "train_max_label_end": usable.label_end.max(),
                           "validation_start": fold.validation_start,
                           "validation_max_label_end": validation.label_end.max(),
                           "test_start": fold.test_start, "test_end": fold.test_end})
            if name == "linear":
                # Time-horizon labels use a conservative observed maximum asset-event span.
                hac_lags = horizon or int(usable.groupby("asset").size().max() ** 0.5)
                importance = _linear_importance(model, usable, columns, hac_lags)
            elif name == "forest":
                scored = validation.loc[validation.direction.notna()]
                permutation = permutation_importance(model, scored[columns], scored.direction,
                    n_repeats=3, random_state=config["seed"], scoring="balanced_accuracy", n_jobs=1)
                importance = pd.DataFrame({"feature": columns, "asset": "POOLED",
                                          "permutation_mean": permutation.importances_mean,
                                          "permutation_std": permutation.importances_std})
            else:
                coefficients = model.named_steps["model"].coef_
                importance = pd.DataFrame({"feature": columns, "asset": "POOLED",
                                          "coefficient_bps": np.nan,
                                          "logistic_coefficient_abs": np.abs(coefficients).mean(axis=0)})
            importance["model"], importance["fold"] = name, fold.number
            importance_parts.append(importance)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if predictions.duplicated(["model", "timestamp", "asset"]).any():
        raise ValueError("Overlapping OOS prediction rows")
    return ResearchResult(predictions.sort_values(["model", "timestamp", "asset"]).reset_index(drop=True),
                          pd.DataFrame(metric_rows), pd.DataFrame(audits),
                          pd.concat(importance_parts, ignore_index=True), models, columns)
