"""OOS aggregate evaluation and fixed, causal feature ablations."""

from copy import deepcopy

import numpy as np
import pandas as pd

from market_microstructure.features.feature_pipeline import feature_columns
from market_microstructure.models.evaluation import (
    classification_metrics,
    correlation,
    regression_metrics,
)
from market_microstructure.models.research import run_research


def oos_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate underlying OOS observations, rather than averaging unequal fold scores."""
    rows = []
    for model, group in predictions.groupby("model"):
        if model == "linear":
            metrics = regression_metrics(group.future_return_bps.to_numpy(), group.prediction_bps.to_numpy())
        else:
            columns = [column for column in group if column.startswith("prob_") and group[column].notna().any()]
            columns.sort(key=lambda x: int(x.removeprefix("prob_")))
            classes = np.array([int(column.removeprefix("prob_")) for column in columns])
            metrics = classification_metrics(group.direction.to_numpy(), group.class_prediction.to_numpy(),
                                             group[columns].fillna(0).to_numpy(), classes)
        rows.append({"model": model, "rows": len(group),
                     "labeled_rows": int(group.future_return_bps.notna().sum()),
                     "signal_ic": correlation(group.signal, group.future_return_bps), **metrics})
    return pd.DataFrame(rows)


def feature_ablations(frame: pd.DataFrame, predictions: pd.DataFrame,
                      config: dict) -> pd.DataFrame:
    """Same folds and hyperparameters; predeclared sets, no OOS-driven feature selection."""
    settings = deepcopy(config)
    settings["model"]["compare"] = ["logistic"]
    rows = []
    for name, columns in config["diagnostics"]["ablation_sets"].items():
        research = run_research(frame, settings, feature_subset=columns)
        metrics = oos_metrics(research.predictions).iloc[0].to_dict()
        rows.append({"feature_set": name, "feature_count": len(columns), **metrics})
    full = oos_metrics(predictions.loc[predictions.model == "logistic"])
    if len(full):
        rows.append({"feature_set": "full_features", "feature_count": len(feature_columns(frame)), **full.iloc[0].to_dict()})
    return pd.DataFrame(rows)
