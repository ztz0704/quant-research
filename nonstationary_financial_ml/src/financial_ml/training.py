"""Nested past validation, model persistence and honest OOS prediction."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from financial_ml.metrics import daily_ic
from financial_ml.models import make_model
from financial_ml.preprocessing import residualize
from financial_ml.temporal import partitions, walk_forward
from financial_ml.utils import logger, output, path, sha256, write_json


@threadpool_limits.wrap(limits=1)
def select_model(name: str, train: pd.DataFrame, validation: pd.DataFrame, features: list[str], cfg: dict):
    """Validation-only tuning; test arrays are deliberately absent from this API."""
    candidates = cfg["models"].get("grids", {}).get(name, [{}])
    records = []
    best = None
    best_metric = -np.inf
    for candidate, params in enumerate(candidates):
        model = make_model(name, params, cfg["seed"])
        started = time.perf_counter()
        model.fit(train[features].to_numpy(), train.target.to_numpy())
        validation_predictions = validation[["date", "target"]].copy()
        validation_predictions["prediction"] = model.predict(validation[features].to_numpy())
        metric = daily_ic(validation_predictions).rank_ic.mean()
        comparable = float(metric) if np.isfinite(metric) else -1.0
        records.append({"model": name, "candidate": candidate, "parameters": json.dumps(params, sort_keys=True),
                        "validation_rank_ic": metric, "training_seconds": time.perf_counter() - started})
        if best is None or comparable > best_metric:
            best, best_metric = params, comparable
    model = make_model(name, best, cfg["seed"])
    fit = pd.concat([train, validation]).sort_values(["date", "asset"])
    started = time.perf_counter()
    model.fit(fit[features].to_numpy(), fit.target.to_numpy())
    return model, best, records, time.perf_counter() - started


@threadpool_limits.wrap(limits=1)
def train_all(panel: pd.DataFrame, features: list[str], cfg: dict) -> pd.DataFrame:
    """Train each model on each fold and persist an input/config-bound manifest."""
    destination = output(cfg)
    log = logger(cfg, "train_models")
    predictions, tuning, fitting, boundaries = [], [], [], []
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    for fold in walk_forward(panel, cfg["validation"]):
        train, validation, test = partitions(panel, fold)
        if len(train) < cfg["validation"]["min_train_rows"] or len(validation) < 100 or test.empty:
            raise ValueError(f"Fold {fold.number} insufficient eligible history: {len(train)}, {len(validation)}, {len(test)}")
        boundaries.append({**asdict(fold), "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
                           "max_train_label_end": train.label_end.max(), "max_validation_label_end": validation.label_end.max()})
        for name in cfg["models"]["enabled"]:
            model, params, candidates, seconds = select_model(name, train, validation, features, cfg)
            for row in candidates:
                row["fold"] = fold.number
            tuning.extend(candidates)
            fitted_until = max(train.date.max(), validation.date.max())
            fitting.append({"fold": fold.number, "model": name, "parameters": json.dumps(params, sort_keys=True),
                            "training_rows": len(train) + len(validation), "training_seconds": seconds,
                            "fitted_until": fitted_until, "max_fit_label_end": max(train.label_end.max(), validation.label_end.max())})
            identity = [key for key in ["date", "asset", "target", "label_end", "target_valid", "industry", "size", "momentum", "volatility", "liquidity"] if key in test]
            result = test[identity].copy()
            result["raw_prediction"] = model.predict(test[features].to_numpy())
            result["neutralized_prediction"] = np.nan
            styles = tuple(style for style in ("size", "momentum", "volatility") if style in test)
            for _, index in result.groupby("date", sort=True).groups.items():
                result.loc[index, "neutralized_prediction"] = residualize(result.loc[index, "raw_prediction"].to_numpy(), result.loc[index], styles)
            result["fold"], result["model"] = fold.number, name
            result["trained_until"] = fitted_until
            result["model_age_sessions"] = dates.get_indexer(result.date) - dates.get_indexer([fitted_until])[0]
            predictions.append(result)
            joblib.dump({"estimator": model, "features": features, "parameters": params, "trained_until": fitted_until,
                         "max_fit_label_end": max(train.label_end.max(), validation.label_end.max())}, destination / f"models/fold_{fold.number}_{name}.joblib")
            log.info("fold=%s model=%s train=%s validation=%s test=%s fit_seconds=%.3f parameters=%s", fold.number, name, len(train), len(validation), len(test), seconds, params)
    joined = pd.concat(predictions, ignore_index=True).sort_values(["model", "date", "asset"])
    if joined.duplicated(["model", "date", "asset"]).any():
        raise AssertionError("Duplicate OOS prediction")
    joined.to_parquet(destination / "predictions/oos_predictions.parquet", index=False)
    pd.DataFrame(tuning).to_csv(destination / "tables/tuning_results.csv", index=False)
    pd.DataFrame(fitting).to_csv(destination / "tables/model_fits.csv", index=False)
    pd.DataFrame(boundaries).to_csv(destination / "tables/fold_boundaries.csv", index=False)
    write_json({"config": cfg, "features_sha256": sha256(path(cfg["data"]["features_path"])),
                "predictions_sha256": sha256(destination / "predictions/oos_predictions.parquet"),
                "feature_names": features, "models": cfg["models"]["enabled"], "folds": len(boundaries)}, destination / "tables/training_manifest.json")
    return joined
