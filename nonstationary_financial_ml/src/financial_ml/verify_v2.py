"""Reconstruct real/synthetic snapshots, replay models and perturb future prices."""

import json

import joblib
import numpy as np
import pandas as pd
from robust_alpha.data.research_contract import (
    add_calendar_labels,
    add_decision_eligibility,
    digest_file,
    normalize_dynamic_panel,
)
from robust_alpha.evaluation.conditional_reversal import build_pressure_features
from threadpoolctl import threadpool_limits

from financial_ml.research_v2 import train_incremental
from financial_ml.upgrade_pipeline import load_v2_artifacts
from financial_ml.utils import path, write_json


@threadpool_limits.wrap(limits=1)
def verify_v2(cfg: dict) -> dict:
    archive = load_v2_artifacts(cfg)
    destination = path(cfg["output"]["directory"])
    raw = pd.read_parquet(destination / "source_snapshot.parquet")
    calendar = pd.read_parquet(destination / "calendar_snapshot.parquet")

    def rebuild(source):
        frame = add_decision_eligibility(
            normalize_dynamic_panel(source, calendar),
            cfg["data"]["min_history"],
            allow_unknown_status=True,
        )
        frame, names = build_pressure_features(
            frame, cfg["data"]["min_assets"], cfg["data"]["min_amount_20"]
        )
        return add_calendar_labels(frame, calendar), names

    panel, features = rebuild(raw)
    pd.testing.assert_frame_equal(panel, pd.read_parquet(destination / "research_panel.parquet"))
    replayed = 0
    for (fold, model), group in archive.groupby(["oos_fold", "model"]):
        saved = joblib.load(destination / f"models/fold_{fold}_{model}.joblib")
        scores = saved["estimator"].predict(group[features].to_numpy())
        np.testing.assert_array_equal(scores, group.score)
        np.testing.assert_array_equal(saved["calibration"].predict(scores), group.expected_return)
        replayed += len(group)
    first = archive.loc[archive.oos_fold.eq(archive.oos_fold.min())]
    cutoff = sorted(first.date.unique())[first.date.nunique() // 2]
    mutated = raw.copy()
    future = mutated.date.gt(cutoff)
    mutated.loc[future, ["raw_open", "raw_high", "raw_low", "raw_close"]] *= 5
    mutated.loc[future, ["amount", "volume"]] *= 1000
    changed, _ = rebuild(mutated)
    historical = panel.date.le(cutoff)
    pd.testing.assert_frame_equal(
        panel.loc[historical, features + ["eligible"]],
        changed.loc[historical, features + ["eligible"]],
    )
    # Refit exactly the first fold; price changes alter future labels including
    # labels whose decision is before the perturbation, testing real maturity.
    stop = first.date.max()
    changed = changed.loc[changed.date.le(stop)].copy()
    changed.loc[changed.date.gt(cutoff), "target"] = 0.8
    changed.loc[changed.date.gt(cutoff), "target_valid"] = False
    revised, _, _, _ = train_incremental(changed, features, cfg)
    columns = [
        "date",
        "asset",
        "model",
        "score",
        "expected_return",
        "neutralized_score",
        "model_version",
    ]
    expected = first.loc[first.date.le(cutoff), columns].reset_index(drop=True)
    actual = revised.loc[revised.date.le(cutoff), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual)
    result = {
        "evidence_grade": cfg["phase"],
        "snapshot_reconstruction": True,
        "model_rows_replayed": replayed,
        "future_price_label_perturbation": True,
        "cutoff": str(cutoff),
        "historical_rows_checked": len(expected),
        "formal_pit_empirical": False,
    }
    target = destination / "verification.json"
    write_json(result, target)
    manifest_path = destination / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][str(target)] = digest_file(target)
    write_json(manifest, manifest_path)
    return result
