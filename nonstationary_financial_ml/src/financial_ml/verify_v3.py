"""Independent artifact checks, all model replay, three folds/two asset boundaries."""

import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from financial_ml.research_v2 import train_incremental, validate_archive
from financial_ml.research_v3_pipeline import build_panel, clock_at
from financial_ml.shared_v3 import PROJECTS, file_sha, implementation_hashes
from financial_ml.utils import path, write_json


def load_checked_v3(cfg: dict) -> tuple:
    destination = path(cfg["output"]["directory"])
    manifest = json.loads((destination / "experiment_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "complete"
        or manifest["config"] != cfg
        or manifest["implementation_hashes"] != implementation_hashes()
        or manifest["protocol_sha256"] != file_sha(PROJECTS / "Robust_Alpha_Discovery/configs/protocols/research_v2.txt")
    ):
        raise ValueError("Stale V3 experiment: code/config/protocol changed")
    for name, sha in manifest["artifacts"].items():
        if file_sha(destination / name) != sha:
            raise ValueError(f"Stale V3 artifact: {name}")
    for name, sha in manifest["original_source_files"].items():
        if file_sha(path(name)) != sha:
            raise ValueError(f"Changed original data source: {name}")
    archive = pd.read_parquet(destination / "oos_predictions.parquet")
    validate_archive(archive)
    return destination, manifest, archive


@threadpool_limits.wrap(limits=1)
def verify_v3(cfg: dict) -> dict:
    destination, manifest, archive = load_checked_v3(cfg)
    raw = pd.read_parquet(destination / "source_snapshot.parquet")
    calendar = pd.read_parquet(destination / "calendar_snapshot.parquet")
    panel, features = build_panel(raw, calendar, cfg)
    pd.testing.assert_frame_equal(panel, pd.read_parquet(destination / "research_panel.parquet"))
    replayed = 0
    for (fold, model), group in archive.groupby(["oos_fold", "model"]):
        saved = joblib.load(destination / f"models/fold_{fold}_{model}.joblib")
        scores = saved["estimator"].predict(group[features].to_numpy())
        np.testing.assert_array_equal(scores, group.score)
        np.testing.assert_array_equal(saved["calibration"].predict(scores), group.expected_return)
        replayed += len(group)
    folds = sorted(archive.oos_fold.unique())
    chosen = sorted({folds[0], folds[len(folds) // 2], folds[-1]})
    checked = []
    columns = [
        "date",
        "asset",
        "model",
        "score",
        "expected_return",
        "neutralized_score",
        "model_version",
    ]
    for fold in chosen:
        first = archive.loc[archive.oos_fold.eq(fold)]
        cutoff = sorted(first.date.unique())[first.date.nunique() // 2]
        assets = sorted(first.asset.unique())
        boundaries = [assets[0], assets[-1]]
        mutated = raw.copy()
        future = mutated.date.gt(cutoff) & mutated.asset.isin(boundaries)
        mutated.loc[future, ["raw_open", "raw_high", "raw_low", "raw_close"]] *= 5
        mutated.loc[future, ["amount", "volume"]] *= 1000
        mutated.loc[future, ["buyable", "sellable", "universe_member"]] = False
        changed, _ = build_panel(mutated, calendar, cfg)
        historical = panel.date.le(cutoff)
        pd.testing.assert_frame_equal(
            panel.loc[historical, features + ["eligible"]],
            changed.loc[historical, features + ["eligible"]],
        )
        # Also perturb all labels NOT yet mature at the cutoff, including labels
        # whose decision date precedes it. Fit/calibration must remain unchanged.
        immature = changed.label_available_at.gt(clock_at(cutoff, 18))
        changed.loc[immature, "target"] = 0.8
        revised, _, _, _ = train_incremental(changed, features, cfg, fold_numbers={int(fold)})
        expected = first.loc[first.date.le(cutoff), columns].reset_index(drop=True)
        actual = revised.loc[revised.date.le(cutoff), columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual)
        checked.append(
            {
                "fold": int(fold),
                "cutoff": str(cutoff),
                "asset_boundaries": boundaries,
                "historical_prediction_rows": len(expected),
                "future_perturbation_passed": True,
            }
        )
    events = pd.read_parquet(destination / "portfolio/portfolio_events.parquet")
    known = events.pre_nav.notna() & events.post_nav.notna()
    np.testing.assert_allclose(
        events.loc[known, "post_nav"],
        events.loc[known, "pre_nav"] - events.loc[known, "fees"],
        atol=1e-7,
    )
    if events.cash.lt(-1e-7).any():
        raise AssertionError("Negative event cash")
    periods = pd.read_parquet(destination / "portfolio/portfolio_periods.parquet")
    for _, group in periods.groupby(["strategy", "cost_bps_per_full_l1"]):
        if len(group) != manifest["oos_dates"] or group.date.duplicated().any():
            raise AssertionError("Portfolio silently removed an OOS calendar date")
        if group.net_return.notna().all():
            np.testing.assert_allclose((1 + group.net_return).cumprod(), group.nav, atol=1e-9)
    v2_preserved = None
    if cfg["phase"] == "historical_integration":
        v2 = destination.parents[1] / "research_v2/historical_integration"
        old_manifest = json.loads((v2 / "experiment_manifest.json").read_text(encoding="utf-8"))
        for name, expected in old_manifest["artifacts"].items():
            if file_sha(path(name)) != expected:
                raise AssertionError("Original V2 artifact was changed")
        old = pd.read_parquet(v2 / "oos_predictions.parquet")
        fields = [
            "date",
            "asset",
            "model",
            "score",
            "expected_return",
            "neutralized_score",
            "model_version",
        ]
        pd.testing.assert_frame_equal(
            old[fields].reset_index(drop=True), archive[fields].reset_index(drop=True)
        )
        v2_preserved = True
    output = {
        "status": "passed",
        "snapshot_reconstruction": True,
        "all_model_rows_replayed": replayed,
        "fold_asset_future_checks": checked,
        "known_event_nav_identities": int(known.sum()),
        "all_strategy_calendars_preserved": True,
        "formal_pit_empirical": False,
        "v2_historical_artifacts_and_predictions_preserved": v2_preserved,
    }
    target = destination / "verification.json"
    write_json(output, target)
    manifest["artifacts"]["verification.json"] = file_sha(target)
    write_json(manifest, destination / "experiment_manifest.json")
    return output
