import copy

import numpy as np
import pandas as pd
import pytest
from robust_alpha.data.research_contract import (
    add_calendar_labels,
    add_decision_eligibility,
    normalize_dynamic_panel,
)
from robust_alpha.data.research_fixture import engineering_sample
from robust_alpha.evaluation.conditional_reversal import build_pressure_features

from financial_ml.research_v2 import (
    compare_incremental,
    fit_calibration,
    matched_predictions,
    train_incremental,
    validate_archive,
)
from financial_ml.upgrade_pipeline import PROJECTS, load_v2_artifacts, run_v2


@pytest.fixture(scope="module")
def fitted():
    raw, calendar, _ = engineering_sample(days=130, assets=30)
    stop = sorted(raw.date.unique())[107]
    raw, calendar = raw.loc[raw.date.le(stop)], calendar.loc[calendar.date.le(stop)]
    panel = add_decision_eligibility(normalize_dynamic_panel(raw, calendar), 1)
    panel, features = build_pressure_features(panel, 5)
    panel = add_calendar_labels(panel, calendar)
    cfg = {
        "seed": 8,
        "validation": {
            "mode": "rolling",
            "train_days": 60,
            "validation_days": 32,
            "test_days": 12,
            "step_days": 12,
            "embargo_days": 2,
            "min_train_rows": 100,
        },
        "models": {
            "enabled": ["ridge", "boosting"],
            "grids": {
                "ridge": [{"alpha": 1.0}],
                "boosting": [{"max_iter": 10, "min_samples_leaf": 10, "max_leaf_nodes": 7}],
            },
        },
    }
    archive, fits, _, models = train_incremental(panel, features, cfg)
    return panel, features, cfg, archive, fits, models


def test_future_labels_do_not_change_historical_fit_predictions_or_calibration(fitted):
    panel, features, cfg, before, _, _ = fitted
    cutoff = before.date.min() + pd.Timedelta(days=5)
    changed = panel.copy()
    future = changed.date.gt(cutoff)
    changed.loc[future, "target"] = 0.8
    changed.loc[future, "target_valid"] = False
    changed.loc[future, features] *= 1000
    after, _, _, _ = train_incremental(changed, features, cfg)
    columns = [
        "date",
        "asset",
        "model",
        "score",
        "expected_return",
        "neutralized_score",
        "model_version",
        "model_fit_cutoff",
    ]
    pd.testing.assert_frame_equal(
        before.loc[before.date.le(cutoff), columns].reset_index(drop=True),
        after.loc[after.date.le(cutoff), columns].reset_index(drop=True),
    )


def test_archive_maturity_and_model_replay(fitted):
    _, features, _, archive, fits, models = fitted
    validate_archive(archive)
    assert pd.to_datetime(fits.max_fit_label_available).lt(fits.estimator_fit_cutoff).all()
    for (fold, name), group in archive.groupby(["oos_fold", "model"]):
        saved = models[f"fold_{fold}_{name}"]
        np.testing.assert_array_equal(
            saved["estimator"].predict(group[features].to_numpy()), group.score
        )
        np.testing.assert_array_equal(
            saved["calibration"].predict(group.score), group.expected_return
        )
    broken = archive.copy()
    broken["max_fit_label_available"] = broken.estimator_fit_cutoff
    with pytest.raises(ValueError, match="immature"):
        validate_archive(broken)


def test_calibration_hand_answer_and_maturity():
    dates = pd.bdate_range("2020-01-01", periods=12)
    sample = pd.DataFrame(
        {
            "date": dates,
            "score": np.arange(12.0),
            "target": 0.001 + 0.002 * np.arange(12.0),
            "label_available_at": dates + pd.Timedelta(hours=18),
        }
    )
    cutoff = dates[-1] + pd.Timedelta(days=2)
    mapping = fit_calibration(sample, cutoff)
    assert mapping.intercept == pytest.approx(0.001)
    assert mapping.slope == pytest.approx(0.002)
    with pytest.raises(ValueError, match="matured"):
        fit_calibration(sample, dates[-1])


def test_missing_forecast_support_matches_assets_without_target_selection(fitted):
    _, _, _, archive, _, _ = fitted
    changed = archive.copy()
    date = changed.date.min()
    asset = changed.loc[changed.date.eq(date), "asset"].iloc[0]
    changed = changed.loc[
        ~(changed.date.eq(date) & changed.asset.eq(asset) & changed.model.eq("boosting"))
    ]
    matched, coverage = matched_predictions(changed)
    assert coverage.loc[coverage.date.eq(date), "excluded_model_support"].iloc[0] == 1
    assert matched.loc[matched.date.eq(date) & matched.asset.eq(asset)].empty
    missing_outcome = changed.copy()
    missing_outcome.loc[missing_outcome.date.eq(date), "target"] = np.nan
    missing_outcome.loc[missing_outcome.date.eq(date), "target_valid"] = False
    after, _ = matched_predictions(missing_outcome)
    pd.testing.assert_frame_equal(
        matched[["date", "asset", "model", "score"]], after[["date", "asset", "model", "score"]]
    )


def test_seed_reproducibility(fitted):
    panel, features, cfg, before, _, _ = fitted
    after, _, _, _ = train_incremental(panel, features, copy.deepcopy(cfg))
    pd.testing.assert_frame_equal(before, after)


def test_full_pipeline_rejects_changed_config_predictions_and_data(fitted, tmp_path):
    from robust_alpha.data.research_contract import digest_file

    _, _, settings, _, _, _ = fitted
    cfg = copy.deepcopy(settings)
    cfg.update(
        phase="engineering_validation",
        protocol={
            "status": "frozen",
            "empirical_status": "draft",
            "sha256": digest_file(PROJECTS / "Robust_Alpha_Discovery/configs/protocols/research_v2.txt"),
            "holdout_confirmed_untouched": False,
        },
        data={"source": "synthetic", "min_history": 1, "min_assets": 5, "min_amount_20": 0},
        synthetic={"days": 130, "assets": 30, "seed": 8},
        statistics={"hac_lags": 5, "block_days": 5, "bootstrap_replications": 100},
        portfolio={"quantile": 0.2, "cost_bps_per_full_l1": [0, 10]},
        output={"directory": str(tmp_path / "v2")},
    )
    destination = run_v2(cfg)
    assert not load_v2_artifacts(cfg).empty
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["seed"] += 1
    with pytest.raises(ValueError, match="Stale"):
        load_v2_artifacts(bad_cfg)
    for filename in ("oos_predictions.parquet", "source_snapshot.parquet"):
        artifact = destination / filename
        saved = artifact.read_bytes()
        artifact.write_bytes(saved + b"changed")
        with pytest.raises(ValueError, match="Stale"):
            load_v2_artifacts(cfg)
        artifact.write_bytes(saved)


def test_formal_gate_runs_before_any_data_read(fitted, monkeypatch, tmp_path):
    from financial_ml import upgrade_pipeline

    cfg = copy.deepcopy(fitted[2])
    cfg.update(
        phase="holdout",
        protocol={"status": "draft", "sha256": "not_frozen", "holdout_confirmed_untouched": False},
        data={"source": "bundle"},
        output={"directory": str(tmp_path)},
    )

    def unexpected_read(*_args):
        raise AssertionError("Sealed data were accessed")

    monkeypatch.setattr(upgrade_pipeline, "read_bundle", unexpected_read)
    with pytest.raises(ValueError, match="frozen"):
        run_v2(cfg, include_holdout=True)


def test_missing_entire_oos_date_is_reported_without_compressing_calendar(fitted):
    archive = fitted[3].copy()
    lost = sorted(archive.date.unique())[3]
    archive = archive.loc[~archive.date.eq(lost)].copy()
    cfg = {"seed": 8, "statistics": {"hac_lags": 2, "bootstrap_replications": 100, "block_days": 3}}
    result = compare_incremental(archive, cfg)
    row = result["oos_model_coverage"].set_index("date").loc[lost]
    assert row.no_prediction_date
    assert row.common_assets == 0
    # The fixture also has two terminal forecasts whose t+2 opens are outside
    # its observed calendar. All three missing dates must remain in inference.
    assert result["oos_model_comparison"].missing_dates.eq(3).all()
    assert result["oos_model_comparison"].n_dates.eq(9).all()
