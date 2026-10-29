import copy
import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from test_research_v2 import fitted  # noqa: F401

from financial_ml.diagnostics_v3 import model_diagnostics
from financial_ml.research_v3_pipeline import run_v3
from financial_ml.shared_v3 import bind_alpha
from financial_ml.utils import config
from financial_ml.verify_v3 import load_checked_v3, verify_v3


@pytest.fixture(scope="module")
def integrated(tmp_path_factory):
    cfg = config("configs/research_v3_engineering.yaml")
    cfg["synthetic"].update(days=160, assets=30)
    cfg["data"].update(min_history=1, min_assets=5)
    cfg["validation"].update(
        train_days=60, validation_days=32, test_days=12, step_days=12, min_train_rows=100
    )
    cfg["models"]["grids"] = {
        "ridge": [{"alpha": 1.0}],
        "boosting": [{"max_iter": 10, "max_leaf_nodes": 7, "min_samples_leaf": 10}],
    }
    cfg["open_risk"].update(lookback=30, min_observations=15, min_specific_observations=10)
    cfg["portfolio"]["cost_bps_per_full_l1"] = [10]
    cfg["output"]["directory"] = str(tmp_path_factory.mktemp("integration_v3") / "research_v3")
    result = run_v3(cfg, diagnostics=False)
    return cfg, result


def test_cross_project_subprocess_schema_accounting_and_three_fold_verification(integrated):
    cfg, destination = integrated
    manifest = json.loads((destination / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["formal_empirical"] is False
    assert manifest["holdout_run"] is False
    contract = json.loads((destination / "risk_inputs/contract.json").read_text(encoding="utf-8"))
    assert contract["interface_version"] == "ashare-open-events-v3"
    assert len(contract["inputs"]) == 3
    signals = pd.read_parquet(destination / "risk_inputs/signals.parquet")
    assert "target" not in signals and "exit_sellable" not in signals
    performance = pd.read_parquet(destination / "portfolio/portfolio_performance.parquet")
    assert performance.return_coverage.eq(1).all()
    assert (
        performance.loc[
            performance.strategy.isin(["D_HGB_risk", "E_HGB_risk_cost"]), "optimized_decisions"
        ]
        .gt(0)
        .all()
    )
    verified = verify_v3(cfg)
    assert len(verified["fold_asset_future_checks"]) == 3
    assert all(
        len(check["asset_boundaries"]) == 2 for check in verified["fold_asset_future_checks"]
    )
    assert verified["all_strategy_calendars_preserved"]
    assert verified["all_model_rows_replayed"] == len(
        pd.read_parquet(destination / "oos_predictions.parquet")
    )


def test_stale_artifact_rejection_preserves_provenance(integrated):
    cfg, destination = integrated
    target = destination / "calibration_stability_fake.csv"
    manifest_file = destination / "experiment_manifest.json"
    original = manifest_file.read_text(encoding="utf-8")
    manifest = json.loads(original)
    manifest["artifacts"][target.name] = "wrong"
    target.write_text("unexpected_change", encoding="utf-8")
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="Stale V3 artifact"):
            load_checked_v3(cfg)
    finally:
        manifest_file.write_text(original, encoding="utf-8")


def test_wrong_checkout_binding_and_real_synthetic_gate(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "robust_alpha", SimpleNamespace(__file__="wrong_checkout/__init__.py")
    )
    with pytest.raises(ImportError, match="Different Alpha checkout"):
        bind_alpha()
    cfg = copy.deepcopy(config("configs/research_v3_engineering.yaml"))
    cfg["phase"] = "historical_integration"
    with pytest.raises(ValueError, match="separation"):
        run_v3(cfg)


def test_holdout_and_protocol_rejected_before_data_read():
    cfg = config("configs/research_v3_engineering.yaml")
    cfg["phase"] = "holdout"
    with pytest.raises(ValueError, match="holdout"):
        run_v3(cfg)
    cfg["phase"] = "engineering_validation"
    cfg["protocol"]["sha256"] = "stale"
    with pytest.raises(ValueError, match="hash changed"):
        run_v3(cfg)


def test_size_classification_uses_decision_support_not_future_label_validity(request):
    panel, features, cfg, archive, fits, models = request.getfixturevalue("fitted")
    settings = copy.deepcopy(cfg)
    settings.update(
        statistics={"bootstrap_replications": 100, "block_days": 3, "hac_lags": 2},
        diagnostics={"permutation_seed": 947},
    )
    before = model_diagnostics(archive, panel, fits, models, features, settings)
    changed = archive.copy()
    mask = changed.asset.eq(sorted(changed.asset.unique())[-1])
    changed.loc[mask, "target_valid"] = False
    changed.loc[mask, "target"] = float("nan")
    after = model_diagnostics(changed, panel, fits, models, features, settings)
    pd.testing.assert_frame_equal(
        before["model_size_group_classifications"], after["model_size_group_classifications"]
    )
