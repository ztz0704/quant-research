"""Audit the empirical artifact chain, saved model predictions and temporal boundaries."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from financial_ml.data import load
from financial_ml.evaluation import load_research
from financial_ml.features import build_features
from financial_ml.labels import add_labels
from financial_ml.temporal import partitions, walk_forward
from financial_ml.utils import ROOT, config, output, path, sha256, write_json


@threadpool_limits.wrap(limits=1)
def verify(cfg: dict, notebooks_required: bool) -> None:
    panel, predictions, features = load_research(cfg)
    destination = output(cfg)
    rebuilt, rebuilt_names = build_features(load(path(cfg["data"]["input_path"])), cfg)
    rebuilt = add_labels(rebuilt, cfg["labels"]["forward_horizon"])
    assert rebuilt_names == features
    pd.testing.assert_frame_equal(panel, rebuilt, check_exact=True)
    model_errors = []
    for fold in walk_forward(panel, cfg["validation"]):
        train, validation, test = partitions(panel, fold)
        assert train.label_end.max() < fold.validation_start
        assert validation.label_end.max() < fold.test_start
        for name in cfg["models"]["enabled"]:
            saved = joblib.load(destination / f"models/fold_{fold.number}_{name}.joblib")
            assert saved["max_fit_label_end"] < fold.test_start
            actual = predictions.loc[predictions.fold.eq(fold.number) & predictions.model.eq(name)].set_index(["date", "asset"])
            ordered = test.set_index(["date", "asset"]).loc[actual.index]
            recomputed = saved["estimator"].predict(ordered[features].to_numpy())
            difference = float(np.max(np.abs(recomputed - actual.raw_prediction.to_numpy())))
            if not np.array_equal(recomputed, actual.raw_prediction.to_numpy()):
                raise AssertionError(f"Saved-model predictions differ: {fold.number}, {name}, {difference}")
            model_errors.append({"fold": fold.number, "model": name, "maximum_difference": difference})
    checks = pd.read_csv(destination / "tables/no_lookahead_checks.csv")
    assert checks.passed.all()
    manifest = json.loads((destination / "tables/reproducibility_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"] == cfg
    assert len(manifest["figure_sha256"]) >= 21
    for filename, digest in manifest["figure_sha256"].items():
        assert sha256(destination / "figures" / filename) == digest
    for filename, digest in manifest["code_sha256"].items():
        assert sha256(ROOT / filename) == digest
    notebook_status = "not_requested"
    if notebooks_required:
        executed = json.loads((destination / "tables/notebook_execution.json").read_text(encoding="utf-8"))
        assert len(executed["notebooks"]) == 4
        for item in executed["notebooks"]:
            assert item["status"] == "passed"
            assert item["sha256"] == sha256(ROOT / "notebooks" / item["notebook"])
        notebook_status = "four_executed"
    write_json({"status": "passed", "models_checked": len(model_errors), "model_prediction_checks": model_errors,
                "future_perturbation_models": len(checks), "figures": len(manifest["figure_sha256"]),
                "notebooks": notebook_status}, destination / "tables/artifact_verification.json")
    print(f"Verified {len(model_errors)} saved models, {len(checks)} future-perturbation checks and {len(manifest['figure_sha256'])} figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--require-notebooks", action="store_true")
    args = parser.parse_args()
    verify(config(args.config), args.require_notebooks)
