"""Run held-out importance, regimes, perturbations and sensitivity; never select a test winner."""
import argparse

from financial_ml.importance import study as importance_study
from financial_ml.regimes import study as regime_study
from financial_ml.robustness import (
    future_perturbation,
    paired_comparison,
    parameter_and_seed_study,
    perturbations,
    synthetic_shift_study,
)
from financial_ml.utils import config, logger, output, sha256, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = config(args.config)
    log = logger(cfg, "diagnostics")
    for stage in (importance_study, regime_study, perturbations, parameter_and_seed_study,
                  paired_comparison, future_perturbation, synthetic_shift_study):
        log.info("Starting %s.%s", stage.__module__, stage.__name__)
        stage(cfg)
    destination = output(cfg)
    names = ["feature_importance_summary", "model_coefficients", "importance_stability", "regime_summary",
             "robustness_summary", "hyperparameter_sensitivity", "seed_sensitivity", "paired_model_comparison", "no_lookahead_checks"]
    write_json({"config": cfg, "training_manifest_sha256": sha256(destination / "tables/training_manifest.json"),
                "table_sha256": {name: sha256(destination / f"tables/{name}.csv") for name in names}}, destination / "tables/diagnostics_manifest.json")
    log.info("All declared diagnostics completed")
