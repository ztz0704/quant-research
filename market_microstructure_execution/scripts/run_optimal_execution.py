"""Run all optimal-execution schedules with training-only real A-share risk inputs."""

import json

import pandas as pd

from market_microstructure.optimal_execution.experiments import optimal_experiments
from market_microstructure.utils.config import script_config
from market_microstructure.utils.logging import configure_logging
from market_microstructure.utils.paths import research_path


def main() -> None:
    """Write analytic and rounded schedule diagnostics and a calibration audit."""
    config = script_config(__doc__)
    logger = configure_logging(config["logging"]["level"])
    features = pd.read_parquet(research_path(config, config["data"]["features_path"]))
    folds = pd.read_csv(research_path(config, "outputs/tables/fold_audit.csv"))
    cutoff = pd.to_datetime(folds.loc[folds.fold == 0, "train_last_timestamp"], utc=True).min()
    summary, paths, audit = optimal_experiments(features, cutoff, config)
    summary.to_csv(research_path(config, "outputs/tables/optimal_execution_summary.csv"), index=False)
    paths.to_csv(research_path(config, "outputs/tables/optimal_execution_trajectories.csv"), index=False)
    research_path(config, "outputs/tables/optimal_execution_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    logger.info("Optimal execution completed: %d schedule/parameter cases", len(summary))


if __name__ == "__main__":
    main()
