"""Generate real-data research figures, aggregate tables and standalone reports."""

import json

import pandas as pd

from market_microstructure.optimal_execution.experiments import optimal_experiments
from market_microstructure.reporting.execution_plots import execution_figures
from market_microstructure.reporting.optimal_plots import optimal_figures
from market_microstructure.reporting.report import write_report
from market_microstructure.reporting.signal_plots import book_figures, signal_figures
from market_microstructure.reporting.statistics import feature_ablations, oos_metrics
from market_microstructure.utils.config import script_config
from market_microstructure.utils.logging import configure_logging
from market_microstructure.utils.paths import ensure_directories, research_path


def main() -> None:
    """Fail clearly on missing real data rather than generating synthetic substitute results."""
    config = script_config(__doc__)
    if config["data"].get("provider") != "public_szse_reconstructed":
        raise ValueError("This empirical report requires the audited real SZSE source")
    ensure_directories()
    logger = configure_logging(config["logging"]["level"])
    features = pd.read_parquet(research_path(config, config["data"]["features_path"]))
    predictions = pd.read_parquet(research_path(config, "data/processed/predictions.parquet"))
    aggregate = oos_metrics(predictions)
    aggregate.to_csv(research_path(config, "outputs/tables/model_oos_summary.csv"), index=False)
    logger.info("Running fixed feature ablations on the same purged folds")
    ablations = feature_ablations(features, predictions, config)
    ablations.to_csv(research_path(config, "outputs/tables/feature_ablation_summary.csv"), index=False)
    folds = pd.read_csv(research_path(config, "outputs/tables/fold_audit.csv"))
    cutoff = pd.to_datetime(folds.loc[folds.fold == 0, "train_last_timestamp"], utc=True).min()
    optimal, trajectories, audit = optimal_experiments(features, cutoff, config)
    optimal.to_csv(research_path(config, "outputs/tables/optimal_execution_summary.csv"), index=False)
    trajectories.to_csv(research_path(config, "outputs/tables/optimal_execution_trajectories.csv"), index=False)
    research_path(config, "outputs/tables/optimal_execution_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    logger.info("Rendering figures from saved OOS signals and execution ledgers")
    figures = book_figures(features) + signal_figures(predictions, config["model"]["type"], config["diagnostics"]["rolling_window"])
    figures += execution_figures(config["model"]["type"], config["diagnostics"]["rolling_window"], config["labels"]["horizon_events"] or 10)
    figures += optimal_figures(optimal, trajectories)
    write_report(config, features, predictions, figures, aggregate, ablations)
    logger.info("Report completed: %d figures; outputs/research_report.html", len(figures))


if __name__ == "__main__":
    main()
