"""Train fixed baseline models and save purged out-of-sample signal artifacts."""

import json

import joblib
import pandas as pd

from market_microstructure.diagnostics.signal_decay import (
    decay_half_life,
    rolling_signal_ic,
    signal_decay,
)
from market_microstructure.models.evaluation import (
    block_bootstrap_ic,
    correlation,
    quantile_analysis,
)
from market_microstructure.models.research import run_research
from market_microstructure.utils.config import script_config
from market_microstructure.utils.logging import configure_logging
from market_microstructure.utils.paths import research_path


def main() -> None:
    """Training output includes both statistical diagnostics and provenance audits."""
    config = script_config(__doc__)
    logger = configure_logging(config["logging"]["level"])
    frame = pd.read_parquet(research_path(config, config["data"]["features_path"]))
    research = run_research(frame, config)
    research.predictions.to_parquet(research_path(config, "data/processed/predictions.parquet"), index=False)
    for name, table in (("model_summary", research.metrics), ("fold_audit", research.audit),
                        ("feature_importance", research.importance)):
        table.to_csv(research_path(config, f"outputs/tables/{name}.csv"), index=False)
    rows = []
    for name, predicted in research.predictions.groupby("model"):
        intervals = block_bootstrap_ic(predicted, config["diagnostics"]["bootstrap_block_events"],
                                      config["diagnostics"]["bootstrap_samples"], config["seed"])
        rows.append({"model": name, "ic": correlation(predicted.signal, predicted.future_return_bps),
                     "rank_ic": correlation(predicted.signal, predicted.future_return_bps, True), **intervals})
        quantile_analysis(predicted, config["diagnostics"]["quantiles"]).to_csv(
            research_path(config, f"outputs/tables/quantiles_{name}.csv"), index=False)
    pd.DataFrame(rows).to_csv(research_path(config, "outputs/tables/signal_summary.csv"), index=False)
    selected = research.predictions.loc[research.predictions.model == config["model"]["type"]]
    if selected.empty:
        raise ValueError("Selected model must appear in model.compare")
    decay = signal_decay(frame, selected, config["diagnostics"]["horizons"])
    decay.to_csv(research_path(config, "outputs/tables/signal_decay.csv"), index=False)
    decay_half_life(decay).to_csv(research_path(config, "outputs/tables/signal_half_life.csv"), index=False)
    rolling_signal_ic(selected, config["diagnostics"]["rolling_window"]).to_csv(
        research_path(config, "outputs/tables/rolling_ic.csv"), index=False)
    for name, model in research.models.items():
        joblib.dump({"model": model, "columns": research.columns,
                     "purpose": "Final fold model; do not backfill earlier OOS predictions"},
                    research_path(config, f"outputs/models/{name}.joblib"))
    research_path(config, "outputs/models/feature_columns.json").write_text(json.dumps(research.columns), encoding="utf-8")
    logger.info("Research completed: %d OOS rows; %d fold/model evaluations",
                len(research.predictions), len(research.metrics))


if __name__ == "__main__":
    main()
