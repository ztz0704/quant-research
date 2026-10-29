"""Compare OOS model signals under a mid upper bound and quote-side execution."""

import json

import pandas as pd

from market_microstructure.backtest.engine import BacktestConfig, run_backtest
from market_microstructure.backtest.metrics import performance_metrics
from market_microstructure.diagnostics.execution import maker_markouts
from market_microstructure.diagnostics.regime import regime_analysis
from market_microstructure.diagnostics.robustness import robustness_suite
from market_microstructure.utils.config import script_config
from market_microstructure.utils.logging import configure_logging
from market_microstructure.utils.paths import research_path


def main() -> None:
    """Save complete ledgers and economic/statistical comparison tables."""
    config = script_config(__doc__)
    logger = configure_logging(config["logging"]["level"])
    features = pd.read_parquet(research_path(config, config["data"]["features_path"]))
    predictions = pd.read_parquet(research_path(config, "data/processed/predictions.parquet"))
    summaries = []
    regime_parts = []
    for model, signal in predictions.groupby("model"):
        book = features.merge(signal[["timestamp", "asset"]], on=["timestamp", "asset"], validate="one_to_one")
        for mode in ("mid", "taker", "maker"):
            result = run_backtest(book, signal, BacktestConfig.from_config(config, mode))
            prefix = f"{model}_{mode}"
            for name, table in (("events", result.events), ("fills", result.fills),
                                ("orders", result.orders), ("closed_lots", result.closed_lots)):
                table.to_parquet(research_path(config, f"outputs/backtests/{prefix}_{name}.parquet"), index=False)
            metrics = performance_metrics(result)
            summaries.append({"model": model, "mode": mode,
                              "assumption": "UNREALISTIC UPPER BOUND" if mode == "mid" else "simulated quote execution",
                              **metrics})
            if model == config["model"]["type"]:
                cutoff = pd.read_csv(research_path(config, "outputs/tables/fold_audit.csv"))
                history_end = pd.to_datetime(cutoff.loc[cutoff.fold == 0, "train_last_timestamp"], utc=True).min()
                historical = features.loc[features.timestamp <= history_end]
                regime = regime_analysis(historical, book, signal, result)
                regime["mode"] = mode
                regime_parts.append(regime)
                if mode == "maker":
                    maker_markouts(book, result, config["labels"]["horizon_events"] or 10).to_csv(
                        research_path(config, "outputs/tables/maker_markouts.csv"), index=False)
            logger.info("%s: net PnL %.4f, fills %d", prefix, metrics["net_pnl"], metrics["trade_count"])
    summary = pd.DataFrame(summaries)
    selected = predictions.loc[predictions.model == config["model"]["type"]]
    selected_book = features.merge(selected[["timestamp", "asset"]], on=["timestamp", "asset"], validate="one_to_one")
    passive = selected.copy()
    passive["signal"] = 1.0
    baseline = run_backtest(selected_book, passive, BacktestConfig.from_config(config, "taker"))
    for name, table in (("events", baseline.events), ("fills", baseline.fills)):
        table.to_parquet(research_path(config, f"outputs/backtests/always_long_taker_{name}.parquet"), index=False)
    summary = pd.concat([summary, pd.DataFrame([{"model": "always_long", "mode": "taker",
        "assumption": "constant long benchmark, identical capacity/cost/T+1 rules",
        **performance_metrics(baseline)}])], ignore_index=True)
    summary.to_csv(research_path(config, "outputs/tables/execution_summary.csv"), index=False)
    pd.concat(regime_parts, ignore_index=True).to_csv(research_path(config, "outputs/tables/regime_summary.csv"), index=False)
    robustness_suite(selected_book, selected, config).to_csv(research_path(config, "outputs/tables/robustness_summary.csv"), index=False)
    summary[["model", "mode", "gross_pnl", "realized_gross_alpha", "inventory_mark_to_market",
             "spread_cost", "total_fees", "total_slippage", "market_impact", "net_pnl",
             "net_pnl_reconciliation_error"]].to_csv(research_path(config, "outputs/tables/pnl_decomposition.csv"), index=False)
    research_path(config, "outputs/backtests/config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
