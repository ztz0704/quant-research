"""Run isolated empirical validation without overwriting the synthetic research artifacts."""

import argparse

from risk_aware_portfolio.data.loader import load_panel
from risk_aware_portfolio.diagnostics.robustness import local_weight_sensitivity, run_robustness
from risk_aware_portfolio.reporting.figures import generate_figures
from risk_aware_portfolio.reporting.real_notebook import write_real_notebook
from risk_aware_portfolio.reporting.real_validation import (
    real_no_lookahead_checks,
    run_primary_comparisons,
    verify_real_data,
    write_real_report,
)
from risk_aware_portfolio.reporting.tables import build_tables
from risk_aware_portfolio.risk.io import save_risk_snapshots
from risk_aware_portfolio.risk.model import estimate_risk_history
from risk_aware_portfolio.utils.config import load_config
from risk_aware_portfolio.utils.logging import configure_logging
from risk_aware_portfolio.utils.manifest import write_manifest
from risk_aware_portfolio.utils.paths import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ashare.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    verify_real_data(config)
    output = resolve_path(config["real_data"]["output_dir"])
    for name in ("figures", "tables", "backtests", "logs", "risk_model"):
        (output / name).mkdir(parents=True, exist_ok=True)
    configure_logging("validation", output / "logs")
    input_path = resolve_path(config["data"]["input_path"])
    alpha_path = config["data"]["alpha_path"]
    panel = load_panel(input_path, resolve_path(alpha_path) if alpha_path else None)
    print("Estimating causal real-data risk history", flush=True)
    history = estimate_risk_history(panel, config)
    save_risk_snapshots(history, output / "risk_model")
    history.factor_returns.to_parquet(output / "risk_model/factor_returns.parquet")
    history.specific_returns.to_parquet(output / "risk_model/specific_returns.parquet")
    history.regressions.to_csv(output / "tables/factor_regression_diagnostics.csv")
    results = run_primary_comparisons(panel, history, config, output / "backtests")
    frames = build_tables(results, panel, history, config, output / "tables")
    robustness = run_robustness(
        panel, config, results["risk_aware_tc"], history, output / "backtests/experiments"
    )
    robustness.to_csv(output / "tables/robustness_summary.csv", index=False)
    frames["robustness_summary"] = robustness
    local_weight_sensitivity(panel, config, results["risk_aware_tc"], history).to_csv(
        output / "tables/weight_sensitivity.csv"
    )
    print("Verifying actual-panel prefix and future-data perturbation", flush=True)
    checks = real_no_lookahead_checks(panel, config, results["risk_aware_tc"])
    checks.to_csv(output / "tables/no_lookahead_checks.csv", index=False)
    generate_figures(results, config, output / "tables", output / "figures")
    write_real_report(frames, results, config, checks, output)
    write_real_notebook(args.config)
    write_manifest(output / "manifest.json", config, input_path)
    print(f"Empirical validation complete: {output / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
