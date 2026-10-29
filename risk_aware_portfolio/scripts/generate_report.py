"""Generate the complete predeclared research suite, figures, tables and notebooks."""

import argparse

import pandas as pd

from risk_aware_portfolio.backtest.engine import BacktestResult
from risk_aware_portfolio.data.loader import load_panel
from risk_aware_portfolio.diagnostics.robustness import local_weight_sensitivity, run_robustness
from risk_aware_portfolio.reporting.figures import generate_figures
from risk_aware_portfolio.reporting.notebooks import write_notebooks
from risk_aware_portfolio.reporting.research_report import write_research_report
from risk_aware_portfolio.reporting.tables import build_tables
from risk_aware_portfolio.risk.model import estimate_risk_history
from risk_aware_portfolio.utils.config import load_config
from risk_aware_portfolio.utils.logging import configure_logging
from risk_aware_portfolio.utils.manifest import verify_manifest, write_manifest
from risk_aware_portfolio.utils.paths import ensure_directories, resolve_path

PORTFOLIOS = {
    "baseline": "Baseline",
    "risk_aware": "Risk-Aware Portfolio",
    "risk_aware_tc": "Risk-Aware + TC",
    "neutralized": "Neutralized Alpha Portfolio",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Reuse experiments only after provenance validation",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_directories()
    logger = configure_logging("generate_report")
    input_path = resolve_path(config["data"]["input_path"])
    verify_manifest(resolve_path("outputs/backtests/manifest.json"), config, input_path)
    alpha_path = config["data"]["alpha_path"]
    panel = load_panel(
        input_path,
        resolve_path(alpha_path) if alpha_path else None,
        config["data"]["max_abs_return"],
    )
    results = {
        slug: BacktestResult.load(name, resolve_path(f"outputs/backtests/{slug}"))
        for slug, name in PORTFOLIOS.items()
    }
    history = estimate_risk_history(panel, config)
    frames = build_tables(results, panel, history, config)
    experiment_output = resolve_path("outputs/backtests/experiments")
    if args.skip_experiments:
        verify_manifest(experiment_output / "provenance.json", config, input_path)
        robustness = pd.read_csv(resolve_path("outputs/tables/robustness_summary.csv"))
    else:
        robustness = run_robustness(
            panel, config, results["risk_aware_tc"], history, experiment_output
        )
        robustness.to_csv(resolve_path("outputs/tables/robustness_summary.csv"), index=False)
        write_manifest(experiment_output / "provenance.json", config, input_path)
    frames["robustness_summary"] = robustness
    sensitivity = local_weight_sensitivity(panel, config, results["risk_aware_tc"], history)
    sensitivity.to_csv(resolve_path("outputs/tables/weight_sensitivity.csv"))
    generate_figures(
        results, config, resolve_path("outputs/tables"), resolve_path("outputs/figures")
    )
    write_research_report(frames, results, config)
    write_notebooks()
    write_manifest(resolve_path("outputs/report_manifest.json"), config, input_path)
    logger.info("Generated all figures, tables, report and notebooks")
    print("Research artifacts ready: outputs/figures, outputs/tables, outputs/research_report.md")


if __name__ == "__main__":
    main()
