"""Run the four predeclared portfolio comparisons on identical periods."""

import argparse

from risk_aware_portfolio.backtest.comparisons import STRATEGIES
from risk_aware_portfolio.backtest.engine import run_backtest
from risk_aware_portfolio.data.loader import load_panel
from risk_aware_portfolio.risk.model import estimate_risk_history
from risk_aware_portfolio.utils.config import load_config, variant
from risk_aware_portfolio.utils.logging import configure_logging
from risk_aware_portfolio.utils.manifest import write_manifest
from risk_aware_portfolio.utils.paths import ensure_directories, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_directories()
    logger = configure_logging("backtest")
    input_path = resolve_path(config["data"]["input_path"])
    alpha_path = config["data"]["alpha_path"]
    panel = load_panel(
        input_path,
        resolve_path(alpha_path) if alpha_path else None,
        config["data"]["max_abs_return"],
    )
    history = estimate_risk_history(panel, config)
    for name, slug, baseline, neutral, penalty in STRATEGIES:
        updates = {
            "cost_penalty": config["portfolio"]["cost_penalty"] if penalty is None else penalty
        }
        settings = variant(
            config,
            portfolio=updates,
            alpha={"neutralize": config["alpha"]["neutralize"] if neutral is None else neutral},
        )
        result = run_backtest(panel, settings, name, baseline, history)
        result.save(resolve_path(f"outputs/backtests/{slug}"))
        print(
            f"{name}: {len(result.returns)} periods, final NAV {result.returns['nav'].iloc[-1]:.6f}"
        )
        logger.info(
            "Finished %s: %d periods, final NAV %.6f",
            name,
            len(result.returns),
            result.returns["nav"].iloc[-1],
        )
    write_manifest(resolve_path("outputs/backtests/manifest.json"), config, input_path)


if __name__ == "__main__":
    main()
