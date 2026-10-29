"""Estimate sequential risk artifacts, including all pre/post-repair eigenvalues."""

import argparse

import pandas as pd

from risk_aware_portfolio.data.loader import load_panel
from risk_aware_portfolio.risk.io import save_risk_snapshots
from risk_aware_portfolio.risk.model import estimate_risk_history
from risk_aware_portfolio.utils.config import load_config
from risk_aware_portfolio.utils.logging import configure_logging
from risk_aware_portfolio.utils.paths import ensure_directories, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_directories()
    logger = configure_logging("estimate_risk_model")
    alpha_path = config["data"]["alpha_path"]
    panel = load_panel(
        resolve_path(config["data"]["input_path"]), resolve_path(alpha_path) if alpha_path else None
    )
    history = estimate_risk_history(panel, config)
    if not history.snapshots:
        raise ValueError("Insufficient observations for risk model warmup")
    save_risk_snapshots(history, resolve_path("data/interim"))
    for name, frame in (
        ("factor_returns", history.factor_returns),
        ("specific_returns", history.specific_returns),
        ("regression_diagnostics", history.regressions),
    ):
        frame.to_parquet(resolve_path(f"data/interim/{name}.parquet"))
    rows, eigen_rows, factor_rows = [], [], []
    for date, snapshot in history.snapshots.items():
        rows.append(
            {
                "date": date,
                **snapshot.audit.metrics(),
                "factor_condition_number": snapshot.factor_audit.metrics()["condition_number"],
            }
        )
        for i, (before, after) in enumerate(zip(snapshot.audit.before, snapshot.audit.after)):
            eigen_rows.append({"date": date, "component": i, "before": before, "after": after})
        for i, (before, after) in enumerate(
            zip(snapshot.factor_audit.before, snapshot.factor_audit.after)
        ):
            factor_rows.append({"date": date, "component": i, "before": before, "after": after})
    pd.DataFrame(rows).to_csv(
        resolve_path("outputs/tables/covariance_diagnostics.csv"), index=False
    )
    pd.DataFrame(eigen_rows).to_csv(
        resolve_path("outputs/tables/covariance_eigenvalues.csv"), index=False
    )
    pd.DataFrame(factor_rows).to_csv(
        resolve_path("outputs/tables/factor_eigenvalues.csv"), index=False
    )
    logger.info(
        "Completed %d regressions and %d as-of risk forecasts",
        len(history.factor_returns),
        len(history.snapshots),
    )


if __name__ == "__main__":
    main()
