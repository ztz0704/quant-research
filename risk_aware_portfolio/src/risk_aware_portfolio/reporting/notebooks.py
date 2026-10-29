"""Thin, executable research notebooks: diagnostics and visualization only."""

import hashlib
import json

from risk_aware_portfolio.utils.paths import resolve_path

SETUP = """import pandas as pd
from IPython.display import Image, Markdown, display
from risk_aware_portfolio.utils.config import load_config
from risk_aware_portfolio.utils.paths import resolve_path
config = load_config()
"""


def _cell(kind: str, source: str) -> dict:
    cell = {
        "cell_type": kind,
        "id": hashlib.sha256(source.encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _write(name: str, cells: list[tuple[str, str]]) -> None:
    notebook = {
        "cells": [_cell(kind, source) for kind, source in cells],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.12 (project .venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    resolve_path(f"notebooks/{name}.ipynb").write_text(
        json.dumps(notebook, indent=2), encoding="utf-8"
    )


def write_notebooks() -> None:
    """Create all four notebooks from reusable core APIs and verified result artifacts."""
    _write(
        "01_data_diagnostics",
        [
            (
                "markdown",
                "# Data diagnostics\n\nThese are ex-post descriptive checks, not fitted preprocessing statistics. Core data validation rejects missing records and never fills them.",
            ),
            (
                "code",
                SETUP
                + """from risk_aware_portfolio.data.loader import load_panel
from risk_aware_portfolio.data.validation import validate_panel
from risk_aware_portfolio.data.universe import require_balanced_universe
alpha_path = config["data"]["alpha_path"]
panel = load_panel(resolve_path(config["data"]["input_path"]),
                   resolve_path(alpha_path) if alpha_path else None)
validate_panel(panel)
print("Assets:", len(require_balanced_universe(panel)))
display(panel.head())
display(panel.groupby("date").size().describe())
""",
            ),
            (
                "markdown",
                "Alpha scores at decision date t are compared with return at t+1. The following IC series is descriptive; it is never fed into historical portfolio weights.",
            ),
            (
                "code",
                """display(pd.read_csv(resolve_path("outputs/tables/alpha_ic_summary.csv")))
ic = pd.read_csv(resolve_path("outputs/tables/alpha_ic.csv"), parse_dates=["date"]).set_index("date")
ic.rolling(21).mean().plot(title="Rolling next-period rank IC");
""",
            ),
        ],
    )
    _write(
        "02_risk_model_diagnostics",
        [
            (
                "markdown",
                "# Factor risk model diagnostics\n\nRegression uses X_(t-1) for r_t; covariance uses trailing factor returns through t. Repair and conditioning are auditable.",
            ),
            (
                "code",
                SETUP
                + """display(pd.read_csv(resolve_path("outputs/tables/risk_summary.csv")))
display(pd.read_csv(resolve_path("outputs/tables/factor_return_recovery.csv")))
display(pd.read_parquet(resolve_path("data/interim/regression_diagnostics.parquet")).describe())
""",
            ),
            (
                "code",
                """for name in ["covariance_eigenvalues", "covariance_condition_number", "predicted_vs_realized_risk"]:
    display(Image(filename=str(resolve_path(f"outputs/figures/{name}.png")), width=850))
""",
            ),
            (
                "markdown",
                "Forward realized volatility freezes today's weights for the next 21 sessions. Overlapping windows and ex-post bins are diagnostic only. Regression factors can absorb the part of true synthetic alpha aligned with risk exposures.",
            ),
            (
                "code",
                """display(pd.read_csv(resolve_path("outputs/tables/risk_calibration_buckets.csv")))
""",
            ),
        ],
    )
    _write(
        "03_portfolio_analysis",
        [
            (
                "markdown",
                "# Portfolio construction and implementation costs\n\nAll four portfolios use identical evaluation periods and realized cost assumptions. The rank baseline has different construction constraints.",
            ),
            (
                "code",
                SETUP
                + """display(pd.read_csv(resolve_path("outputs/tables/performance_summary.csv")))
display(pd.read_csv(resolve_path("outputs/tables/pairwise_comparisons.csv")))
""",
            ),
            (
                "code",
                """for name in ["cumulative_gross_vs_net_return", "portfolio_concentration", "turnover", "factor_exposures"]:
    display(Image(filename=str(resolve_path(f"outputs/figures/{name}.png")), width=850))
""",
            ),
            (
                "code",
                """display(pd.read_csv(resolve_path("outputs/tables/risk_aware_tc_binding_frequency.csv")))
display(pd.read_csv(resolve_path("outputs/tables/risk_aware_tc_solver_status.csv")))
display(pd.read_csv(resolve_path("outputs/tables/weight_sensitivity.csv")).describe())
""",
            ),
        ],
    )
    _write(
        "04_final_report",
        [
            (
                "markdown",
                "# Risk-Aware Portfolio Construction\n\nBarra-style Factor Risk Model, Constrained Portfolio Optimization and Performance Attribution\n\nThis report studies the transformation of a prevalidated alpha into a portfolio. Synthetic returns establish correctness and mechanisms; they do not demonstrate empirical alpha.",
            ),
            (
                "code",
                SETUP
                + """display(Markdown(resolve_path("outputs/research_report.md").read_text(encoding="utf-8")))
""",
            ),
            (
                "code",
                """for name in ["cumulative_net_comparison", "drawdown", "factor_attribution", "alpha_attribution", "optimizer_stability"]:
    display(Image(filename=str(resolve_path(f"outputs/figures/{name}.png")), width=850))
""",
            ),
            (
                "markdown",
                "## Interpretation limits\n\nFactor + specific − cost is an accounting identity. Forecast alpha + residual-after-forecast is a labeled proxy split, not causal identification. Synthetic truth is stored separately and content-hash checked. Risk neutralization may remove predictive factor premia; optimization may amplify estimation errors; turnover and impact can reverse gross gains. No parameter setting is selected using these results. Real deployment needs point-in-time fundamentals, event handling, execution prices, borrow and funding costs, and capacity analysis.",
            ),
        ],
    )
