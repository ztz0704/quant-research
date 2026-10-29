"""Measured research narrative and separate experiment report."""

import pandas as pd

from risk_aware_portfolio.backtest.engine import BacktestResult
from risk_aware_portfolio.utils.config import Config
from risk_aware_portfolio.utils.paths import resolve_path


def markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table, keeping all comparison rows."""
    formatted = frame.copy()
    for name in formatted.select_dtypes(include="number"):
        formatted[name] = formatted[name].map(lambda value: f"{value:.4f}")
    lines = [
        "| " + " | ".join(formatted.columns) + " |",
        "| " + " | ".join(["---"] * len(formatted.columns)) + " |",
    ]
    lines += [
        "| " + " | ".join(map(str, row)) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def measured_results(
    frames: dict[str, pd.DataFrame], results: dict[str, BacktestResult], config: Config
) -> str:
    """State simulation scope and actual outcomes without a profitability claim."""
    main = results["risk_aware_tc"]
    daily = main.returns
    performance = frames["performance_summary"]
    comparison = performance[
        [
            "portfolio",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
            "average_turnover",
            "total_transaction_cost",
        ]
    ]
    solved = main.solver_diagnostics
    solved = solved[solved["status"].isin(["optimal", "optimal_inaccurate"])]
    inaccurate = int((solved["status"] == "optimal_inaccurate").sum())
    risk = frames["risk_summary"].set_index("portfolio").loc[main.name]
    pairs = frames["pairwise_comparisons"]
    robustness = frames["robustness_summary"]
    failures = int((robustness["status"] == "failed").sum())
    text = f"""Fixed-seed synthetic evaluation; {len(main.weights.columns)} assets and {len(daily)} realized
periods, {daily["return_date"].iloc[0].date()} through {daily["return_date"].iloc[-1].date()}.
The sample, costs and parameters were declared in `configs/base.yaml` before evaluation.
All portfolios pay the same ex-post cost model. The rank baseline has no optimizer
constraints; this comparison therefore evaluates the combined construction policy.

{markdown_table(comparison)}

Annualized return is geometric; Sharpe uses daily arithmetic mean and sample
volatility, zero risk-free rate. Turnover is one-way L1 actual trades / pre-trade
NAV, including initial deployment. Total cost sums fee fractions; exact cumulative
NAV drag appears in the linked attribution ledger.

Risk-Aware + TC has mean forecast annual volatility **{risk["average_predicted_volatility"]:.2%}**
versus realized gross volatility **{risk["realized_gross_volatility"]:.2%}**. Mean squared
daily return / predicted variance is **{risk["mean_squared_return_to_variance"]:.3f}** (a calibration
diagnostic, not a test of strategy profitability). Mean factor share of variance
is **{risk["average_factor_risk_fraction"]:.1%}**.

There were **{len(solved)}** optimization decisions, including **{inaccurate}**
`optimal_inaccurate` solutions. Feasibility is independently checked; inaccurate
status is retained and is not an optimality certificate. The experiment suite
contains **{len(robustness)}** declared cases with **{failures}** failures; every case is retained.

Paired HAC mean net-return differences against the baseline:

{markdown_table(pairs[["portfolio", "mean", "hac_t_stat", "p_value", "holm_p_value"]])}

These are asymptotic HAC tests of mean return differences, not tests of Sharpe
improvement. Holm adjustment covers the three declared baseline comparisons;
parameter-grid outcomes are descriptive and are not used to select a winning strategy.
See `outputs/tables/performance_summary.csv` for block-bootstrap Sharpe intervals.
One synthetic seed and about one evaluation year cannot establish empirical alpha
or investment performance. Positive or negative outcomes are both retained.
"""
    return text


def write_research_report(
    frames: dict[str, pd.DataFrame], results: dict[str, BacktestResult], config: Config
) -> None:
    """Write the experiment appendix; project README is curated documentation."""
    results_text = measured_results(frames, results, config)
    report = "# Fixed-seed research results\n\n" + results_text
    for name in (
        "covariance_method_comparison",
        "risk_summary",
        "attribution_summary",
        "alpha_ic_summary",
        "robustness_summary",
        "factor_return_recovery",
    ):
        if name in frames:
            frame = frames[name]
            if name == "robustness_summary":
                columns = [
                    "experiment",
                    "status",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe",
                    "average_turnover",
                    "maximum_position",
                    "mean_weight_instability_l1",
                ]
                frame = frame[[col for col in columns if col in frame]]
            report += (
                "\n## " + name.replace("_", " ").title() + "\n\n" + markdown_table(frame) + "\n"
            )
    resolve_path("outputs/research_report.md").write_text(report, encoding="utf-8")
