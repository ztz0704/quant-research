"""Publication-quality PNG and SVG figures on the complete declared evaluation period."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk_aware_portfolio.attribution.factor_attribution import cumulative_attribution
from risk_aware_portfolio.attribution.performance import drawdown
from risk_aware_portfolio.backtest.engine import BacktestResult
from risk_aware_portfolio.utils.config import Config

COLORS = ("#64748b", "#dc7f37", "#136f89", "#6955a5")


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    """Preserve vector graphics and export a 300-DPI bitmap."""
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _axis(title: str, ylabel: str, size: tuple[float, float] = (9, 4)) -> tuple:
    fig, ax = plt.subplots(figsize=size)
    ax.set(title=title, ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    return fig, ax


def _comparison(
    results: dict[str, BacktestResult],
    column: str,
    title: str,
    ylabel: str,
    output: Path,
    name: str,
    transform=None,
) -> None:
    fig, ax = _axis(title, ylabel)
    for color, result in zip(COLORS, results.values()):
        series = result.returns[column]
        if transform:
            series = transform(series)
        ax.plot(series.index, series, color=color, label=result.name, lw=1.2)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, output, name)


def _risk_figures(main: BacktestResult, tables: Path, output: Path) -> None:
    calibration = pd.read_csv(
        tables / "risk_calibration_daily.csv", parse_dates=["date"]
    ).set_index("date")
    fig, ax = _axis(
        "Risk forecast versus forward volatility of frozen weights", "Annual volatility"
    )
    ax.plot(
        main.returns["decision_date"], main.returns["predicted_volatility"], label="As-of forecast"
    )
    ax.plot(
        calibration.index,
        calibration["forward_realized_volatility"],
        label="Next 21 sessions",
        alpha=0.7,
    )
    ax.legend()
    _save(fig, output, "predicted_vs_realized_risk")
    risk = main.risk_contributions
    factor = pd.read_csv(tables / "factor_risk_contributions.csv", parse_dates=["date"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(risk.index, risk["factor_risk_fraction"], color=COLORS[2])
    axes[0].set(title="Factor share of variance", ylabel="Fraction")
    means = factor.groupby("factor")["variance_contribution"].mean().sort_values()
    axes[1].barh(means.index, means, color=COLORS[2])
    axes[1].set(title="Signed factor Euler contributions", xlabel="Mean daily variance")
    _save(fig, output, "factor_risk_contribution")
    fig, ax = _axis("Specific share of portfolio variance", "Fraction")
    ax.plot(risk.index, risk["specific_risk_fraction"], color=COLORS[3])
    _save(fig, output, "specific_risk_contribution")


def _covariance_figures(tables: Path, output: Path) -> None:
    stability = pd.read_csv(tables / "covariance_stability.csv", parse_dates=["date"]).set_index(
        "date"
    )
    fig, ax = _axis("Asset covariance eigenvalue range", "Daily variance, log scale")
    for column in ("smallest_eigenvalue", "largest_eigenvalue"):
        ax.plot(stability.index, stability[column], label=column.replace("_", " "))
    ax.set_yscale("log")
    ax.legend()
    _save(fig, output, "covariance_eigenvalues")
    fig, ax = _axis("Covariance conditioning", "Condition number, log scale")
    ax.plot(stability.index, stability["condition_number"], label="Asset covariance")
    ax.plot(stability.index, stability["factor_condition_number"], label="Factor covariance")
    ax.set_yscale("log")
    ax.legend()
    _save(fig, output, "covariance_condition_number")


def _exposure_figures(main: BacktestResult, tables: Path, output: Path, config: Config) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    main.factor_exposures[config["risk_model"]["styles"]].plot(ax=axes[0], lw=1.0)
    axes[0].set(title="Style exposures", ylabel="Exposure (cross-sectional z units)")
    industry = pd.read_csv(tables / "industry_exposures.csv", parse_dates=["date"]).set_index(
        "date"
    )
    industry.plot(ax=axes[1], lw=1.0)
    axes[1].set(title="Industry portfolio weights", ylabel="Net industry weight")
    for ax in axes:
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.2)
    _save(fig, output, "factor_exposures")


def _attribution_figures(main: BacktestResult, tables: Path, output: Path) -> None:
    pnl = cumulative_attribution(main)
    fig, ax = _axis("NAV-consistent cumulative attribution", "Contribution to NAV change")
    for column in ("factor", "alpha_forecast", "specific_after_alpha", "transaction_cost"):
        ax.plot(pnl.index, pnl[column], label=column.replace("_", " "))
    ax.plot(pnl.index, pnl["total"], label="Net NAV change", color="black", lw=1.6)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, output, "factor_attribution")
    fig, ax = _axis(
        "Alpha forecast contribution and optional DGP truth", "Contribution to NAV change"
    )
    ax.plot(pnl.index, pnl["alpha_forecast"], label="Forecast (not causal identification)")
    path = tables / "risk_aware_tc_truth_attribution.csv"
    if path.exists():
        truth = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        if not truth.empty:
            prior_nav = main.returns["nav"].shift(1, fill_value=1)
            ax.plot(
                truth.index,
                (truth["true_alpha"] * prior_nav).cumsum(),
                label="Synthetic true alpha",
            )
    ax.legend()
    _save(fig, output, "alpha_attribution")


def _concentration_figures(results: dict[str, BacktestResult], tables: Path, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for color, (slug, result) in zip(COLORS, results.items()):
        frame = pd.read_csv(tables / f"{slug}_concentration.csv", parse_dates=["date"]).set_index(
            "date"
        )
        axes[0].plot(frame.index, frame["maximum_position"], label=result.name, color=color)
        axes[1].plot(frame.index, frame["effective_number_of_bets"], color=color)
    axes[0].set(title="Position concentration", ylabel="Maximum absolute weight")
    axes[1].set(ylabel="Effective number of bets", xlabel="Decision date")
    axes[0].legend(fontsize=8, ncol=2)
    _save(fig, output, "portfolio_concentration")


def generate_figures(
    results: dict[str, BacktestResult], config: Config, tables: Path, output: Path
) -> None:
    """Use predeclared rolling windows, with no period trimming to improve results."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "lines.linewidth": 1.2,
            "svg.fonttype": "none",
        }
    )
    main = results["risk_aware_tc"]
    annual = config["backtest"]["annualization"]
    settings = config["reporting"]
    _comparison(
        results,
        "net_return",
        "Cumulative net wealth: full evaluation sample",
        "NAV",
        output,
        "cumulative_net_comparison",
        lambda x: (1 + x).cumprod(),
    )
    fig, ax = _axis("Risk-Aware + TC: gross and self-financing net wealth", "NAV")
    for col, label in (("gross_nav", "Gross (same net-path weights)"), ("nav", "Net")):
        ax.plot(main.returns.index, main.returns[col], label=label)
    ax.legend()
    _save(fig, output, "cumulative_gross_vs_net_return")
    _comparison(
        results,
        "net_return",
        "Drawdown including initial NAV",
        "Drawdown",
        output,
        "drawdown",
        drawdown,
    )
    window = settings["rolling_sharpe_window"]
    _comparison(
        results,
        "net_return",
        f"Rolling Sharpe ({window} sessions)",
        "Sharpe",
        output,
        "rolling_sharpe",
        lambda x: x.rolling(window).mean() / x.rolling(window).std(ddof=1) * np.sqrt(annual),
    )
    window = settings["rolling_volatility_window"]
    _comparison(
        results,
        "net_return",
        f"Rolling volatility ({window} sessions)",
        "Annual volatility",
        output,
        "rolling_volatility",
        lambda x: x.rolling(window).std(ddof=1) * np.sqrt(annual),
    )
    _comparison(
        results, "turnover", "Actual one-way trades / pre-trade NAV", "Turnover", output, "turnover"
    )
    _comparison(
        results,
        "transaction_cost",
        "Cumulative fee fractions (not additive NAV drag)",
        "Cost fraction",
        output,
        "transaction_cost",
        lambda x: x.cumsum(),
    )
    _concentration_figures(results, tables, output)
    _risk_figures(main, tables, output)
    _covariance_figures(tables, output)
    _exposure_figures(main, tables, output, config)
    _attribution_figures(main, tables, output)
    sensitivity = pd.read_csv(tables / "weight_sensitivity.csv", parse_dates=["date"]).set_index(
        "date"
    )
    fig, ax = _axis(
        "Local alpha sensitivity with frozen risk and prior holdings", "L1 weight change"
    )
    ax.plot(sensitivity.index, sensitivity["weight_sensitivity_l1"], color=COLORS[2])
    _save(fig, output, "optimizer_stability")
