"""Entire AC risk grid plus one predeclared baseline slice for readable trajectories."""

import pandas as pd

from market_microstructure.reporting.plots import new_plot, save_plot


def optimal_figures(summary: pd.DataFrame, trajectories: pd.DataFrame) -> list[str]:
    """Trajectory slice is maximum configured lambda, unit multipliers; all cases retained."""
    paths = []
    base = summary.loc[(summary.sigma_multiplier == 1) & (summary.eta_multiplier == 1)
                       & (summary.size_multiplier == 1)]
    selected = base.loc[base.risk_aversion_per_cny == base.risk_aversion_per_cny.max()]
    sample = trajectories.loc[trajectories.experiment == selected.experiment.iloc[0]]
    for column, name, title, unit in (("trade_shares", "almgren_chriss_execution_trajectory", "Execution trajectories | maximum configured risk aversion", "Planned child order (shares)"),
                                     ("inventory_shares", "almgren_chriss_inventory_trajectory", "Inventory trajectories | maximum configured risk aversion", "Remaining inventory (shares)")):
        fig, ax = new_plot(title, "Elapsed execution time (seconds)", unit)
        for schedule, part in sample.groupby("schedule"):
            part = part.loc[part.step > 0] if column == "trade_shares" else part
            ax.plot(part.elapsed_seconds, part[column], marker=".", label=schedule.replace("_", " "))
        ax.legend(fontsize=8)
        paths.append(save_plot(fig, name))
    fig, ax = new_plot("Expected execution cost across risk preferences", "Risk aversion (1/CNY; zero shown separately)", "Expected shortfall (CNY)")
    for schedule, part in base.groupby("schedule"):
        part = part.sort_values("risk_aversion_per_cny")
        ax.plot(part.risk_aversion_per_cny, part.expected_cost_cny, marker="o", label=schedule.replace("_", " "))
    ax.set_xscale("symlog", linthresh=0.00001)
    ax.legend(fontsize=8)
    paths.append(save_plot(fig, "execution_cost_vs_risk_aversion"))
    fig, ax = new_plot("Execution cost-risk frontier | real training risk, assumed impact", "Shortfall standard deviation (CNY)", "Expected shortfall (CNY)")
    for schedule, part in base.groupby("schedule"):
        ax.plot(part.cost_std_cny, part.expected_cost_cny, marker="o", label=schedule.replace("_", " "))
    ax.legend(fontsize=8)
    paths.append(save_plot(fig, "execution_cost_risk_frontier"))
    fig, ax = new_plot("AC sensitivity to impact, volatility and parent size", "Cost standard deviation (CNY)", "Expected shortfall (CNY)")
    ac = summary.loc[summary.schedule == "Almgren_Chriss"]
    for (sigma, eta, size), part in ac.groupby(["sigma_multiplier", "eta_multiplier", "size_multiplier"]):
        ax.plot(part.cost_std_cny, part.expected_cost_cny, marker=".", label=f"sigma {sigma:g}x, eta {eta:g}x, size {size:g}x")
    ax.legend(fontsize=7, ncols=2)
    paths.append(save_plot(fig, "optimal_execution_parameter_sensitivity"))
    return paths
