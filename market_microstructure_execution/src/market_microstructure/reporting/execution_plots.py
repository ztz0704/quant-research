"""Execution ledgers, inventory, sensitivity and simplified maker diagnostics."""

import numpy as np
import pandas as pd

from market_microstructure.reporting.plots import (
    COLORS,
    empty_annotation,
    local_time,
    new_plot,
    save_plot,
)
from market_microstructure.utils.paths import project_path


def execution_figures(selected_model: str, rolling_window: int, markout_horizon: int) -> list[str]:
    """Use marked ledgers and explicitly display unobservable closed-trade statistics."""
    paths = []
    events = {mode: pd.read_parquet(project_path(f"outputs/backtests/{selected_model}_{mode}_events.parquet"))
              for mode in ("mid", "taker", "maker")}
    for column, name, title, unit in (("gross_pnl", "cumulative_gross_pnl", "Cumulative gross marked PnL", "Gross marked PnL (CNY)"),
                                     ("net_pnl", "cumulative_net_pnl", "Cumulative net marked PnL", "Net marked PnL (CNY)"),
                                     ("position", "inventory_path", "Inventory path | A-share T+1", "Inventory (shares)")):
        fig, ax = new_plot(f"{title} | {selected_model}", "Exchange time (Asia/Shanghai)", unit)
        for mode, table in events.items():
            label = "mid (UNREALISTIC UPPER BOUND)" if mode == "mid" else mode
            ax.plot(local_time(table.timestamp), table[column], label=label, color=COLORS[mode], lw=1)
        if column == "net_pnl":
            passive = pd.read_parquet(project_path("outputs/backtests/always_long_taker_events.parquet"))
            ax.plot(local_time(passive.timestamp), passive.net_pnl, color="#639453", ls="--", label="constant long taker")
        ax.legend(fontsize=8)
        paths.append(save_plot(fig, name))
    summary = pd.read_csv(project_path("outputs/tables/execution_summary.csv"))
    chosen = summary.loc[summary.model == selected_model].set_index("mode").reindex(["mid", "taker", "maker"])
    fig, ax = new_plot(f"Gross alpha and execution deductions | {selected_model}", "Execution assumption", "Final marked PnL / contribution (CNY)")
    x = np.arange(3)
    ax.bar(x - 0.25, chosen.gross_pnl, 0.16, label="gross alpha")
    for j, (column, label) in enumerate((("spread_cost", "spread"), ("total_fees", "fees"),
                                       ("total_slippage", "slippage"), ("market_impact", "impact"))):
        ax.bar(x - 0.08 + j * 0.12, -chosen[column], 0.1, label=label)
    ax.scatter(x, chosen.net_pnl, marker="D", color="black", label="net marked PnL", zorder=5)
    ax.set_xticks(x, ["mid: unrealistic", "taker", "maker: simplified"])
    ax.legend(fontsize=8, ncols=3)
    paths.append(save_plot(fig, "pnl_decomposition"))
    fig, ax = new_plot(f"Rolling net marked PnL | {selected_model}", "Exchange time (Asia/Shanghai)", f"PnL over {rolling_window} observations (CNY)")
    for mode in ("taker", "maker"):
        table = events[mode]
        delta = table.net_pnl.diff().fillna(table.net_pnl.iloc[0]).rolling(rolling_window).sum()
        ax.plot(local_time(table.timestamp), delta, label=mode, color=COLORS[mode], lw=0.8)
    ax.axhline(0, color="black", lw=0.7)
    ax.legend()
    paths.append(save_plot(fig, "rolling_pnl"))
    closed = pd.read_parquet(project_path(f"outputs/backtests/{selected_model}_taker_closed_lots.parquet"))
    fig, ax = new_plot("Rolling closed-trade hit rate | taker", "Closed-trade sequence (index)", "Positive net closed-lot fraction")
    if closed.empty:
        empty_annotation(ax, "No closed lots in this sample\nT+1 blocks sales of same-day purchases\nHit rate and completed holding time are unavailable")
    else:
        ax.plot(closed.net_pnl.gt(0).rolling(rolling_window, min_periods=1).mean())
    ax.set_ylim(0, 1)
    paths.append(save_plot(fig, "rolling_hit_rate"))
    counts = summary.loc[summary.model != "always_long"].pivot(index="model", columns="mode", values="trade_count")
    fig, ax = new_plot("Fill counts across all model/execution comparisons", "Model", "Executed fills (count; not round trips)")
    counts.plot.bar(ax=ax, color=[COLORS[c] for c in counts], rot=0)
    paths.append(save_plot(fig, "trade_count"))
    stress = pd.read_csv(project_path("outputs/tables/robustness_summary.csv"))
    for dimension, name, xlabel in (("latency_events", "pnl_vs_latency", "Latency (retained observations; approximately seconds)"),
                                     ("spread_multiplier", "pnl_vs_spread", "Quoted spread multiplier (x)"),
                                     ("fee_multiplier", "pnl_vs_transaction_cost", "Broker commission multiplier, including minimum (x)")):
        part = stress.loc[stress.dimension == dimension].sort_values("parameter")
        fig, ax = new_plot("Conditional execution sensitivity | fixed OOS signal", xlabel, "Final net marked PnL (CNY)")
        ax.plot(part.parameter, part.net_pnl, marker="o", color="#126b85")
        paths.append(save_plot(fig, name))
    fig, ax = new_plot("All execution stresses | baseline-relative result", "Scenario (all predeclared cases)", "Change in net marked PnL (CNY)")
    delta = stress.net_pnl - stress.loc[stress.scenario == "baseline", "net_pnl"].iloc[0]
    ax.bar(stress.scenario, delta, color=np.where(delta < 0, "#a64462", "#126b85"))
    ax.tick_params(axis="x", rotation=65, labelsize=8)
    paths.append(save_plot(fig, "all_stress_scenarios"))
    orders = pd.read_parquet(project_path(f"outputs/backtests/{selected_model}_maker_orders.parquet"))
    attempts = orders.loc[orders.status == "maker_attempt"]
    fig, ax = new_plot("Simplified maker probabilities | uncalibrated", "Per-observation conditional probability (fraction)", "Eligible attempts (count)")
    eligible = attempts.loc[attempts.eligible.eq(True)]
    if eligible.empty:
        empty_annotation(ax, "No eligible maker attempts")
    else:
        ax.hist(eligible.fill_probability, bins=20, color="#dd8b2e")
    paths.append(save_plot(fig, "fill_probability"))
    markouts = pd.read_csv(project_path("outputs/tables/maker_markouts.csv"))
    fig, ax = new_plot("Maker adverse selection | signed post-fill mid markout", f"Signed {markout_horizon}-observation markout (bps; negative is adverse)", "Maker fills (count)")
    valid = markouts.signed_markout_bps.dropna()
    if valid.empty:
        empty_annotation(ax, "No maker fills with available future labels")
    else:
        ax.hist(valid, bins=min(15, len(valid)), color="#dd8b2e")
        ax.axvline(0, color="black", ls="--", lw=1)
    paths.append(save_plot(fig, "maker_adverse_selection"))
    regimes = pd.read_csv(project_path("outputs/tables/regime_summary.csv"))
    regime = regimes.loc[regimes["mode"] == "taker"]
    fig, ax = new_plot("Regime event-PnL attribution | training-only cutoffs", "Marginal regime (pairs overlap across dimensions)", "Attributed net marked PnL (CNY)")
    ax.bar(regime.regime, regime.net_pnl, color="#126b85")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    paths.append(save_plot(fig, "regime_pnl"))
    return paths
