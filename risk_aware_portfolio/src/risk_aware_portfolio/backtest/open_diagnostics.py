"""Long-only rank diagnostics for V2 open labels using existing NAV accounting.

Targets are determined at close solely by scores. At next open, old holdings
have drifted to that execution instant, which is used ONLY in ex-post costs and
accounting, never in target selection. Unknown exits stop the continuous NAV
chain. This is a full-fill research proxy, not an order/fill or risk optimizer.
"""

import numpy as np
import pandas as pd

from risk_aware_portfolio.backtest.accounting import account_period
from risk_aware_portfolio.costs.transaction_cost import TransactionCost


def run_open_event_backtest(*args, **kwargs):
    """V3 event ledger entry; V2 rank_open_diagnostic remains a frozen comparator."""
    from risk_aware_portfolio.backtest.open_events import run_open_events

    return run_open_events(*args, **kwargs)


def rank_targets(section: pd.DataFrame, score: str, quantile: float) -> pd.Series:
    if section.empty or not 0 < quantile <= 0.5 or not np.isfinite(section[score]).all():
        raise ValueError("Invalid rank diagnostic scores/quantile")
    result = pd.Series(0.0, index=section.asset)
    if section[score].std(ddof=0) < 1e-12:
        result[:] = 1 / len(result)
    else:
        count = max(1, int(len(section) * quantile))
        chosen = section.sort_values([score, "asset"]).asset.iloc[-count:]
        result.loc[chosen] = 1 / count
    return result


def rank_open_diagnostic(
    predictions: pd.DataFrame,
    score: str,
    cost_bps: float,
    quantile: float = 0.2,
    outcome_panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.duplicated(["date", "asset"]).any():
        raise ValueError("Duplicate diagnostic prediction")
    required_quotes = {
        "entry_raw_open",
        "entry_buyable",
        "entry_sellable",
        "exit_sellable",
        "target",
        "target_valid",
    }
    source = outcome_panel if outcome_panel is not None else predictions
    if required_quotes - set(source):
        raise ValueError("Open diagnostics require explicit raw prices and directional states")
    if not predictions.return_convention.eq(
        "close_decision_next_open_to_following_open_simple"
    ).all():
        raise ValueError("Open diagnostic requires exact V2 interval")
    rows, weights = [], []
    previous = pd.Series(dtype=float)
    nav, previous_exit, blocked = 1.0, None, False
    blocked_reason, blocked_assets = "", ""
    cost = TransactionCost(linear_bps=cost_bps, impact_coefficient=0)
    for date, section in predictions.groupby("date", sort=True):
        target = rank_targets(section, score, quantile)
        weights.append(
            pd.DataFrame({"date": date, "asset": target.index, "weight": target.to_numpy()})
        )
        entry, exit_date = section.label_start.iloc[0], section.label_end.iloc[0]
        if section.label_start.nunique() != 1 or section.label_end.nunique() != 1:
            raise ValueError("Mixed exchange intervals need event scheduling")
        # No implicit liquidation/reset across omitted dates or OOS fold gaps.
        if previous_exit is not None and entry != previous_exit:
            blocked = True
            blocked_reason = blocked_reason or "noncontiguous_holding_intervals"
        universe = target.index.union(previous.index)
        current = target.reindex(universe, fill_value=0.0)
        held = previous.reindex(universe, fill_value=0.0)
        # Leaving prediction eligibility does not erase an available liquidation
        # quote. Retrieve actual outcomes from the full panel, without stale scores.
        full = (
            outcome_panel.loc[outcome_panel.date.eq(date)] if outcome_panel is not None else section
        )
        if full.asset.duplicated().any():
            raise ValueError("Duplicate outcome quote")
        outcomes = full.set_index("asset").reindex(universe)
        required = current.ne(0) | held.ne(0)
        active_invalid = current.ne(0) & (
            outcomes.target.isna()
            | ~outcomes.target_valid.eq(True).fillna(False)
            | ~outcomes.entry_buyable.eq(True).fillna(False)
            | ~outcomes.exit_sellable.eq(True).fillna(False)
        )
        liquidation_invalid = (
            held.ne(0) & current.eq(0) & ~outcomes.entry_sellable.eq(True).fillna(False)
        )
        invalid = active_invalid | liquidation_invalid
        invalid |= required & ~outcomes.entry_raw_open.gt(0).fillna(False)
        if invalid.loc[required].any():
            blocked = True
            if not blocked_reason:
                blocked_reason = "missing_or_unexecutable_required_quote"
                blocked_assets = ";".join(universe[invalid & required].astype(str))
        row = {
            "date": date,
            "entry_date": entry,
            "exit_date": exit_date,
            "score": score,
            "cost_bps_per_full_l1": cost_bps,
            "gross_return": np.nan,
            "net_return": np.nan,
            "cost": np.nan,
            "turnover_full_l1": np.nan,
            "capital_ratio": np.nan,
            "nav": np.nan,
            "benchmark_return": np.nan,
            "status": "blocked_unknown_holdings_or_execution",
            "blocked_reason": blocked_reason,
            "blocked_assets": blocked_assets,
        }
        if not blocked:
            returns = outcomes.target.where(current.ne(0), 0).to_numpy(float)
            account = account_period(current.to_numpy(), held.to_numpy(), returns, cost)
            nav *= 1 + account.net_return
            previous = pd.Series(account.end_weights, index=universe)
            row.update(
                gross_return=account.gross_return,
                net_return=account.net_return,
                cost=account.transaction_cost,
                turnover_full_l1=account.turnover,
                capital_ratio=account.capital_ratio,
                nav=nav,
                status="full_fill_research_proxy",
            )
            if section.target_valid.all():
                row["benchmark_return"] = float(section.target.mean())
        previous_exit = exit_date
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(weights, ignore_index=True)
