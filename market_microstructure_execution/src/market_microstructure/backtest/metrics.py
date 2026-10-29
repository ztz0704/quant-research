"""Economic metrics with explicit holding, turnover and Sharpe definitions."""

import numpy as np

from market_microstructure.backtest.engine import BacktestResult


def performance_metrics(result: BacktestResult) -> dict:
    """Sharpe uses fixed-clock portfolio returns; unannualized unless explicitly configured."""
    events, fills, lots, config = result.events, result.fills, result.closed_lots, result.config
    final = events.iloc[-1]
    equity = events.groupby("timestamp").equity.last()
    equity = equity.resample(f"{config.sharpe_interval_ms}ms").last().ffill()
    returns = equity.diff().fillna(equity.iloc[0] - config.initial_cash) / config.initial_cash
    sharpe = returns.mean() / returns.std(ddof=1) if len(returns) > 1 and returns.std(ddof=1) > 0 else np.nan
    multiplier = np.sqrt(config.annualization_periods) if config.annualization_periods is not None else 1.0
    if config.annualization_periods is not None and config.annualization_periods <= 0:
        raise ValueError("Annualization periods must be positive")
    values = np.r_[config.initial_cash, equity.to_numpy()]
    drawdown = values - np.maximum.accumulate(values)
    trade_count = len(fills)
    traded_quantity = fills.quantity.abs().sum()
    remaining = events.groupby("asset").position.last()
    return {"gross_pnl": final.gross_pnl, "net_pnl": final.net_pnl,
            "realized_gross_alpha": final.realized_gross_alpha,
            "inventory_mark_to_market": final.inventory_mark_to_market,
            "pnl_per_fill": final.net_pnl / trade_count if trade_count else np.nan,
            "hit_rate": (lots.net_pnl > 0).mean() if not lots.empty else np.nan,
            "trade_count": trade_count, "closed_lot_count": len(lots),
            "turnover_notional": (fills.quantity.abs() * fills.price).sum(),
            "turnover_initial_equity": (fills.quantity.abs() * fills.price).sum() / config.initial_cash,
            "average_holding_events": np.average(lots.holding_events, weights=lots.quantity) if len(lots) else np.nan,
            "average_holding_ms": np.average(lots.holding_ms, weights=lots.quantity) if len(lots) else np.nan,
            "average_spread_paid_per_share": final.spread_cost / traded_quantity if traded_quantity else np.nan,
            "average_slippage_per_share": final.slippage / traded_quantity if traded_quantity else np.nan,
            "spread_cost": final.spread_cost, "total_fees": final.fees,
            "total_slippage": final.slippage, "market_impact": final.impact,
            "inventory_volatility": events.position.std(), "max_inventory": events.position.abs().max(),
            "residual_inventory_abs": remaining.abs().sum(), "max_drawdown": -drawdown.min(),
            "sharpe_ratio": sharpe * multiplier, "sharpe_annualized": config.annualization_periods is not None,
            "sharpe_interval_ms": config.sharpe_interval_ms,
            "net_pnl_reconciliation_error": final.net_pnl - (
                final.gross_pnl - final.spread_cost - final.fees - final.slippage - final.impact)}
