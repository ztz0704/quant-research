"""Self-financing single-period NAV and holdings accounting."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from risk_aware_portfolio.costs.transaction_cost import TransactionCost


@dataclass(frozen=True)
class PeriodAccounting:
    gross_return: float
    net_return: float
    transaction_cost: float
    capital_ratio: float
    turnover: float
    trades: np.ndarray
    end_weights: np.ndarray


def account_period(
    target: np.ndarray,
    pretrade: np.ndarray,
    returns: np.ndarray,
    cost: TransactionCost,
    rebalance: bool = True,
) -> PeriodAccounting:
    """Solve q + TC(q*target - pretrade) = 1, then earn q*(1+w'r)-1.

    target is relative to post-cost NAV; pretrade is relative to pre-trade NAV.
    This pays fees by reducing invested capital rather than creating hidden debt.
    Cash earns zero; shorts have explicit collateral cash but no borrow/funding cost.
    """
    target, pretrade, returns = (np.asarray(x, dtype=float) for x in (target, pretrade, returns))
    if target.shape != pretrade.shape or target.shape != returns.shape:
        raise ValueError("Accounting shape mismatch")
    if not np.isfinite(np.column_stack([target, pretrade, returns])).all():
        raise ValueError("Nonfinite accounting inputs")
    if (returns <= -1).any():
        raise ValueError("Simple asset return must exceed -1")
    if rebalance:
        if cost.realized(-pretrade) >= 1:
            raise ValueError("Fees exhaust portfolio capital")
        ratio = brentq(lambda q: q + cost.realized(q * target - pretrade) - 1, 0, 1, xtol=1e-14)
        trades = ratio * target - pretrade
        fees = cost.realized(trades)
    else:
        if not np.allclose(target, pretrade, atol=1e-12, rtol=0):
            raise ValueError("Non-rebalance target must equal current holdings")
        ratio, fees, trades = 1.0, 0.0, np.zeros_like(target)
    gross = float(target @ returns)
    if 1 + gross <= 0:
        raise ValueError("Portfolio insolvency; backtest aborted")
    net = ratio * (1 + gross) - 1
    drifted = target * (1 + returns) / (1 + gross)
    return PeriodAccounting(gross, net, fees, ratio, float(np.abs(trades).sum()), trades, drifted)
