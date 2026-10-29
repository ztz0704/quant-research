"""Walk-forward portfolio construction with explicit decision/realization dates."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from risk_aware_portfolio.backtest.accounting import PeriodAccounting, account_period
from risk_aware_portfolio.costs.transaction_cost import TransactionCost
from risk_aware_portfolio.data.signal_contract import require_engine_compatible
from risk_aware_portfolio.portfolio.baseline import equal_weight_benchmark, rank_portfolio
from risk_aware_portfolio.portfolio.optimizer import OptimizationResult, optimize_portfolio
from risk_aware_portfolio.portfolio.rebalance import is_rebalance
from risk_aware_portfolio.preprocessing.neutralization import preprocess_alpha
from risk_aware_portfolio.risk.model import RiskHistory, RiskSnapshot, estimate_risk_history
from risk_aware_portfolio.utils.config import Config

logger = logging.getLogger("risk_aware_portfolio")


@dataclass
class BacktestResult:
    name: str
    weights: pd.DataFrame
    pretrade_weights: pd.DataFrame
    returns: pd.DataFrame
    factor_exposures: pd.DataFrame
    alphas: pd.DataFrame
    attribution: pd.DataFrame
    risk_contributions: pd.DataFrame
    solver_diagnostics: pd.DataFrame

    def save(self, directory: Path) -> None:
        """Save labeled ledgers; no object pickle is needed to inspect results."""
        directory.mkdir(parents=True, exist_ok=True)
        for field_name in (
            "weights",
            "pretrade_weights",
            "returns",
            "factor_exposures",
            "alphas",
            "attribution",
            "risk_contributions",
            "solver_diagnostics",
        ):
            getattr(self, field_name).to_parquet(directory / f"{field_name}.parquet")

    @classmethod
    def load(cls, name: str, directory: Path) -> "BacktestResult":
        fields = (
            "weights",
            "pretrade_weights",
            "returns",
            "factor_exposures",
            "alphas",
            "attribution",
            "risk_contributions",
            "solver_diagnostics",
        )
        return cls(name, **{key: pd.read_parquet(directory / f"{key}.parquet") for key in fields})


@dataclass
class _WalkState:
    weights: pd.Series
    nav: float = 1.0
    gross_nav: float = 1.0
    previous_date: pd.Timestamp | None = None
    pending_rebalance: bool = False
    rows: dict[str, list] = field(
        default_factory=lambda: {
            key: []
            for key in (
                "weights",
                "pretrade_weights",
                "returns",
                "factor_exposures",
                "alphas",
                "attribution",
                "risk_contributions",
                "solver_diagnostics",
            )
        }
    )


def _target(
    alpha: pd.Series,
    snapshot: RiskSnapshot,
    state: _WalkState,
    benchmark: pd.Series,
    config: Config,
    cost: TransactionCost,
    baseline: bool,
    rebalance: bool,
) -> tuple[pd.Series, OptimizationResult | None]:
    if not rebalance:
        return state.weights.copy(), None
    if baseline:
        p = config["portfolio"]
        return rank_portfolio(alpha, p["baseline_quantile"], p["long_only"]), None
    result = optimize_portfolio(
        alpha, snapshot, state.weights, benchmark, config, cost, initial=state.previous_date is None
    )
    return result.weights, result


def _risk_ledger(
    weights: pd.Series, benchmark: pd.Series, snapshot: RiskSnapshot
) -> dict[str, float]:
    w, x, f = (
        weights.to_numpy(),
        snapshot.exposures.to_numpy(),
        snapshot.factor_covariance.to_numpy(),
    )
    exposure = x.T @ w
    factor_variance = float(exposure @ f @ exposure)
    specific_variance = float(w**2 @ snapshot.specific_variance.to_numpy())
    total = float(w @ snapshot.covariance.to_numpy() @ w)
    active = w - benchmark.to_numpy()
    return {
        "total_variance": total,
        "factor_variance": factor_variance,
        "specific_variance": specific_variance,
        "repair_variance_adjustment": total - factor_variance - specific_variance,
        "active_variance": float(active @ snapshot.covariance.to_numpy() @ active),
        "factor_risk_fraction": factor_variance / total if total > 0 else np.nan,
        "specific_risk_fraction": specific_variance / total if total > 0 else np.nan,
    }


def _attribution(
    target: pd.Series,
    benchmark: pd.Series,
    selected_alpha: pd.Series,
    snapshot: RiskSnapshot,
    history: RiskHistory,
    next_date: pd.Timestamp,
    accounting: PeriodAccounting,
) -> dict[str, float]:
    x, w, b = snapshot.exposures.to_numpy(), target.to_numpy(), benchmark.to_numpy()
    factors = history.factor_returns.loc[next_date].to_numpy()
    residual = history.specific_returns.loc[next_date].reindex(target.index).to_numpy()
    factor = float(w @ x @ factors)
    specific_total = float(w @ residual)
    alpha_forecast = float(w @ selected_alpha.to_numpy())
    q = accounting.capital_ratio
    # Alpha here is an ex-ante forecast; the remaining residual closes the identity.
    return {
        "factor": q * factor,
        "alpha_forecast": q * alpha_forecast,
        "specific_after_alpha": q * (specific_total - alpha_forecast),
        "specific_total": q * specific_total,
        "transaction_cost": -accounting.transaction_cost,
        "active_factor": float((q * w - b) @ x @ factors),
        "active_specific": float((q * w - b) @ residual),
    }


def _record(
    state: _WalkState,
    target: pd.Series,
    alpha: pd.DataFrame,
    snapshot: RiskSnapshot,
    benchmark: pd.Series,
    history: RiskHistory,
    next_date: pd.Timestamp,
    accounting: PeriodAccounting,
    solved: OptimizationResult | None,
    rebalance: bool,
    config: Config,
    cost: TransactionCost,
) -> None:
    risk = _risk_ledger(target, benchmark, snapshot)
    status = solved.status if solved else ("rank" if rebalance else "held")
    state.nav *= 1 + accounting.net_return
    state.gross_nav *= 1 + accounting.gross_return
    state.rows["weights"].append(target)
    state.rows["pretrade_weights"].append(state.weights.copy())
    state.rows["factor_exposures"].append(snapshot.exposures.T @ target)
    state.rows["alphas"].append(alpha.assign(date=snapshot.date).reset_index())
    state.rows["risk_contributions"].append(risk)
    state.rows["attribution"].append(
        _attribution(
            target, benchmark, alpha["selected_alpha"], snapshot, history, next_date, accounting
        )
    )
    state.rows["solver_diagnostics"].append(
        {
            "status": status,
            "binding_count": solved.binding_count if solved else 0,
            "objective": solved.objective if solved else np.nan,
            **(solved.slacks if solved else {}),
        }
    )
    row = {
        "decision_date": snapshot.date,
        "return_date": next_date,
        "gross_return": accounting.gross_return,
        "net_return": accounting.net_return,
        "benchmark_return": float(benchmark @ history_return(history, snapshot, next_date)),
        "turnover": accounting.turnover,
        "target_turnover": float((target - state.weights).abs().sum()) if rebalance else 0.0,
        "transaction_cost": accounting.transaction_cost,
        "sell_tax": float(cost.sell_bps / 1e4 * np.maximum(-accounting.trades, 0).sum()),
        "effective_sell_bps": cost.sell_bps,
        "predicted_cost": solved.predicted_cost if solved else np.nan,
        "nav": state.nav,
        "gross_nav": state.gross_nav,
        "rebalance": rebalance,
        "deferred_rebalance": state.pending_rebalance,
        "solver_status": status,
        "binding_count": solved.binding_count if solved else 0,
        "predicted_volatility": np.sqrt(
            risk["total_variance"] * config["backtest"]["annualization"]
        ),
        "covariance_condition_number": snapshot.audit.metrics()["condition_number"],
    }
    state.rows["returns"].append(row)
    logger.info(
        "date=%s assets=%d status=%s risk=%.5f turnover=%.5f cost=%.7f binding=%d",
        snapshot.date.date(),
        len(target),
        status,
        row["predicted_volatility"],
        accounting.turnover,
        accounting.transaction_cost,
        row["binding_count"],
    )


def history_return(
    history: RiskHistory, snapshot: RiskSnapshot, next_date: pd.Timestamp
) -> np.ndarray:
    """Reconstruct realized asset returns only for ex-post attribution/benchmark accounting."""
    return (
        snapshot.exposures.to_numpy() @ history.factor_returns.loc[next_date].to_numpy()
        + history.specific_returns.loc[next_date].reindex(snapshot.exposures.index).to_numpy()
    )


def _assemble(name: str, state: _WalkState) -> BacktestResult:
    dates = pd.DatetimeIndex([row["decision_date"] for row in state.rows["returns"]], name="date")
    realized_dates = pd.DatetimeIndex(
        [row["return_date"] for row in state.rows["returns"]], name="date"
    )
    frames = {}
    for key, rows in state.rows.items():
        if key == "alphas":
            frames[key] = pd.concat(rows, ignore_index=True)
        else:
            frames[key] = pd.DataFrame(
                rows, index=realized_dates if key in {"returns", "attribution"} else dates
            )
    return BacktestResult(name, **frames)


def run_backtest(
    panel: pd.DataFrame,
    config: Config,
    name: str = "Risk-Aware + TC",
    baseline: bool = False,
    history: RiskHistory | None = None,
    cost_override: TransactionCost | None = None,
) -> BacktestResult:
    """Construct weight_t before accessing return_(t+1); use drifted prior holdings."""
    require_engine_compatible(panel, config)
    history = history or estimate_risk_history(panel, config)
    sections = {date: section for date, section in panel.groupby("date", sort=True)}
    dates = list(sections)
    if not history.snapshots:
        raise ValueError("No risk forecasts after warmup")
    assets = next(iter(history.snapshots.values())).exposures.index
    state = _WalkState(pd.Series(0.0, index=assets))
    cost_model = cost_override or TransactionCost(**config["transaction_cost"])
    for date, next_date in zip(dates[:-1], dates[1:]):
        if date not in history.snapshots:
            continue
        evaluation_start = config["backtest"].get("evaluation_start")
        if evaluation_start and date < pd.Timestamp(evaluation_start):
            continue
        snapshot = history.snapshots[date]
        cost = cost_model.at_date(date)
        alpha = preprocess_alpha(sections[date], config)
        benchmark = equal_weight_benchmark(assets, config["portfolio"]["long_only"])
        due = state.pending_rebalance or is_rebalance(
            date, state.previous_date, config["backtest"]["rebalance_frequency"]
        )
        available = bool(sections[date]["tradable"].all()) if "tradable" in sections[date] else True
        rebalance = due and available
        state.pending_rebalance = due and not available
        if state.pending_rebalance:
            logger.warning("Rebalance deferred: known suspension at %s", date.date())
        target, solved = _target(
            alpha["selected_alpha"], snapshot, state, benchmark, config, cost, baseline, rebalance
        )
        # Only after target selection is the next-period return retrieved.
        realized = sections[next_date].set_index("asset")["return"].reindex(assets).to_numpy()
        accounting = account_period(
            target.to_numpy(), state.weights.to_numpy(), realized, cost, rebalance
        )
        _record(
            state,
            target,
            alpha,
            snapshot,
            benchmark,
            history,
            next_date,
            accounting,
            solved,
            rebalance,
            config,
            cost,
        )
        state.weights = pd.Series(accounting.end_weights, index=assets)
        state.previous_date = date
    if not state.rows["returns"]:
        raise ValueError("No realized periods after warmup")
    return _assemble(name, state)
