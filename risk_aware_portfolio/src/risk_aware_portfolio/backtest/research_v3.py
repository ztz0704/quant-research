"""Versioned open-event A/B/C/D/E comparison using the existing optimizer.

Rank rules use current scores only. Optimized rules use historical calibrated
decimal returns and matured risk. Failed decisions retain explicit state/orders;
they never become a secretly relaxed portfolio. All coverage stays on calendar.
"""

from __future__ import annotations

import json
import os
import platform
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from risk_aware_portfolio.backtest.open_events import (
    INTERFACE_VERSION,
    _mark,
    run_open_events,
    validate_event_input,
)
from risk_aware_portfolio.costs.transaction_cost import TransactionCost
from risk_aware_portfolio.portfolio.constraints import (
    ConstraintSpec,
    check_constraints,
    constraint_exposures,
)
from risk_aware_portfolio.portfolio.optimizer import OptimizationError, optimize_portfolio
from risk_aware_portfolio.risk.open_model import OPEN_CONVENTION, fit_open_history, open_snapshot
from risk_aware_portfolio.utils.config import validate_config


def validate_signals(signals: pd.DataFrame, panel: pd.DataFrame, dates: pd.DatetimeIndex) -> None:
    fields = {
        "date",
        "asset",
        "model",
        "score",
        "expected_return",
        "rev",
        "feature_asof",
        "decision_time",
        "model_fit_cutoff",
        "estimator_fit_cutoff",
        "calibration_fit_cutoff",
        "max_fit_label_available",
        "max_calibration_label_available",
        "return_convention",
        "return_unit",
        "model_version",
    }
    if fields - set(signals):
        raise ValueError("V3 signal schema mismatch")
    if signals.duplicated(["date", "asset", "model"]).any():
        raise ValueError("Duplicate V3 signal identity")
    if set(signals.model) != {"ridge", "boosting"}:
        raise ValueError("Exactly the frozen two models are required")
    if (
        not signals.return_convention.eq(OPEN_CONVENTION).all()
        or not signals.return_unit.eq("decimal_simple").all()
    ):
        raise ValueError("Incompatible V3 signal units / interval")
    for name in ("feature_asof", "model_fit_cutoff", "calibration_fit_cutoff"):
        if signals[name].isna().any() or signals[name].gt(signals.decision_time).any():
            raise ValueError(f"Future/unknown signal provenance: {name}")
    if not signals.max_fit_label_available.lt(signals.estimator_fit_cutoff).all() or not (
        signals.max_calibration_label_available.lt(signals.calibration_fit_cutoff).all()
    ):
        raise ValueError("Immature model/calibration labels")
    if not np.isfinite(signals[["score", "expected_return", "rev"]]).all().all():
        raise ValueError("Nonfinite V3 forecasts")
    keys = pd.MultiIndex.from_frame(panel[["date", "asset"]])
    if not pd.MultiIndex.from_frame(signals[["date", "asset"]]).isin(keys).all():
        raise ValueError("Signal security absent from event panel")
    if not pd.DatetimeIndex(signals.date.unique()).isin(dates).all():
        raise ValueError("Signal outside declared OOS calendar")
    for _, day in signals.groupby("date"):
        a = set(day.loc[day.model.eq("ridge"), "asset"])
        b = set(day.loc[day.model.eq("boosting"), "asset"])
        if a != b:
            raise ValueError("V3 portfolios require identical model support")
    joined = signals[["date", "asset"]].merge(
        panel[["date", "asset", "eligible"]], on=["date", "asset"], validate="many_to_one"
    )
    if not joined.eligible.all():
        raise ValueError("Signal included a contemporaneously ineligible security")


def performance(periods: pd.DataFrame, benchmark: pd.Series, annualization: int = 252) -> dict:
    returns = periods.set_index("date").net_return
    bench = benchmark.reindex(returns.index)
    comparable = returns.notna() & bench.notna()
    complete = bool(comparable.all())
    result = {
        "periods": len(returns),
        "valid_return_dates": int(returns.notna().sum()),
        "nav_valid_dates": int(periods.nav_valid.sum()),
        "missing_valuation_dates": int((~periods.nav_valid).sum()),
        "paired_benchmark_dates": int(comparable.sum()),
        "return_coverage": float(returns.notna().mean()),
        "full_comparable_window": complete,
        "performance_status": "complete_scenario" if complete else "BLOCKED_incomplete_window",
        "cagr": np.nan,
        "annual_volatility": np.nan,
        "sharpe": np.nan,
        "max_drawdown": np.nan,
        "annual_active_mean": np.nan,
        "tracking_error_annual": np.nan,
        "information_ratio": np.nan,
    }
    if complete:
        growth = (1 + returns).cumprod()
        if not np.allclose(growth.to_numpy(), periods.nav.to_numpy(), atol=1e-9):
            raise AssertionError("Continuous return chain disagrees with event NAV")
        active, sd = returns - bench, returns.std(ddof=1)
        active_sd = active.std(ddof=1)
        peaks = np.maximum.accumulate(np.r_[1.0, growth])
        result.update(
            cagr=float(growth.iloc[-1] ** (annualization / len(returns)) - 1),
            annual_volatility=float(sd * np.sqrt(annualization)),
            sharpe=float(returns.mean() / sd * np.sqrt(annualization)) if sd > 0 else np.nan,
            max_drawdown=float((np.r_[1.0, growth] / peaks - 1).min()),
            annual_active_mean=float(active.mean() * annualization),
            tracking_error_annual=float(active_sd * np.sqrt(annualization)),
            information_ratio=float(active.mean() / active_sd * np.sqrt(annualization))
            if active_sd > 0
            else np.nan,
        )
    return result


def validate_open_labels(panel: pd.DataFrame, calendar: pd.DataFrame) -> None:
    """Prove supplied labels describe exact exchange-calendar open intervals."""
    if {"research_open", "label_start", "label_end", "target_valid", "target"} - set(panel):
        raise ValueError("Open label verification schema mismatch")
    indexed = panel.set_index(["date", "asset"]).research_open
    maps = {}
    for exchange, group in calendar.sort_values("date").groupby("exchange"):
        dates = list(group.date)
        maps.update({(exchange, d): (dates[i + 1], dates[i + 2]) for i, d in enumerate(dates[:-2])})
    valid = panel.loc[panel.target_valid & panel.target.notna()]
    entries, exits = [], []
    for row in valid.itertuples():
        expected = maps.get((row.exchange, row.date))
        if expected != (row.label_start, row.label_end):
            raise ValueError("Shifted/renamed open interval is forbidden")
        entries.append(indexed.get((row.label_start, row.asset), np.nan))
        exits.append(indexed.get((row.label_end, row.asset), np.nan))
    expected = np.asarray(exits) / np.asarray(entries) - 1
    if not np.isfinite(expected).all() or not np.allclose(
        expected, valid.target, atol=1e-12, rtol=1e-10
    ):
        raise ValueError("Risk label does not match adjusted raw open interval")


def run_portfolios(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    signals: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict,
    settings: dict,
) -> dict[str, pd.DataFrame]:
    validate_event_input(panel, calendar)
    validate_open_labels(panel, calendar)
    validate_config(config)
    validate_signals(signals, panel, dates)
    history = fit_open_history(panel, config)
    groups = {
        (name, date): day.set_index("asset").sort_index()
        for (name, date), day in signals.groupby(["model", "date"])
    }
    snapshots, snapshot_errors, risk_audits = {}, {}, []
    for date in dates:
        day = groups.get(("ridge", date))
        if day is None:
            snapshot_errors[date] = "no_common_predictions"
            continue
        try:
            snapshots[date], audit = open_snapshot(history, date, day.index, config)
            risk_audits.append(audit)
        except ValueError as error:
            snapshot_errors[date] = str(error)
            risk_audits.append({"date": date, "status": "BLOCKED", "reason": str(error)})
    names = {
        "A_REV": ("ridge", "rev", False, 0.0),
        "B_Ridge": ("ridge", "score", False, 0.0),
        "C_HGB": ("boosting", "score", False, 0.0),
        "D_HGB_risk": ("boosting", "expected_return", True, 0.0),
        "E_HGB_risk_cost": ("boosting", "expected_return", True, 1.0),
        "R_Ridge_risk_cost": ("ridge", "expected_return", True, 1.0),
        "Benchmark_equal": ("ridge", None, False, 0.0),
    }
    collections = {
        name: [] for name in ("events", "holdings", "orders", "targets", "decisions", "periods")
    }
    summary_rows, attribution, exposures, contributions, risk_calibration = [], [], [], [], []
    actual_constraints = []
    benchmark = None
    # The reference uses the identical current pool and event rules, no fees.
    ordered = ["Benchmark_equal", *[n for n in names if n != "Benchmark_equal"]]
    for strategy in ordered:
        model, rank_column, optimized, penalty = names[strategy]
        scenarios = (
            [0] if strategy == "Benchmark_equal" else settings["portfolio"]["cost_bps_per_full_l1"]
        )
        for bps in scenarios:
            local = deepcopy(config)
            local["portfolio"]["cost_penalty"] = penalty
            cost = TransactionCost(linear_bps=bps, impact_coefficient=0.0)
            optimization_context = {}

            def build(date, rows, state):
                day = groups.get((model, date))
                if day is None or day.empty:
                    return None, {"status": "BLOCKED", "reason": "no_current_common_predictions"}
                if not optimized:
                    if rank_column is None:
                        weights = pd.Series(1 / len(day), index=day.index)
                    else:
                        selected = day.sort_values(
                            [rank_column, "asset"], ascending=[False, True]
                        ).head(max(1, int(np.ceil(len(day) * settings["portfolio"]["quantile"]))))
                        weights = pd.Series(1 / len(selected), index=selected.index)
                    return weights, {
                        "status": "rank_target",
                        "forecast_column": rank_column or "equal",
                    }
                if date not in snapshots:
                    return None, {"status": "BLOCKED", "reason": snapshot_errors[date]}
                nav, reason = _mark(state, rows, "raw_close")
                held = {a for a, q in state.quantities.items() if q > 1e-10}
                if not np.isfinite(nav) or not held.issubset(day.index):
                    return None, {
                        "status": "BLOCKED",
                        "reason": reason or "held_asset_outside_current_risk_support",
                    }
                previous = pd.Series(
                    {
                        a: state.quantities.get(a, 0.0) * rows.loc[a, "raw_close"] / nav
                        if state.quantities.get(a, 0.0) > 1e-10
                        else 0.0
                        for a in day.index
                    }
                )
                benchmark_weights = pd.Series(1 / len(day), index=day.index)
                try:
                    optimization_context[date] = (
                        previous,
                        benchmark_weights,
                        state.trade_count == 0,
                    )
                    solution = optimize_portfolio(
                        day.expected_return,
                        snapshots[date],
                        previous,
                        benchmark_weights,
                        local,
                        cost,
                        initial=state.trade_count == 0,
                    )
                    # Numerical negatives within solver tolerance are NOT clipped.
                    # The event engine needs genuinely nonnegative quantities.
                    if solution.weights.min() < 0:
                        raise OptimizationError(
                            "numerically_negative_weights_rejected_without_clipping"
                        )
                    return solution.weights, {
                        "status": "optimized",
                        "solver_status": solution.status,
                        "objective": solution.objective,
                        "predicted_target_cost": solution.predicted_cost,
                        "binding_constraints": solution.binding_count,
                        **{f"slack_{k}": v for k, v in solution.slacks.items()},
                    }
                except (OptimizationError, ValueError) as error:
                    return None, {
                        "status": "BLOCKED",
                        "solver_status": "failed",
                        "reason": str(error),
                        "constraints_relaxed": False,
                    }

            result = run_open_events(
                panel,
                calendar,
                dates,
                build,
                cost,
                settings["event_portfolio"]["initial_cash"],
                settings["event_portfolio"]["terminal_liquidation"],
            )
            for name in collections:
                frame = getattr(result, name).copy()
                frame["strategy"], frame["cost_bps_per_full_l1"] = strategy, bps
                collections[name].append(frame)
            if strategy == "Benchmark_equal":
                benchmark = result.periods.set_index("date").net_return
            summary = performance(result.periods, benchmark, config["backtest"]["annualization"])
            log = result.orders
            summary.update(
                strategy=strategy,
                cost_bps_per_full_l1=bps,
                actual_trade_amount=float(result.events.trade_amount.sum()),
                actual_cost_cash=float(result.events.fees.sum()),
                mean_turnover_full_l1=float(result.periods.turnover_full_l1.mean()),
                pending_order_retained_events=int(log.event.eq("pending_order_retained").sum()),
                rejected_order_events=int(log.event.str.startswith("rejected_").sum()),
                unfilled_sell_events=int(
                    (
                        log.event.str.startswith("rejected_")
                        & log.get("side", pd.Series(index=log.index, dtype=str)).eq("sell")
                    ).sum()
                ),
                carried_inventory_rows=int(
                    (result.holdings.pending_quantity.lt(0) & result.holdings.quantity.gt(0)).sum()
                )
                if not result.holdings.empty
                else 0,
                final_cash=result.final_state.cash,
                terminal_unliquidated_assets=sum(
                    q > 1e-10 for q in result.final_state.quantities.values()
                ),
                unresolved_action_assets=";".join(sorted(result.final_state.uncertain_actions)),
                optimized_decisions=int(result.decisions.status.eq("optimized").sum()),
                blocked_decisions=int(result.decisions.status.eq("BLOCKED").sum()),
            )
            summary_rows.append(summary)
            # Keep all dates, including unvalued periods. Attribution never fills a gap.
            for _, period in result.periods.iterrows():
                date = period.date
                row = {
                    "date": date,
                    "strategy": strategy,
                    "cost_bps_per_full_l1": bps,
                    "net_return": period.net_return,
                    "cost": period.cost,
                    "gross_return": period.gross_return,
                    "factor_return": np.nan,
                    "specific_return": np.nan,
                    "basis_return": np.nan,
                    "status": "BLOCKED",
                }
                snapshot = snapshots.get(date)
                holdings = result.holdings.loc[
                    result.holdings.date.eq(period.entry_date) & result.holdings.quantity.gt(1e-10)
                ].set_index("asset")
                entry = result.events.loc[result.events.date.eq(period.entry_date)].iloc[0]
                if (
                    snapshot is not None
                    and np.isfinite(period.net_return)
                    and holdings.index.isin(snapshot.exposures.index).all()
                ):
                    weights = (holdings.quantity * holdings.raw_open / entry.pre_nav).reindex(
                        snapshot.exposures.index, fill_value=0.0
                    )
                    x = snapshot.exposures
                    factor_weight = x.T @ weights
                    covariance = snapshot.covariance.to_numpy()
                    w = weights.to_numpy()
                    variance = float(w @ covariance @ w)
                    if optimized and date in optimization_context:
                        previous, bench_weights, initial = optimization_context[date]
                        actual = weights * entry.pre_nav / entry.post_nav
                        industry, style = constraint_exposures(snapshot, local)
                        audit = {
                            "date": date,
                            "strategy": strategy,
                            "cost_bps_per_full_l1": bps,
                            "cash_weight": entry.cash / entry.post_nav,
                            "asset_weight_sum": actual.sum(),
                            "max_actual_weight": actual.max(),
                            "check_basis": "actual_open_fills_vs_close_decision_constraints",
                        }
                        try:
                            slacks, _ = check_constraints(
                                actual.to_numpy(),
                                previous.to_numpy(),
                                bench_weights.to_numpy(),
                                covariance,
                                industry,
                                style,
                                ConstraintSpec.from_config(local, initial),
                                local["portfolio"]["constraint_tolerance"],
                            )
                            audit.update(
                                status="within_constraints",
                                **{f"slack_{k}": v for k, v in slacks.items()},
                            )
                        except ValueError as error:
                            audit.update(status="fill_or_price_drift_violation", reason=str(error))
                        actual_constraints.append(audit)
                    active = w - np.full(len(w), 1 / len(w))
                    risk_calibration.append(
                        {
                            "date": date,
                            "strategy": strategy,
                            "cost_bps_per_full_l1": bps,
                            "predicted_variance": variance,
                            "realized_gross_return": period.gross_return,
                            "realized_net_return": period.net_return,
                            "predicted_active_variance": float(active @ covariance @ active),
                            "realized_active_return": period.net_return
                            - benchmark.get(date, np.nan),
                        }
                    )
                    for name, value in factor_weight.items():
                        exposures.append(
                            {
                                "date": date,
                                "strategy": strategy,
                                "cost_bps_per_full_l1": bps,
                                "factor": name,
                                "actual_entry_exposure": value,
                            }
                        )
                    factor_marginal = snapshot.factor_covariance @ factor_weight
                    for name in factor_weight.index:
                        contributions.append(
                            {
                                "date": date,
                                "strategy": strategy,
                                "cost_bps_per_full_l1": bps,
                                "component": name,
                                "variance_contribution": factor_weight[name]
                                * factor_marginal[name],
                            }
                        )
                    contributions.append(
                        {
                            "date": date,
                            "strategy": strategy,
                            "cost_bps_per_full_l1": bps,
                            "component": "specific",
                            "variance_contribution": float(
                                (weights.pow(2) * snapshot.specific_variance).sum()
                            ),
                        }
                    )
                    if (
                        date in history.factor_returns.index
                        and holdings.index.isin(history.specific_returns.columns).all()
                    ):
                        specific = history.specific_returns.loc[date].reindex(weights.index)
                        if specific.loc[weights.gt(1e-10)].notna().all():
                            factor_return = float(factor_weight @ history.factor_returns.loc[date])
                            specific_return = float((weights * specific.fillna(0.0)).sum())
                            # Zero weights on unheld names only; no held missing residual is filled.
                            basis = period.gross_return - factor_return - specific_return
                            row.update(
                                status="reconciled_with_explicit_basis",
                                factor_return=factor_return,
                                specific_return=specific_return,
                                basis_return=basis,
                            )
                            if not np.isclose(
                                factor_return + specific_return + basis - period.cost,
                                period.net_return,
                                atol=1e-12,
                            ):
                                raise AssertionError(
                                    "Factor/specific/basis/fee reconciliation failed"
                                )
                attribution.append(row)
    tables = {}
    for name, parts in collections.items():
        nonempty = [part for part in parts if not part.empty]
        tables[f"portfolio_{name}"] = (
            pd.concat(nonempty, ignore_index=True) if nonempty else parts[0].iloc[:0].copy()
        )
    tables.update(
        portfolio_performance=pd.DataFrame(summary_rows),
        open_risk_regression_diagnostics=history.diagnostics,
        open_risk_snapshot_audit=pd.DataFrame(risk_audits),
        portfolio_attribution=pd.DataFrame(attribution),
        portfolio_factor_exposures=pd.DataFrame(exposures),
        portfolio_risk_contributions=pd.DataFrame(contributions),
        risk_calibration_daily=pd.DataFrame(risk_calibration),
        actual_fill_constraint_audit=pd.DataFrame(actual_constraints),
    )
    sensitivities = []
    anchors = sorted({dates[0], dates[len(dates) // 2], dates[-1]})
    for date in anchors:
        day = groups.get(("boosting", date))
        if day is None:
            continue
        bench = pd.Series(1 / len(day), index=day.index)
        for method in ("sample", "ewma"):
            for aversion in (1.0, 5.0, 20.0):
                local = deepcopy(config)
                local["risk_model"]["covariance_method"] = method
                local["portfolio"].update(risk_aversion=aversion, cost_penalty=1.0)
                record = {
                    "date": date,
                    "covariance_method": method,
                    "risk_aversion": aversion,
                    "previous_weight_scenario": "equal_current_benchmark",
                    "scope": "exploratory_target_only_not_new_nav_experiment",
                }
                try:
                    snapshot, _ = open_snapshot(history, date, day.index, local)
                    solution = optimize_portfolio(
                        day.expected_return,
                        snapshot,
                        bench,
                        bench,
                        local,
                        TransactionCost(linear_bps=10.0, impact_coefficient=0.0),
                    )
                    active = solution.weights - bench
                    record.update(
                        status=solution.status,
                        objective=solution.objective,
                        target_turnover_full_l1=float(active.abs().sum()),
                        target_expected_return=float(solution.weights @ day.expected_return),
                        target_variance=float(
                            solution.weights @ snapshot.covariance @ solution.weights
                        ),
                        min_slack=min(solution.slacks.values()),
                    )
                except (ValueError, OptimizationError) as error:
                    record.update(status="BLOCKED", reason=str(error))
                sensitivities.append(record)
    tables["risk_parameter_target_sensitivity"] = pd.DataFrame(sensitivities)
    return tables


def run_bundle(bundle: Path) -> Path:
    """Read a SHA-bound v3 contract, validate it, save a separately hashed result."""
    import hashlib

    contract = json.loads((bundle / "contract.json").read_text(encoding="utf-8"))
    if contract["interface_version"] != INTERFACE_VERSION:
        raise ValueError("Unsupported V3 Risk bridge version")
    if set(contract["inputs"]) != {"panel.parquet", "calendar.parquet", "signals.parquet"}:
        raise ValueError("Incomplete V3 input hash contract")
    projects = Path(__file__).resolve().parents[4]
    for name, expected in contract["implementation_hashes"].items():
        source = (projects / name).resolve()
        if (
            not source.is_relative_to(projects)
            or hashlib.sha256(source.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Risk bridge stale implementation: {name}")
    for name, expected in contract["inputs"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Risk bridge stale/unsafe input: {name}")
    destination = Path(contract["output_directory"])
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "risk_manifest.json"
    manifest = {
        "interface_version": INTERFACE_VERSION,
        "status": "running",
        "contract": contract,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "cvxpy", "clarabel")},
        "formal_empirical": False,
        "numerical_threads": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
    }
    try:
        panel = pd.read_parquet(bundle / "panel.parquet")
        calendar = pd.read_parquet(bundle / "calendar.parquet")
        signals = pd.read_parquet(bundle / "signals.parquet")
        dates = pd.DatetimeIndex(contract["decision_dates"])
        tables = run_portfolios(
            panel, calendar, signals, dates, contract["risk_config"], contract["settings"]
        )
        for name, frame in tables.items():
            frame.to_parquet(destination / f"{name}.parquet", index=False)
            frame.to_csv(destination / f"{name}.csv", index=False)
        manifest.update(
            status="complete",
            artifacts={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(destination.iterdir())
                if p.is_file() and p != manifest_path
            },
        )
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    return destination
