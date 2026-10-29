"""Predeclared one-at-a-time experiments, five stresses, and local weight sensitivity."""

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from risk_aware_portfolio.attribution.performance import summarize_performance
from risk_aware_portfolio.backtest.engine import BacktestResult, run_backtest
from risk_aware_portfolio.costs.transaction_cost import TransactionCost
from risk_aware_portfolio.portfolio.baseline import equal_weight_benchmark
from risk_aware_portfolio.portfolio.optimizer import OptimizationError, optimize_portfolio
from risk_aware_portfolio.preprocessing.neutralization import preprocess_alpha
from risk_aware_portfolio.risk.covariance import build_asset_covariance, ensure_psd
from risk_aware_portfolio.risk.model import RiskHistory, estimate_risk_history
from risk_aware_portfolio.utils.config import Config, variant

logger = logging.getLogger("risk_aware_portfolio")


def experiment_configs(config: Config) -> list[tuple[str, str, Config]]:
    """Return the full grid from configuration; do not select best settings."""
    e = config["experiments"]
    result = []
    for section, key, values in (
        ("portfolio", "risk_aversion", e["risk_aversion"]),
        ("portfolio", "cost_penalty", e["cost_penalty"]),
        ("portfolio", "max_weight", e["position_limits"]),
        ("backtest", "rebalance_frequency", e["rebalance_frequencies"]),
    ):
        for value in values:
            result.append((f"{key}_{value}", key, variant(config, **{section: {key: value}})))
    for method in e["covariance_methods"]:
        result.append(
            (
                f"covariance_{method}",
                "covariance",
                variant(
                    config, risk_model={"covariance_method": method, "specific_method": method}
                ),
            )
        )
    for mode in e["alpha_modes"]:
        result.append(
            (f"alpha_{mode}", "alpha", variant(config, alpha={"neutralize": mode == "neutralized"}))
        )
    return result


def perturb_covariance(history: RiskHistory, amount: float) -> RiskHistory:
    """Mix factor covariance with its diagonal, preserving PSD and decomposition."""
    snapshots = {}
    if not 0 <= amount <= 1:
        raise ValueError("PSD-preserving perturbation must lie in [0, 1]")
    for date, snapshot in history.snapshots.items():
        f = snapshot.factor_covariance
        values = (1 - amount) * f.to_numpy() + amount * np.diag(np.diag(f))
        values, factor_audit = ensure_psd(values)
        changed = pd.DataFrame(values, index=f.index, columns=f.columns)
        covariance, audit = build_asset_covariance(
            snapshot.exposures, changed, snapshot.specific_variance
        )
        snapshots[date] = replace(
            snapshot,
            factor_covariance=changed,
            covariance=covariance,
            audit=audit,
            factor_audit=factor_audit,
        )
    return replace(history, snapshots=snapshots)


def _stress_cases(panel: pd.DataFrame, history: RiskHistory, config: Config) -> list[tuple]:
    e = config["experiments"]
    rng = np.random.default_rng(e["stress_seed"])
    noisy = panel.copy()
    noisy["alpha"] += rng.normal(0, e["alpha_noise_std"], len(panel))
    shocked = panel.copy()
    event_date = panel["date"].unique()[int(panel["date"].nunique() * 2 / 3)]
    shocked.loc[shocked["date"] >= event_date, "return"] *= e["volatility_multiplier"]
    cost = config["transaction_cost"]
    expensive = TransactionCost(
        **{
            **cost,
            "linear_bps": cost["linear_bps"] * e["cost_multiplier"],
            "impact_coefficient": cost["impact_coefficient"] * e["cost_multiplier"],
        }
    )
    illiquid = TransactionCost(
        **{**cost, "impact_coefficient": cost["impact_coefficient"] * e["liquidity_multiplier"]}
    )
    return [
        ("alpha_noise", noisy, history, None),
        (
            "covariance_perturbation",
            panel,
            perturb_covariance(history, e["covariance_perturbation"]),
            None,
        ),
        ("transaction_cost_increase", panel, history, expensive),
        ("liquidity_shock", panel, history, illiquid),
        ("volatility_shock", shocked, None, None),
    ]


def _metrics(result: BacktestResult, reference: BacktestResult, config: Config) -> dict[str, float]:
    common = reference.weights.index.intersection(result.weights.index)
    distance = (result.weights.loc[common] - reference.weights.loc[common]).abs().sum(axis=1)
    metrics = summarize_performance(
        result.returns, result.weights, config["backtest"]["annualization"], uncertainty=False
    )
    metrics.update(
        {
            "mean_weight_instability_l1": float(distance.mean()),
            "maximum_weight_instability_l1": float(distance.max()),
            "inaccurate_solver_fraction": float(
                (result.solver_diagnostics["status"] == "optimal_inaccurate").mean()
            ),
            "mean_covariance_condition_number": float(
                result.returns["covariance_condition_number"].mean()
            ),
        }
    )
    return metrics


def run_robustness(
    panel: pd.DataFrame,
    config: Config,
    reference: BacktestResult,
    history: RiskHistory,
    output: Path,
) -> pd.DataFrame:
    """Run every declared experiment and stress, retaining failures and exact configs."""
    output.mkdir(parents=True, exist_ok=True)
    declarations = experiment_configs(config)
    cases = [
        (name, category, panel, settings, None, None) for name, category, settings in declarations
    ]
    cases += [
        (name, "stress", data, config, risk, cost)
        for name, data, risk, cost in _stress_cases(panel, history, config)
    ]
    records, manifests = [], []
    for name, category, data, settings, supplied_history, cost in cases:
        print(f"Research experiment: {name}", flush=True)
        manifests.append(
            {
                "name": name,
                "category": category,
                "config": settings,
                "stress": config["experiments"] if category == "stress" else None,
            }
        )
        try:
            reusable = (
                history
                if settings["risk_model"] == config["risk_model"] and category != "stress"
                else None
            )
            risk = supplied_history or reusable or estimate_risk_history(data, settings)
            result = run_backtest(data, settings, name, history=risk, cost_override=cost)
            result.save(output / name)
            records.append(
                {
                    "experiment": name,
                    "category": category,
                    "status": "completed",
                    **_metrics(result, reference, settings),
                }
            )
        except (ValueError, OptimizationError) as error:
            logger.exception("Experiment %s failed", name)
            records.append(
                {"experiment": name, "category": category, "status": "failed", "error": str(error)}
            )
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifests, indent=2), encoding="utf-8"
    )
    return pd.DataFrame(records)


def local_weight_sensitivity(
    panel: pd.DataFrame, config: Config, reference: BacktestResult, history: RiskHistory
) -> pd.DataFrame:
    """Perturb alpha by a small fixed amount, freezing risk and prior holdings.

    Unlike full-path stress distance, this isolates optimizer sensitivity.
    It does not mix the perturbation with accumulated turnover/drift changes.
    """
    rng = np.random.default_rng(config["experiments"]["stress_seed"])
    rows = []
    dates = reference.returns.loc[reference.returns["rebalance"], "decision_date"]
    for date in dates:
        section = panel[panel["date"].eq(date)]
        alpha = preprocess_alpha(section, config)["selected_alpha"]
        noise = rng.normal(
            0,
            config["alpha"]["scale"] * config["experiments"]["local_alpha_noise_fraction"],
            len(alpha),
        )
        benchmark = equal_weight_benchmark(alpha.index, config["portfolio"]["long_only"])
        changed = optimize_portfolio(
            alpha + noise,
            history.snapshots[date],
            reference.pretrade_weights.loc[date],
            benchmark,
            config,
            TransactionCost(**config["transaction_cost"]).at_date(date),
            initial=date == reference.weights.index[0],
        )
        delta = float((changed.weights - reference.weights.loc[date]).abs().sum())
        rows.append(
            {
                "date": date,
                "alpha_perturbation_l1": np.abs(noise).sum(),
                "weight_sensitivity_l1": delta,
                "sensitivity_ratio": delta / np.abs(noise).sum(),
                "covariance_condition_number": history.snapshots[date].audit.metrics()[
                    "condition_number"
                ],
                "solver_status": changed.status,
            }
        )
    return pd.DataFrame(rows).set_index("date")
