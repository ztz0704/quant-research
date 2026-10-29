"""Daily-unit mean-variance optimization with exact convex nonlinear trade penalty."""

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from risk_aware_portfolio.costs.transaction_cost import TransactionCost
from risk_aware_portfolio.portfolio.constraints import (
    ConstraintSpec,
    build_constraints,
    check_constraints,
    constraint_exposures,
)
from risk_aware_portfolio.risk.covariance import ensure_psd
from risk_aware_portfolio.risk.model import RiskSnapshot
from risk_aware_portfolio.utils.config import Config

logger = logging.getLogger("risk_aware_portfolio")


class OptimizationError(RuntimeError):
    """A solver failure or infeasibility must not silently become a held portfolio."""


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    status: str
    objective: float
    predicted_cost: float
    binding_count: int
    slacks: dict[str, float]


def optimize_portfolio(
    alpha: pd.Series,
    snapshot: RiskSnapshot,
    previous: pd.Series,
    benchmark: pd.Series,
    config: Config,
    cost: TransactionCost | None = None,
    initial: bool = False,
) -> OptimizationResult:
    """Maximize alpha'w - lambda*w'Sigma*w - gamma*TC(w-prev).

    An overall objective multiplier improves solver absolute accuracy without
    changing the optimum. Initial deployment is exempt from the turnover cap
    because full investment requires one unit of absolute cash-to-asset trades.
    """
    if not all(alpha.index.equals(obj.index) for obj in (previous, benchmark, snapshot.exposures)):
        raise ValueError("Optimizer asset alignment mismatch")
    if (not alpha.index.equals(snapshot.covariance.index)
            or not alpha.index.equals(snapshot.covariance.columns)
            or not alpha.index.equals(snapshot.specific_variance.index)
            or not snapshot.exposures.columns.equals(snapshot.factor_covariance.index)
            or not snapshot.factor_covariance.index.equals(snapshot.factor_covariance.columns)):
        raise ValueError("Optimizer covariance/factor label alignment mismatch")
    if not np.isfinite(np.column_stack([alpha, previous, benchmark])).all():
        raise ValueError("Nonfinite optimization input")
    p = config["portfolio"]
    covariance, audit = ensure_psd(snapshot.covariance.to_numpy(), 0.0)
    if audit.repaired:
        raise ValueError("Optimizer requires already validated PSD covariance")
    cost = cost or TransactionCost(**config["transaction_cost"])
    spec = ConstraintSpec.from_config(config, initial)
    industry, style = constraint_exposures(snapshot, config)
    w = cp.Variable(len(alpha))
    constraints = build_constraints(
        w, previous.to_numpy(), benchmark.to_numpy(), covariance, industry, style, spec
    )
    objective = alpha.to_numpy() @ w - p["risk_aversion"] * cp.quad_form(w, cp.psd_wrap(covariance))
    if p["cost_penalty"] > 0:
        objective -= p["cost_penalty"] * cost.expression(w - previous.to_numpy())
    problem = cp.Problem(cp.Maximize(1e4 * objective), constraints)
    if not problem.is_dcp():
        raise OptimizationError("Objective/constraints violate DCP")
    try:
        problem.solve(solver=p["solver"], **p.get("solver_options", {}))
    except cp.error.SolverError as error:
        raise OptimizationError(f"{snapshot.date}: solver failed") from error
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
        raise OptimizationError(f"{snapshot.date}: {problem.status}")
    if problem.status == cp.OPTIMAL_INACCURATE:
        logger.warning("Inaccurate solution at %s; validating feasibility", snapshot.date)
    weights = np.asarray(w.value).ravel()
    slacks, binding = check_constraints(
        weights,
        previous.to_numpy(),
        benchmark.to_numpy(),
        covariance,
        industry,
        style,
        spec,
        p["constraint_tolerance"],
    )
    return OptimizationResult(
        pd.Series(weights, index=alpha.index),
        problem.status,
        float(problem.value / 1e4),
        cost.predicted(weights - previous.to_numpy()),
        binding,
        slacks,
    )
