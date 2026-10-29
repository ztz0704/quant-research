"""Convex constraints and independent post-solve numerical feasibility checks."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from risk_aware_portfolio.risk.model import RiskSnapshot
from risk_aware_portfolio.utils.config import Config


@dataclass(frozen=True)
class ConstraintSpec:
    long_only: bool
    max_weight: float
    gross_limit: float
    turnover_limit: float | None
    industry_tolerance: float | None
    style_tolerance: float | None
    tracking_error_daily: float | None
    min_weight: float = 0.0

    @classmethod
    def from_config(cls, config: Config, initial: bool = False) -> "ConstraintSpec":
        """Convert annual tracking error to daily standard deviation."""
        p = config["portfolio"]
        te = p["tracking_error_annual"]
        return cls(
            p["long_only"],
            p["max_weight"],
            p["gross_limit"],
            None if initial else p["turnover_limit"],
            p["industry_tolerance"],
            p["style_tolerance"],
            None if te is None else te / np.sqrt(config["backtest"]["annualization"]),
            p.get("min_weight", 0.0),
        )


def constraint_exposures(snapshot: RiskSnapshot, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the dropped reference dummy for complete industry constraints."""
    x = snapshot.exposures
    others = x[[f"industry:{s}" for s in config["risk_model"]["industries"][1:]]].to_numpy()
    industry = np.column_stack([1 - others.sum(axis=1), others])
    return industry, x[config["risk_model"]["styles"]].to_numpy()


def build_constraints(
    w: cp.Variable,
    previous: np.ndarray,
    benchmark: np.ndarray,
    covariance: np.ndarray,
    industry: np.ndarray,
    style: np.ndarray,
    spec: ConstraintSpec,
) -> list[cp.Constraint]:
    """Quadratic tracking error bound is convex and DCP compliant."""
    lower = spec.min_weight if spec.long_only else -spec.max_weight
    constraints = [
        cp.sum(w) == (1.0 if spec.long_only else 0.0),
        w >= lower,
        w <= spec.max_weight,
        cp.norm1(w) <= spec.gross_limit,
    ]
    if spec.turnover_limit is not None:
        constraints.append(cp.norm1(w - previous) <= spec.turnover_limit)
    for x, tolerance in ((industry, spec.industry_tolerance), (style, spec.style_tolerance)):
        if tolerance is not None:
            constraints.append(cp.abs(x.T @ (w - benchmark)) <= tolerance)
    if spec.tracking_error_daily is not None:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        root = np.sqrt(np.maximum(eigenvalues, 0))[:, None] * eigenvectors.T
        if spec.tracking_error_daily == 0:
            constraints.append(root @ (w - benchmark) == 0)
        else:
            constraints.append(cp.norm(root @ (w - benchmark) / spec.tracking_error_daily, 2) <= 1)
    return constraints


def check_constraints(
    weights: np.ndarray,
    previous: np.ndarray,
    benchmark: np.ndarray,
    covariance: np.ndarray,
    industry: np.ndarray,
    style: np.ndarray,
    spec: ConstraintSpec,
    tolerance: float,
) -> tuple[dict[str, float], int]:
    """Compute slacks using returned weights, never clipped/renormalized weights."""
    lower = spec.min_weight if spec.long_only else -spec.max_weight
    slack = {
        "budget": tolerance - abs(weights.sum() - (1.0 if spec.long_only else 0.0)),
        "position_lower": float((weights - lower).min()),
        "position_upper": float((spec.max_weight - weights).min()),
        "gross": float(spec.gross_limit - np.abs(weights).sum()),
    }
    binding = int(
        np.sum(weights - lower <= tolerance) + np.sum(spec.max_weight - weights <= tolerance)
    )
    if spec.turnover_limit is not None:
        slack["turnover"] = float(spec.turnover_limit - np.abs(weights - previous).sum())
    for name, x, limit in (
        ("industry", industry, spec.industry_tolerance),
        ("style", style, spec.style_tolerance),
    ):
        if limit is not None:
            individual = limit - np.abs(x.T @ (weights - benchmark))
            slack[name] = float(individual.min())
            binding += int(np.sum(individual <= tolerance))
    if spec.tracking_error_daily is not None:
        active = weights - benchmark
        # Check TE in standard deviation units, not a loose variance tolerance.
        slack["tracking_error"] = float(
            spec.tracking_error_daily - np.sqrt(max(float(active @ covariance @ active), 0))
        )
    if any(value < -tolerance for value in slack.values()):
        raise ValueError(f"Post-solve constraint violation: {slack}")
    binding += sum(
        slack[name] <= tolerance
        for name in ("gross", "turnover", "tracking_error")
        if name in slack
    )
    return slack, int(binding)
