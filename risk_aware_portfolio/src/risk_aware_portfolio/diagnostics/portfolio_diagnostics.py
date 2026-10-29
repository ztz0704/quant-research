"""Concentration, factor/asset Euler risk, solver and alpha diagnostics."""

import pandas as pd
from scipy.stats import spearmanr

from risk_aware_portfolio.backtest.engine import BacktestResult
from risk_aware_portfolio.risk.model import RiskHistory
from risk_aware_portfolio.utils.config import Config


def concentration(result: BacktestResult, config: Config) -> pd.DataFrame:
    """Report decision weights, including natural drift between constrained trades."""
    w = result.weights
    squared = w.pow(2).sum(axis=1)
    return pd.DataFrame(
        {
            "gross_exposure": w.abs().sum(axis=1),
            "net_exposure": w.sum(axis=1),
            "maximum_position": w.abs().max(axis=1),
            "effective_number_of_bets": 1 / squared,
            "normalized_effective_positions": w.abs().sum(axis=1) ** 2 / squared,
            "positions_at_cap": (
                w.abs()
                >= config["portfolio"]["max_weight"] - config["portfolio"]["constraint_tolerance"]
            ).sum(axis=1),
            "position_limit_exceeded_by_drift": w.abs().max(axis=1)
            > config["portfolio"]["max_weight"] + config["portfolio"]["constraint_tolerance"],
        }
    )


def risk_contributions(
    result: BacktestResult, history: RiskHistory
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signed Euler factor and asset variance contributions; specific risk is nonnegative."""
    factor_rows, asset_rows = [], []
    for date, weights in result.weights.iterrows():
        snapshot = history.snapshots[date]
        w = weights.to_numpy()
        exposure = snapshot.exposures.to_numpy().T @ w
        factor = exposure * (snapshot.factor_covariance.to_numpy() @ exposure)
        marginal = snapshot.covariance.to_numpy() @ w
        for name, value in zip(snapshot.exposures.columns, factor):
            factor_rows.append({"date": date, "factor": name, "variance_contribution": value})
        for name, weight, covariance_weight, d in zip(
            weights.index, w, marginal, snapshot.specific_variance
        ):
            asset_rows.append(
                {
                    "date": date,
                    "asset": name,
                    "total_variance_contribution": weight * covariance_weight,
                    "specific_variance_contribution": weight**2 * d,
                }
            )
    return pd.DataFrame(factor_rows), pd.DataFrame(asset_rows)


def binding_frequency(result: BacktestResult, tolerance: float = 2e-6) -> pd.DataFrame:
    """Binding frequencies on solved rebalance dates only, excluding mandatory budget."""
    solved = result.solver_diagnostics
    solved = solved[solved["status"].isin(["optimal", "optimal_inaccurate"])]
    columns = [
        c
        for c in solved.select_dtypes(include="number")
        if c not in {"binding_count", "objective", "budget"}
    ]
    return pd.DataFrame(
        {
            "constraint": columns,
            "binding_frequency": [(solved[c] <= tolerance).mean() for c in columns],
            "n_solutions": len(solved),
        }
    )


def alpha_ic(result: BacktestResult, panel: pd.DataFrame) -> pd.DataFrame:
    """Same-date alpha versus next-period returns; no same-period IC."""
    rows = []
    for date, alpha in result.alphas.groupby("date", sort=True):
        next_date = result.returns.loc[
            result.returns["decision_date"].eq(date), "return_date"
        ].iloc[0]
        realized = panel[panel["date"].eq(next_date)].set_index("asset")["return"]
        aligned = alpha.set_index("asset")
        rows.append(
            {
                "date": date,
                "raw_ic": spearmanr(
                    aligned["raw_alpha"], realized.reindex(aligned.index)
                ).statistic,
                "neutralized_ic": spearmanr(
                    aligned["neutralized_alpha"], realized.reindex(aligned.index)
                ).statistic,
            }
        )
    return pd.DataFrame(rows).set_index("date")
