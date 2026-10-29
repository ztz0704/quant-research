"""As-of forecast diagnostics and explicitly ex-post forward calibration."""

import numpy as np
import pandas as pd

from risk_aware_portfolio.backtest.engine import BacktestResult
from risk_aware_portfolio.risk.factor_covariance import observation_weights
from risk_aware_portfolio.risk.model import RiskHistory, estimate_risk_history
from risk_aware_portfolio.utils.config import Config, variant


def covariance_stability(history: RiskHistory) -> pd.DataFrame:
    """Measure covariance forecast changes using consecutive snapshots."""
    rows, previous = [], None
    for date, snapshot in history.snapshots.items():
        matrix = snapshot.covariance.to_numpy()
        change = (
            np.nan
            if previous is None
            else np.linalg.norm(matrix - previous, "fro") / np.linalg.norm(previous, "fro")
        )
        rows.append(
            {
                "date": date,
                **snapshot.audit.metrics(),
                "factor_condition_number": snapshot.factor_audit.metrics()["condition_number"],
                "relative_frobenius_change": change,
            }
        )
        previous = matrix
    return pd.DataFrame(rows).set_index("date")


def compare_covariance_estimators(
    panel: pd.DataFrame, config: Config, true_factor_covariance: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compare all declared estimators on identical as-of dates, including DGP error."""
    rows = []
    for method in config["experiments"]["covariance_methods"]:
        settings = variant(
            config, risk_model={"covariance_method": method, "specific_method": method}
        )
        history = estimate_risk_history(panel, settings)
        stability = covariance_stability(history)
        error, effective, factor_minima = [], [], []
        for date, snapshot in history.snapshots.items():
            n = min(settings["risk_model"]["lookback"], len(history.factor_returns.loc[:date]))
            weights = observation_weights(n, settings["risk_model"]["ewma_lambda"])
            effective.append(float(n) if method == "sample" else 1 / (weights @ weights))
            factor_minima.append(snapshot.factor_audit.after.min())
            if true_factor_covariance is not None:
                if not snapshot.factor_covariance.index.equals(true_factor_covariance.index):
                    raise ValueError("Synthetic factor covariance vocabulary mismatch")
                truth = true_factor_covariance.to_numpy()
                error.append(
                    np.linalg.norm(snapshot.factor_covariance.to_numpy() - truth, "fro")
                    / np.linalg.norm(truth, "fro")
                )
        rows.append(
            {
                "method": method,
                "n_forecasts": len(stability),
                "mean_effective_observations": np.mean(effective),
                "mean_asset_condition_number": stability["condition_number"].mean(),
                "mean_factor_condition_number": stability["factor_condition_number"].mean(),
                "minimum_asset_eigenvalue": stability["smallest_eigenvalue"].min(),
                "maximum_asset_eigenvalue": stability["largest_eigenvalue"].max(),
                "minimum_factor_eigenvalue": min(factor_minima),
                "mean_relative_covariance_change": stability["relative_frobenius_change"].mean(),
                "mean_relative_factor_covariance_error": np.mean(error) if error else np.nan,
            }
        )
    return pd.DataFrame(rows)


def risk_calibration(
    result: BacktestResult, panel: pd.DataFrame, horizon: int = 21, annualization: int = 252
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare risk_t with future volatility of the SAME frozen decision-date weights.

    Forward windows and bin boundaries are report diagnostics only. Windows overlap;
    bin results are descriptive, not independent significance tests.
    """
    asset_returns = panel.pivot(index="date", columns="asset", values="return")
    dates = list(asset_returns.index)
    positions = {date: i for i, date in enumerate(dates)}
    rows = []
    for date, w in result.weights.iterrows():
        start = positions[date] + 1
        window = asset_returns.iloc[start : start + horizon].reindex(columns=w.index)
        if len(window) != horizon:
            continue
        realized = window.to_numpy() @ w.to_numpy()
        forecast = result.returns.loc[
            result.returns["decision_date"].eq(date), "predicted_volatility"
        ].iloc[0]
        rows.append(
            {
                "date": date,
                "predicted_volatility": forecast,
                "forward_realized_volatility": realized.std(ddof=1) * np.sqrt(annualization),
                "forward_mean_square_return": np.mean(realized**2),
            }
        )
    frame = pd.DataFrame(rows).set_index("date")
    if frame.empty:
        raise ValueError("Insufficient observations for forward risk calibration")
    frame["bucket"] = pd.qcut(frame["predicted_volatility"], q=5, labels=False, duplicates="drop")
    buckets = frame.groupby("bucket").agg(
        predicted_volatility=("predicted_volatility", "mean"),
        realized_volatility=("forward_realized_volatility", "mean"),
        n_observations=("predicted_volatility", "size"),
    )
    buckets["realized_to_predicted"] = (
        buckets["realized_volatility"] / buckets["predicted_volatility"]
    )
    return frame, buckets


def risk_summary(result: BacktestResult, annualization: int = 252) -> dict[str, float]:
    """One-step calibration via realized gross returns versus daily forecast variance."""
    risk = result.risk_contributions
    daily_variance = risk["total_variance"].to_numpy()
    returns = result.returns["gross_return"].to_numpy()
    return {
        "average_predicted_volatility": float(np.sqrt(daily_variance * annualization).mean()),
        "realized_gross_volatility": float(returns.std(ddof=1) * np.sqrt(annualization)),
        "mean_squared_return_to_variance": float(np.mean(returns**2 / daily_variance)),
        "qlike": float(np.mean(np.log(daily_variance) + returns**2 / daily_variance)),
        "average_factor_risk_fraction": float(risk["factor_risk_fraction"].mean()),
        "average_specific_risk_fraction": float(risk["specific_risk_fraction"].mean()),
        "maximum_repair_variance_adjustment": float(risk["repair_variance_adjustment"].abs().max()),
    }
