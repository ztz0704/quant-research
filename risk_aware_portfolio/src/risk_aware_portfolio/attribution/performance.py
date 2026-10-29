"""Performance estimates with serial-dependence-aware uncertainty."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def hac_mean_test(values: np.ndarray, lags: int = 5) -> dict[str, float]:
    """Newey-West/Bartlett standard error for the daily arithmetic mean."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {
            "mean": np.nan,
            "hac_standard_error": np.nan,
            "hac_t_stat": np.nan,
            "p_value": np.nan,
        }
    centered = x - x.mean()
    long_run = float(centered @ centered / n)
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = float(centered[lag:] @ centered[:-lag] / n)
        long_run += 2 * (1 - lag / (lags + 1)) * gamma
    se = np.sqrt(max(long_run, 0) / n)
    t_stat = x.mean() / se if se > 0 else np.nan
    return {
        "mean": float(x.mean()),
        "hac_standard_error": float(se),
        "hac_t_stat": float(t_stat),
        "p_value": float(2 * norm.sf(abs(t_stat))),
    }


def block_bootstrap_sharpe(
    values: np.ndarray,
    annualization: int = 252,
    block: int = 10,
    repetitions: int = 500,
    seed: int = 987,
) -> tuple[float, float]:
    """Circular moving-block bootstrap percentile interval; no IID return resampling."""
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(x) / block))
    starts = rng.integers(0, len(x), size=(repetitions, blocks))
    indices = (starts[:, :, None] + np.arange(block)) % len(x)
    sample = x[indices.reshape(repetitions, -1)[:, : len(x)]]
    std = sample.std(axis=1, ddof=1)
    estimates = np.divide(
        sample.mean(axis=1) * np.sqrt(annualization),
        std,
        out=np.full(repetitions, np.nan),
        where=std > 0,
    )
    lower, upper = np.nanquantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def drawdown(returns: pd.Series) -> pd.Series:
    """Include initial NAV=1 as a possible high-water mark."""
    nav = (1 + returns).cumprod()
    peak = nav.cummax().clip(lower=1.0)
    return nav / peak - 1


def summarize_performance(
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    annualization: int = 252,
    uncertainty: bool = True,
    hac_lags: int = 5,
    bootstrap_block: int = 10,
    bootstrap_repetitions: int = 500,
    bootstrap_seed: int = 987,
) -> dict[str, float]:
    """Net performance versus the declared fee-free benchmark on identical periods."""
    r, benchmark = daily["net_return"], daily["benchmark_return"]
    active = r - benchmark
    std, active_std = r.std(ddof=1), active.std(ddof=1)
    annual_return = float(np.expm1(np.log1p(r).sum() * annualization / len(r)))
    volatility = float(std * np.sqrt(annualization))
    max_dd = float(drawdown(r).min())
    denominator = weights.pow(2).sum(axis=1)
    result = {
        "annualized_return": annual_return,
        "annualized_volatility": volatility,
        "sharpe": float(r.mean() / std * np.sqrt(annualization)) if std else np.nan,
        "max_drawdown": max_dd,
        "calmar": annual_return / abs(max_dd) if max_dd else np.nan,
        "average_turnover": float(daily["turnover"].mean()),
        "average_rebalance_turnover": float(daily.loc[daily["rebalance"], "turnover"].mean()),
        "total_transaction_cost": float(daily["transaction_cost"].sum()),
        "annualized_active_return": float(active.mean() * annualization),
        "tracking_error": float(active_std * np.sqrt(annualization)),
        "information_ratio": float(active.mean() / active_std * np.sqrt(annualization))
        if active_std
        else np.nan,
        "average_gross_exposure": float(weights.abs().sum(axis=1).mean()),
        "average_net_exposure": float(weights.sum(axis=1).mean()),
        "maximum_position": float(weights.abs().to_numpy().max()),
        "effective_number_of_bets": float((1 / denominator).mean()),
        "normalized_effective_positions": float(
            (weights.abs().sum(axis=1) ** 2 / denominator).mean()
        ),
        "gross_annualized_return": float(
            np.expm1(np.log1p(daily["gross_return"]).sum() * annualization / len(r))
        ),
        "n_periods": float(len(r)),
    }
    if uncertainty:
        lower, upper = block_bootstrap_sharpe(
            r.to_numpy(), annualization, bootstrap_block, bootstrap_repetitions, bootstrap_seed
        )
        result.update(
            {
                "sharpe_ci_lower": lower,
                "sharpe_ci_upper": upper,
                **hac_mean_test(active.to_numpy(), hac_lags),
            }
        )
    return result


def pairwise_comparisons(
    returns: dict[str, pd.DataFrame], baseline_name: str, hac_lags: int = 5
) -> pd.DataFrame:
    """Paired HAC tests of mean net return difference, no claim of optimizer significance."""
    baseline = returns[baseline_name]["net_return"]
    records = []
    for name, frame in returns.items():
        if name == baseline_name:
            continue
        if not frame.index.equals(baseline.index):
            raise ValueError("Comparison periods must be identical")
        records.append(
            {
                "portfolio": name,
                "comparison": baseline_name,
                **hac_mean_test((frame["net_return"] - baseline).to_numpy(), hac_lags),
            }
        )
    result = pd.DataFrame(records)
    result["holm_p_value"] = np.nan
    finite = result["p_value"].dropna().sort_values()
    adjusted = np.maximum.accumulate(finite.to_numpy() * np.arange(len(finite), 0, -1))
    result.loc[finite.index, "holm_p_value"] = np.minimum(adjusted, 1.0)
    return result
