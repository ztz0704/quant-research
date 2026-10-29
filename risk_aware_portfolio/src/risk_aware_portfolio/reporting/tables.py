"""Publication tables and daily audit exports, using core diagnostic modules."""

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from risk_aware_portfolio.attribution.factor_attribution import (
    cumulative_attribution,
    truth_attribution,
    validate_attribution,
)
from risk_aware_portfolio.attribution.performance import (
    hac_mean_test,
    pairwise_comparisons,
    summarize_performance,
)
from risk_aware_portfolio.backtest.engine import BacktestResult
from risk_aware_portfolio.diagnostics.portfolio_diagnostics import (
    alpha_ic,
    binding_frequency,
    concentration,
    risk_contributions,
)
from risk_aware_portfolio.diagnostics.risk_diagnostics import (
    compare_covariance_estimators,
    covariance_stability,
    risk_calibration,
    risk_summary,
)
from risk_aware_portfolio.risk.model import RiskHistory
from risk_aware_portfolio.utils.config import Config
from risk_aware_portfolio.utils.manifest import file_hash
from risk_aware_portfolio.utils.paths import resolve_path


def synthetic_truth(config: Config) -> pd.DataFrame | None:
    """Only attach truth when its content hash is tied to the current input panel."""
    path = resolve_path("data/raw/synthetic_manifest.json")
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    truth_path = resolve_path("data/raw/synthetic_truth.parquet")
    if manifest["panel_sha256"] != file_hash(resolve_path(config["data"]["input_path"])):
        return None
    if manifest["truth_sha256"] != file_hash(truth_path):
        raise ValueError("Synthetic truth was modified")
    for key, filename in (
        ("factor_returns_sha256", "true_factor_returns.parquet"),
        ("factor_covariance_sha256", "true_factor_covariance.parquet"),
    ):
        if manifest.get(key) != file_hash(resolve_path(f"data/raw/{filename}")):
            raise ValueError("Synthetic factor truth is stale or modified; rerun prepare_data")
    return pd.read_parquet(truth_path)


def _write_table(
    frame: pd.DataFrame, name: str, index: bool = True, directory: Path | None = None
) -> None:
    directory = directory or resolve_path("outputs/tables")
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{name}.csv", index=index)


def factor_return_recovery(history: RiskHistory, config: Config) -> pd.DataFrame | None:
    """Independent DGP truth comparison, including estimation error and alpha absorption."""
    if synthetic_truth(config) is None:
        return None
    truth = pd.read_parquet(resolve_path("data/raw/true_factor_returns.parquet"))
    if not truth.columns.equals(history.factor_returns.columns):
        return None
    truth = truth.loc[history.factor_returns.index]
    rows = []
    for factor in truth:
        estimate = history.factor_returns[factor]
        rows.append(
            {
                "factor": factor,
                "correlation": estimate.corr(truth[factor]),
                "rmse": np.sqrt(np.mean((estimate - truth[factor]) ** 2)),
                "mean_error": (estimate - truth[factor]).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_tables(
    results: dict[str, BacktestResult],
    panel: pd.DataFrame,
    history: RiskHistory,
    config: Config,
    output: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Write four-portfolio metrics and detailed ledgers without choosing subperiods."""
    _export = partial(_write_table, directory=output)
    performance, risk, attribution = [], [], []
    truth = synthetic_truth(config)
    annualization = config["backtest"]["annualization"]
    for slug, result in results.items():
        validate_attribution(result)
        performance.append(
            {
                "portfolio": result.name,
                **summarize_performance(
                    result.returns,
                    result.weights,
                    annualization,
                    **{
                        k: config["reporting"][k]
                        for k in (
                            "hac_lags",
                            "bootstrap_block",
                            "bootstrap_repetitions",
                            "bootstrap_seed",
                        )
                    },
                ),
            }
        )
        risk.append({"portfolio": result.name, **risk_summary(result, annualization)})
        pnl = cumulative_attribution(result)
        attribution.append({"portfolio": result.name, **pnl.iloc[-1].to_dict()})
        _export(result.attribution, f"{slug}_daily_attribution")
        _export(pnl, f"{slug}_cumulative_attribution")
        _export(concentration(result, config), f"{slug}_concentration")
        _export(
            binding_frequency(result, config["portfolio"]["constraint_tolerance"]),
            f"{slug}_binding_frequency",
            False,
        )
        counts = (
            result.solver_diagnostics["status"]
            .value_counts()
            .rename_axis("status")
            .to_frame("count")
        )
        _export(counts, f"{slug}_solver_status")
        if truth is not None:
            daily_truth = truth_attribution(result, truth)
            _export(daily_truth, f"{slug}_truth_attribution")
        else:
            _export(
                pd.DataFrame(
                    columns=[
                        "date",
                        "true_factor",
                        "true_alpha",
                        "true_specific",
                        "transaction_cost",
                    ]
                ),
                f"{slug}_truth_attribution",
                False,
            )
    frames = {
        "performance_summary": pd.DataFrame(performance),
        "risk_summary": pd.DataFrame(risk),
        "attribution_summary": pd.DataFrame(attribution),
    }
    frames["pairwise_comparisons"] = pairwise_comparisons(
        {r.name: r.returns for r in results.values()},
        results["baseline"].name,
        config["reporting"]["hac_lags"],
    )
    main = results["risk_aware_tc"]
    ic = alpha_ic(main, panel)
    _export(ic, "alpha_ic")
    frames["alpha_ic_summary"] = pd.DataFrame(
        [
            {
                "alpha_mode": name,
                **hac_mean_test(ic[name].to_numpy(), config["reporting"]["hac_lags"]),
            }
            for name in ic
        ]
    )
    forward, buckets = risk_calibration(
        main, panel, config["backtest"]["realized_risk_window"], annualization
    )
    _export(forward, "risk_calibration_daily")
    _export(buckets, "risk_calibration_buckets")
    _export(covariance_stability(history), "covariance_stability")
    factor, asset = risk_contributions(main, history)
    _export(factor, "factor_risk_contributions", False)
    _export(asset, "asset_risk_contributions", False)
    _export(main.factor_exposures, "style_and_encoded_industry_exposures")
    industries = main.factor_exposures.filter(like="industry:").copy()
    reference = f"industry:{config['risk_model']['industries'][0]}"
    industries[reference] = main.weights.sum(axis=1) - industries.sum(axis=1)
    _export(industries, "industry_exposures")
    recovery = factor_return_recovery(history, config)
    true_covariance = (
        pd.read_parquet(resolve_path("data/raw/true_factor_covariance.parquet"))
        if truth is not None
        else None
    )
    frames["covariance_method_comparison"] = compare_covariance_estimators(
        panel, config, true_covariance
    )
    if recovery is not None:
        frames["factor_return_recovery"] = recovery
    else:
        _export(
            pd.DataFrame(columns=["factor", "correlation", "rmse", "mean_error"]),
            "factor_return_recovery",
            False,
        )
    for name, frame in frames.items():
        _export(frame, name, False)
    return frames
