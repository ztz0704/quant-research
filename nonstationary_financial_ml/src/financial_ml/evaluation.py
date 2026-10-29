"""One common OOS comparison; saved predictions are never used for tuning."""

import json

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from financial_ml.metrics import daily_ic, prediction_metrics
from financial_ml.portfolio import performance, portfolio_daily
from financial_ml.utils import output, path, sha256


def load_research(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Reject stale predictions if input, feature data or declared experiment changed."""
    destination = output(cfg)
    feature_manifest = json.loads((destination / "tables/feature_manifest.json").read_text(encoding="utf-8"))
    training = json.loads((destination / "tables/training_manifest.json").read_text(encoding="utf-8"))
    predictions_path = destination / "predictions/oos_predictions.parquet"
    if (training["config"] != cfg or training["features_sha256"] != sha256(path(cfg["data"]["features_path"]))
            or feature_manifest["input_sha256"] != sha256(path(cfg["data"]["input_path"]))
            or training["predictions_sha256"] != sha256(predictions_path)):
        raise ValueError("Stale research artifacts: rerun feature construction and model training")
    return pd.read_parquet(path(cfg["data"]["features_path"])), pd.read_parquet(predictions_path), training["feature_names"]


def evaluate(cfg: dict) -> dict[str, pd.DataFrame]:
    """Generate statistical and economic diagnostics for every declared model."""
    _, predictions, _ = load_research(cfg)
    destination = output(cfg)
    metrics, ic_rows, portfolios, quantiles, distributions = [], [], [], [], []
    for model, group in predictions.groupby("model", sort=True):
        for column in ("raw_prediction", "neutralized_prediction"):
            metrics.append({"model": model, "prediction": column, **prediction_metrics(group, column, max(5, cfg["labels"]["forward_horizon"] - 1))})
            daily = daily_ic(group, column)
            daily["model"], daily["prediction"] = model, column
            ic_rows.append(daily)
            for kind in ("long_only", "long_short"):
                portfolios.append(portfolio_daily(group, column, kind, cfg))
            for date, section in group.groupby("date", sort=True):
                distributions.append({"date": date, "model": model, "prediction": column,
                                      "mean": section[column].mean(), "std": section[column].std(),
                                      "q01": section[column].quantile(.01), "q99": section[column].quantile(.99)})
                if section[column].std(ddof=0) < 1e-12:
                    continue
                ranked = section.copy()
                ranked["quantile"] = pd.qcut(ranked[column].rank(method="first"), cfg["portfolio"]["quantiles"], labels=False) + 1
                for number, quantile in ranked.groupby("quantile", observed=True):
                    quantiles.append({"date": date, "model": model, "prediction": column, "quantile": number,
                                      "mean_forward_return": quantile.target.mean(), "assets": len(quantile),
                                      "valid_assets": int(quantile.target.notna().sum())})
    statistics = pd.DataFrame(metrics)
    finite = statistics.hac_p.notna()
    statistics["holm_p"] = np.nan
    if finite.any():
        statistics.loc[finite, "holm_p"] = multipletests(statistics.loc[finite, "hac_p"], method="holm")[1]
    portfolio = pd.concat(portfolios, ignore_index=True)
    summaries = []
    for identity, group in portfolio.groupby(["model", "prediction", "portfolio", "cost_bps_per_half_l1"], sort=True):
        summaries.append(dict(zip(["model", "prediction", "portfolio", "cost_bps_per_half_l1"], identity, strict=True)) | performance(group, cfg["labels"]["forward_horizon"]))
    frames = {"model_performance": statistics, "ic_summary": statistics.copy(), "daily_ic": pd.concat(ic_rows, ignore_index=True),
              "portfolio_summary": pd.DataFrame(summaries), "portfolio_daily": portfolio,
              "decile_returns": pd.DataFrame(quantiles), "prediction_distribution": pd.DataFrame(distributions)}
    for name, frame in frames.items():
        frame.to_csv(destination / f"tables/{name}.csv", index=False)
    return frames
