"""Run a derived, historical OOS ML-to-risk experiment using the risk project's Python.

Run from risk_aware_portfolio with its existing environment. This verifies the
original empirical inputs before deriving an isolated panel and does not modify
the original risk configuration, data, manifests or reports.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from risk_aware_portfolio.data.loader import load_panel
from risk_aware_portfolio.reporting.real_validation import (
    real_no_lookahead_checks,
    run_primary_comparisons,
    verify_real_data,
)
from risk_aware_portfolio.reporting.tables import build_tables
from risk_aware_portfolio.risk.model import estimate_risk_history
from risk_aware_portfolio.utils.config import load_config
from risk_aware_portfolio.utils.manifest import file_hash

ML_ROOT = Path(__file__).resolve().parents[1]
RISK_ROOT = ML_ROOT.parent / "risk_aware_portfolio"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--prediction", choices=["raw_prediction", "neutralized_prediction"], default="raw_prediction")
    parser.add_argument("--rebalance-frequency", choices=["daily", "weekly"], default="daily")
    args = parser.parse_args()
    original_config = load_config(str(RISK_ROOT / "configs/ashare.yaml"))
    original = verify_real_data(original_config)
    training = json.loads((ML_ROOT / "outputs/tables/training_manifest.json").read_text(encoding="utf-8"))
    feature_manifest = json.loads((ML_ROOT / "outputs/tables/feature_manifest.json").read_text(encoding="utf-8"))
    ml_config = yaml.safe_load((ML_ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    if training["config"] != ml_config or ml_config["labels"]["forward_horizon"] != 1:
        raise ValueError("Bridge requires the current one-session close-to-close experiment")
    ml_input = ML_ROOT / ml_config["data"]["input_path"]
    prediction_path = ML_ROOT / "outputs/predictions/oos_predictions.parquet"
    if (feature_manifest["input_sha256"] != file_hash(ml_input)
            or training["predictions_sha256"] != file_hash(prediction_path)):
        raise ValueError("ML input/prediction fingerprints changed")
    full_panel = pd.read_parquet(ml_input)
    # Match values as well as provenance; a differently prepared real panel is not interchangeable.
    risk_input = RISK_ROOT / original_config["data"]["input_path"]
    pd.testing.assert_frame_equal(full_panel, pd.read_parquet(risk_input), check_exact=True)
    predictions = pd.read_parquet(prediction_path)
    chosen = predictions.loc[predictions.model.eq(args.model)]
    if chosen.empty:
        raise ValueError("Model not present in OOS predictions")
    first, last = chosen.date.min(), chosen.date.max()
    calendar = pd.DatetimeIndex(sorted(full_panel.date.unique()))
    next_dates = calendar[calendar > last]
    if next_dates.empty:
        raise ValueError("Final forecast needs a next-session valuation observation")
    last_valuation = next_dates[0]
    index = calendar.get_indexer([first])[0]
    warmup = original_config["risk_model"]["lookback"] + 2
    start = calendar[max(0, index - warmup)]
    panel = full_panel.loc[full_panel.date.between(start, last_valuation)].copy()
    scores = chosen[["date", "asset", args.prediction]].rename(columns={args.prediction: "alpha"})
    joined = panel.drop(columns="alpha").merge(scores, on=["date", "asset"], how="left", validate="one_to_one")
    joined["signal_role"] = "oos_prediction"
    missing = joined.alpha.isna()
    warm = missing & joined.date.lt(first)
    valuation_only = missing & joined.date.gt(last)
    unavailable = missing & ~joined.tradable & joined.date.between(first, last)
    if (missing & ~(warm | valuation_only | unavailable)).any():
        raise ValueError("Missing OOS prediction for a tradable stock; refusing fill or stale carry")
    # These explicit sentinels are never presented as ML forecasts. Warmup has no
    # trading; final valuation is after the last decision; known halts defer trades.
    joined.loc[warm, "signal_role"] = "risk_warmup_no_trading"
    joined.loc[valuation_only, "signal_role"] = "post_window_valuation_only"
    joined.loc[unavailable, "signal_role"] = "known_untradable_abstention"
    joined.loc[missing, "alpha"] = 0.0
    destination = ML_ROOT / "outputs/portfolio_bridge" / f"{args.model}_{args.prediction}_{args.rebalance_frequency}"
    for folder in ("tables", "backtests"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    joined.to_parquet(destination / "panel.parquet", index=False)
    joined[["date", "asset", "alpha"]].to_csv(destination / "alpha.csv", index=False)
    joined[["date", "asset", "signal_role"]].to_csv(destination / "signal_roles.csv", index=False)
    cfg = copy.deepcopy(original_config)
    cfg["data"]["input_path"] = str(destination / "panel.parquet")
    cfg["data"]["alpha_path"] = str(destination / "alpha.csv")
    cfg["backtest"]["evaluation_start"] = str(first.date())
    cfg["backtest"]["rebalance_frequency"] = args.rebalance_frequency
    cfg["real_data"]["output_dir"] = str(destination)
    # All orders retain the risk project's score standardization and fixed scale.
    (destination / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    research_panel = load_panel(destination / "panel.parquet", destination / "alpha.csv")
    history = estimate_risk_history(research_panel, cfg)
    results = run_primary_comparisons(research_panel, history, cfg, destination / "backtests")
    frames = build_tables(results, research_panel, history, cfg, destination / "tables")
    checks = real_no_lookahead_checks(research_panel, cfg, results["risk_aware_tc"])
    checks.to_csv(destination / "tables/no_lookahead_checks.csv", index=False)
    manifest = {"model": args.model, "prediction": args.prediction, "rebalance_frequency": args.rebalance_frequency,
                "first_decision": str(first), "last_decision": str(last),
                "last_valuation": str(last_valuation), "score_scale_assumption": cfg["alpha"]["scale"],
                "signal_role_counts": joined.signal_role.value_counts().to_dict(),
                "original_panel_sha256": original["panel_sha256"], "original_manifest_sha256": file_hash(RISK_ROOT / "data/raw/ashare/data_manifest.json"),
                "predictions_sha256": file_hash(prediction_path), "derived_panel_sha256": file_hash(destination / "panel.parquet"),
                "config_sha256": hashlib.sha256((destination / "config.yaml").read_bytes()).hexdigest(),
                "kind": "exploratory historical integration; not a untouched holdout or brokerage fill simulation"}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    text = "# ML 到风险组合的真实数据连接实验\n\n"
    text += f"固定模型：{args.model}；决策 {first.date()} 至 {last.date()}；估值至 {last_valuation.date()}。全部原始风险输入先通过原项目 hash 审核，再派生本目录数据。\n\n"
    text += f"调仓频率：{args.rebalance_frequency}。默认 daily 对齐一日预测目标；weekly 为另外的持有规则研究，不能等同于模型预测了一周收益。此前继承原风险配置的 weekly 实验保留在 ridge_raw_prediction 目录，不用于选择新默认。\n\n"
    text += "外部 alpha 在风险项目中按分数重新标准化，scale=0.0006 是固定假设，不能把优化器输入解释成经过校准的 ML 预测收益。风险预热和末日估值占位值明确记录，预热不交易；可交易股票缺失预测直接失败。\n\n"
    text += "这是相同历史样本上的探索性连接，保留固定幸存股票池、粗行业、来源发布时点未认证、理想化收盘成交、缺少手数/涨跌停/真实成交约束等限制。收益改善不能单独证明模型 alpha。\n\n"
    text += "\n```text\n" + frames["performance_summary"].to_string(index=False) + "\n```\n"
    (destination / "research_report.md").write_text(text + "\n", encoding="utf-8")
    print(destination / "research_report.md")


if __name__ == "__main__":
    main()
