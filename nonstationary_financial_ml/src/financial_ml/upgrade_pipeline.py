"""Run the V2 vertical slice inside existing projects and environments."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from risk_aware_portfolio.backtest.open_diagnostics import rank_open_diagnostic
from robust_alpha.data.research_adapters import cached_sina_sample, read_bundle
from robust_alpha.data.research_contract import (
    add_calendar_labels,
    add_decision_eligibility,
    digest_file,
    normalize_dynamic_panel,
    quality_report,
    require_formal_quality,
    require_frozen,
)
from robust_alpha.data.research_fixture import engineering_sample
from robust_alpha.evaluation.conditional_reversal import (
    build_pressure_features,
    calendar_hac,
    conditional_research,
)

from financial_ml.research_v2 import (
    calendar_block_ci,
    compare_incremental,
    matched_predictions,
    train_incremental,
    validate_archive,
)
from financial_ml.utils import ROOT, path, write_json

PROJECTS = ROOT.parent


def source_hashes() -> dict:
    files = []
    for project in ("Robust_Alpha_Discovery", "nonstationary_financial_ml", "risk_aware_portfolio"):
        files.extend((PROJECTS / project / "src").rglob("*.py"))
        files.extend((PROJECTS / project / "scripts").rglob("*.py"))
        files.append(PROJECTS / project / "uv.lock")
        files.append(PROJECTS / project / "pyproject.toml")
    return {str(p.relative_to(PROJECTS)): digest_file(p) for p in sorted(files)}


def load_v2_artifacts(cfg: dict) -> pd.DataFrame:
    """No stale archive reuse after protocol, code, config, data or output changes."""
    destination = path(cfg["output"]["directory"])
    manifest = json.loads((destination / "experiment_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "complete"
        or manifest["config"] != cfg
        or (manifest["source_hashes"] != source_hashes())
        or manifest["protocol_sha256"] != digest_file(PROJECTS / "Robust_Alpha_Discovery/configs/protocols/research_v2.txt")
    ):
        raise ValueError("Stale V2 experiment: config/code/protocol mismatch")
    for filename, expected in {**manifest["input_files"], **manifest["artifacts"]}.items():
        if digest_file(Path(filename)) != expected:
            raise ValueError(f"Stale V2 artifact or input: {filename}")
    archive = pd.read_parquet(destination / "oos_predictions.parquet")
    validate_archive(archive)
    return archive


def _portfolio_outputs(archive: pd.DataFrame, panel: pd.DataFrame, cfg: dict) -> dict:
    matched, _ = matched_predictions(archive)
    strategies = {
        "A_reversal": ("ridge", "rev"),
        "B_ridge_score": ("ridge", "score"),
        "C_boosting_score": ("boosting", "score"),
        "B_ridge_calibrated": ("ridge", "expected_return"),
        "C_boosting_calibrated": ("boosting", "expected_return"),
    }
    daily, targets, summaries = [], [], []
    for strategy, (model, column) in strategies.items():
        sample = matched.loc[matched.model.eq(model)]
        for bps in cfg["portfolio"]["cost_bps_per_full_l1"]:
            ledger, weights = rank_open_diagnostic(
                sample,
                column,
                bps,
                cfg["portfolio"]["quantile"],
                outcome_panel=panel,
            )
            ledger["strategy"], weights["strategy"] = strategy, strategy
            weights["cost_bps_per_full_l1"] = bps
            daily.append(ledger)
            targets.append(weights)
            values = ledger.net_return
            complete = values.notna().all()
            summaries.append(
                {
                    "strategy": strategy,
                    "cost_bps_per_full_l1": bps,
                    "periods": len(ledger),
                    "missing_periods": int(values.isna().sum()),
                    "return_coverage": float(values.notna().mean()),
                    "annualized_mean": float(values.mean() * 252) if complete else np.nan,
                    "cagr": float(ledger.nav.iloc[-1] ** (252 / len(ledger)) - 1)
                    if complete
                    else np.nan,
                    "mean_turnover_full_l1": ledger.turnover_full_l1.mean(),
                    "status": "diagnostic_only" if complete else "blocked_incomplete_ledger",
                }
            )
    ledger = pd.concat(daily, ignore_index=True)
    paired = []
    for bps, group in ledger.groupby("cost_bps_per_full_l1"):
        matrix = group.pivot(index="date", columns="strategy", values="net_return")
        for version in ("score", "calibrated"):
            diff = matrix[f"C_boosting_{version}"] - matrix[f"B_ridge_{version}"]
            low, high = calendar_block_ci(
                diff,
                cfg["statistics"]["bootstrap_replications"],
                cfg["statistics"]["block_days"],
                cfg["seed"],
            )
            paired.append(
                {
                    "comparison": f"C_minus_B_{version}",
                    "cost_bps_per_full_l1": bps,
                    **calendar_hac(diff, cfg["statistics"]["hac_lags"]),
                    "block_ci_low": low,
                    "block_ci_high": high,
                    "scope": "descriptive_cost_scenario_not_alpha_confirmation",
                }
            )
    status = pd.DataFrame(
        [
            {
                "strategy": "D_ML_risk_control",
                "status": "blocked",
                "reason": "Close covariance/execution/attribution do not support next-open interval",
            },
            {
                "strategy": "E_ML_risk_and_cost_control",
                "status": "blocked",
                "reason": "Dynamic open event/risk/holdings accounting is not implemented",
            },
        ]
    )
    return {
        "portfolio_daily": ledger,
        "portfolio_targets": pd.concat(targets),
        "portfolio_performance": pd.DataFrame(summaries),
        "paired_net_return_intervals": pd.DataFrame(paired),
        "portfolio_strategy_status": status,
    }


def run_v2(cfg: dict, include_holdout: bool = False, quality_only: bool = False) -> Path:
    """Register the frozen protocol before data loading, preserve every failure."""
    protocol_sha = digest_file(PROJECTS / "Robust_Alpha_Discovery/configs/protocols/research_v2.txt")
    require_frozen(cfg, protocol_sha, include_holdout)
    phase, source = cfg["phase"], cfg["data"]["source"]
    formal = phase in {"development", "validation", "holdout"}
    if phase not in {
        "engineering_validation",
        "historical_integration",
        "development",
        "validation",
        "holdout",
    }:
        raise ValueError("Unknown research phase")
    if formal and (source != "bundle" or cfg["protocol"].get("empirical_status") != "frozen"):
        raise ValueError("Formal empirical protocol/data remain draft and blocked")
    if formal and not quality_only:
        raise ValueError("Formal empirical evaluation is not enabled in this engineering slice")
    if source == "synthetic" and phase != "engineering_validation":
        raise ValueError("Synthetic data cannot enter a historical or empirical report")
    if source == "cached_sina" and phase != "historical_integration":
        raise ValueError("Cached fixed basket is historical engineering integration only")
    destination = path(cfg["output"]["directory"])
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "protocol_sha256": protocol_sha,
        "source_hashes": source_hashes(),
        "python": platform.python_version(),
        "evidence_grade": phase,
        "formal_empirical": False,
        "input_files": {},
        "artifacts": {},
        "packages": {
            package: package_version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "joblib")
        },
    }
    manifest_path = destination / "experiment_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["config"] != cfg:
            raise ValueError(
                "Existing experiment has a different config; use a new output directory"
            )
    write_json(manifest, manifest_path)
    try:
        if source == "synthetic":
            raw, calendar, metadata = engineering_sample(**cfg["synthetic"])
        elif source == "cached_sina":
            raw, calendar, metadata = cached_sina_sample(PROJECTS / "risk_aware_portfolio")
        elif source == "bundle":
            if not cfg["data"].get("load_end"):
                raise ValueError("Bundle research requires a pre-read date boundary")
            bundle = path(cfg["data"]["bundle_directory"])
            raw, calendar, metadata = read_bundle(bundle, cfg["data"]["load_end"])
            manifest["input_files"] = {
                str(bundle / name): digest_file(bundle / name)
                for name in ("metadata.json", "panel.parquet", "calendar.parquet")
            }
        else:
            raise ValueError("Unsupported V2 data source")
        manifest["input_files"].update(metadata.get("source_files", {}))
        # Snapshots preserve actual values, never mutate the original cache.
        raw.to_parquet(destination / "source_snapshot.parquet", index=False)
        calendar.to_parquet(destination / "calendar_snapshot.parquet", index=False)
        write_json(metadata, destination / "data_lineage.json")
        panel = normalize_dynamic_panel(raw, calendar)
        quality = quality_report(panel, metadata)
        write_json(quality, destination / "data_quality_report.json")
        pd.DataFrame(quality["fields"]).to_csv(
            destination / "data_field_availability.csv", index=False
        )
        if formal:
            require_formal_quality(quality)
        if not quality_only:
            panel = add_decision_eligibility(
                panel, cfg["data"]["min_history"], allow_unknown_status=not formal
            )
            panel, features = build_pressure_features(
                panel, cfg["data"]["min_assets"], cfg["data"]["min_amount_20"]
            )
            panel = add_calendar_labels(panel, calendar)
            panel.to_parquet(destination / "research_panel.parquet", index=False)
            tables = conditional_research(
                panel, cfg["data"]["min_assets"], cfg["statistics"]["hac_lags"]
            )
            archive, fits, tuning, models = train_incremental(panel, features, cfg)
            archive.to_parquet(destination / "oos_predictions.parquet", index=False)
            # The forecast file retains score separately from optimizer return units.
            archive[
                [
                    "date",
                    "asset",
                    "model",
                    "expected_return",
                    "score",
                    "feature_asof",
                    "model_fit_cutoff",
                    "calibration_fit_cutoff",
                    "model_version",
                    "label_start",
                    "label_end",
                    "entry_time",
                    "exit_time",
                    "return_convention",
                    "return_unit",
                    "oos_fold",
                ]
            ].to_csv(destination / "calibrated_signals.csv", index=False)
            tables.update(model_fits=fits, validation_tuning=tuning)
            tables.update(compare_incremental(archive, cfg))
            tables.update(_portfolio_outputs(archive, panel, cfg))
            for name, table in tables.items():
                table.to_csv(destination / f"{name}.csv", index=False)
            model_dir = destination / "models"
            model_dir.mkdir(exist_ok=True)
            for name, value in models.items():
                joblib.dump(value, model_dir / f"{name}.joblib")
            text = f"# V2 工程验证报告\n\n证据等级：**{phase}**。正式 PIT 实证：**未完成**。\n\n"
            text += (
                f"本轮 {quality['assets']} 资产、{quality['dates']} 日期；OOS {len(archive)} 行，"
            )
            text += f"{archive.date.nunique()} 个决策日期，{archive.oos_fold.nunique()} 折。"
            text += "历史集成样本已查看；合成输出只验证工程，不证明经济假设。\n\n"
            text += "数据正式验收被阻止，字段清单见 data_field_availability.csv；数据阻塞见 data_quality_report.json。\n\n"
            text += "控制后系数、IC、配对增量均按冻结规则完整保存，负值和失败没有删除。\n\n"
            text += "```text\n" + tables["factor_hac_summary"].to_string(index=False) + "\n```\n\n"
            text += (
                "```text\n" + tables["oos_model_comparison"].to_string(index=False) + "\n```\n\n"
            )
            text += (
                "```text\n" + tables["portfolio_performance"].to_string(index=False) + "\n```\n\n"
            )
            text += "A/B/C 是长仓开盘全成交研究代理。未知成交或退出后 NAV 链停止；不存在默认清仓。"
            text += "费用统一为完整 L1、复用自融资记账。缺少公司行动现金/税费/容量/滑点与真实成交，不构成实盘收益。\n\n"
            text += "D/E 开盘风险优化仍被拒绝，风险暴露/归因新时段结果尚未完成。状态、重要性、窗口/重训敏感性"
            text += "未完成完整新协议矩阵，不能声称稳定增量。旧历史诊断留在原目录。\n\n"
            text += "新样本假设成立性、风险解释、成本后 alpha 与跨状态稳定性均没有正式结论。"
            text += "下一步优先补齐动态 PIT 原始数据、实现开盘事件风险账、冻结并评价新独立样本。\n"
            (destination / "engineering_report.md").write_text(text, encoding="utf-8")
        manifest["status"] = "quality_only" if quality_only else "complete"
        manifest["data_quality_formal_ready"] = quality["formal_ready"]
        manifest["artifacts"] = {
            str(p): digest_file(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file() and p != manifest_path
        }
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest, manifest_path)
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(manifest, manifest_path)
        raise
    return destination
