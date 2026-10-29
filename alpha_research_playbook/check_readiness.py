"""Inventory existing empirical artifacts without downloading or running experiments.

This checks local inputs, dimensions and output files. It does not certify
point-in-time data, profitable alpha, or production readiness.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def summarize_panel(path: Path, time_column: str) -> dict[str, Any]:
    """Read only identity columns from a known real-data parquet cache."""
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    frame = pd.read_parquet(path, columns=[time_column, "asset"])
    times = pd.to_datetime(frame[time_column], errors="raise")
    result.update(
        rows=len(frame),
        assets=int(frame["asset"].nunique()),
        first_timestamp=times.min().isoformat(),
        last_timestamp=times.max().isoformat(),
        calendar_dates=int(times.dt.date.nunique()),
        duplicate_keys=int(frame.duplicated([time_column, "asset"]).sum()),
        asset_ids=sorted(frame["asset"].astype(str).unique().tolist()),
    )
    return result


def read_table(path: Path) -> list[dict[str, str]]:
    """Preserve CSV field names and values rather than inventing metric definitions."""
    if not path.is_file():
        return []
    return pd.read_csv(path, dtype=str).fillna("").to_dict("records")


def inspect_project(
    root: Path, name: str, config_name: str, time_column: str,
    panel_key: str, report_path: str, tables: dict[str, str],
) -> dict[str, Any]:
    """Collect an explicitly specified empirical configuration and its artifacts."""
    project = root / name
    config_path = project / config_name
    result: dict[str, Any] = {
        "project": name, "config": str(config_path), "exists": project.is_dir(),
        "report": str(project / report_path),
        "report_exists": (project / report_path).is_file(),
    }
    if not config_path.is_file():
        result["error"] = "Empirical config missing"
        return result
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result["data_source"] = config["data"].get("source")
    result["provider"] = config["data"].get("provider") or config.get("real_data", {}).get("provider")
    if result["data_source"] == "synthetic":
        raise ValueError(f"Refusing synthetic configuration: {config_path}")
    result["panel"] = summarize_panel(project / config["data"][panel_key], time_column)
    result["tables"] = {key: read_table(project / value) for key, value in tables.items()}
    if name == "risk_aware_portfolio":
        result["external_alpha_path"] = config["data"].get("alpha_path")
        result["score_to_daily_return_scale_assumption"] = config["alpha"]["scale"]
        result["evaluation_start"] = config["backtest"]["start_date"] if "start_date" in config["backtest"] else config["backtest"].get("evaluation_start")
    if name == "market_microstructure_execution":
        result["market_rules"] = config["market"]
    return result


def render_report(audit: dict[str, Any]) -> str:
    """Make the machine inventory readable without turning it into an alpha verdict."""
    lines = [
        "# 当前项目与真实数据检查", "",
        f"检查时间（UTC）：{audit['checked_at_utc']}", "",
        "本检查仅核对本地真实数据配置、面板维度和报告存在性；不证明收益有效、PIT 数据完整或可实盘。", "",
        "| 项目 | 数据行数 | 标的数 | 起止时间 | 报告存在 |", "|---|---:|---:|---|---|",
    ]
    for project in audit["projects"]:
        panel = project.get("panel", {})
        interval = f"{panel.get('first_timestamp', '缺失')} — {panel.get('last_timestamp', '缺失')}"
        lines.append(f"| {project['project']} | {panel.get('rows', '缺失')} | {panel.get('assets', '缺失')} | {interval} | {project['report_exists']} |")
    lines += ["", "## ML 项目", "", audit["ml_project_status"], "", "## 当前连接障碍", ""]
    lines.extend(f"- {item}" for item in audit["integration_blockers"])
    lines += ["", "详细原始指标和股票代码见同目录 readiness.json。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    root = args.workspace.resolve()
    definitions = [
        ("Robust_Alpha_Discovery", "configs/akshare_smoke.yaml", "date", "path",
         "reports/generated/akshare_smoke/validation_report.md", {
             "ic": "reports/generated/akshare_smoke/tables/ic_summary.csv",
             "multiple_testing": "reports/generated/akshare_smoke/tables/multiple_testing.csv",
             "walk_forward": "reports/generated/akshare_smoke/tables/walk_forward_portfolio_performance.csv",
         }),
        ("risk_aware_portfolio", "configs/ashare.yaml", "date", "input_path",
         "outputs/ashare/research_report.md", {"performance": "outputs/ashare/tables/performance_summary.csv"}),
        ("market_microstructure_execution", "configs/base.yaml", "timestamp", "input_path",
         "outputs/research_report.md", {
             "execution": "outputs/tables/execution_summary.csv",
             "signal": "outputs/tables/signal_summary.csv",
         }),
        ("nonstationary_financial_ml", "configs/base.yaml", "date", "input_path",
         "outputs/research_report.md", {
             "model_performance": "outputs/tables/model_performance.csv",
             "portfolio_summary": "outputs/tables/portfolio_summary.csv",
         }),
    ]
    projects = [inspect_project(root, *definition) for definition in definitions]
    audit = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(root),
        "verification_scope": "Local empirical artifact inventory; no download, training, holdout access or orders",
        "projects": projects,
        "ml_project_status": "第四个项目 nonstationary_financial_ml 已创建；tmp 是临时脚本和依赖缓存，ML 项目在独立目录。是否完成报告可从上表和 artifact_verification.json 核对。",
        "integration_blockers": [
            "发现项目的 2018—2020 日频样本与风险项目的 2022—2023 日频样本未统一，股票池也不同。",
            "ML 与风险项目已提供独立历史 OOS 桥接实验；同一个 2023 样本的再分析不是独立确认。",
            "发现项目预测次日开盘至后日开盘；风险项目按额外滞后后的收盘收益核算，不能仅改列名连接。",
            "风险项目把外部 alpha 当分数并重新标准化，固定 scale=0.0006 不属于样本外预测收益校准。",
            "风险项目当前采用平衡固定股票池；动态上市退市股票池需要新增生命周期及交易规则处理。",
            "订单簿项目仅有一个股票交易日；A 股 T+1 下当前空账户回放主要是买入并持有估值。",
            "历史股票池、行业/股本可用时间、涨跌停、公司行动和真实成本/成交反馈尚不足以支持实盘结论。",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "readiness.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "readiness.md").write_text(render_report(audit), encoding="utf-8")
    for project in projects:
        panel = project.get("panel", {})
        print(f"{project['project']}: rows={panel.get('rows')}, assets={panel.get('assets')}, report={project['report_exists']}")
    print(f"Inventory: {args.output / 'readiness.md'}")


if __name__ == "__main__":
    main()
