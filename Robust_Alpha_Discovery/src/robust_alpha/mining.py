"""Real-data factor mining, frozen candidate evaluation and dated score export."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from robust_alpha.config import ConfigError, load_config, project_path
from robust_alpha.data import add_eligibility, add_open_to_open_target, load_panel
from robust_alpha.data.schema import REQUIRED_COLUMNS
from robust_alpha.evaluation import daily_spearman_ic, performance_metrics, summarize_ic
from robust_alpha.features import compute_features
from robust_alpha.features.specs import SIGNAL_SPECS
from robust_alpha.portfolio import (
    apply_linear_costs,
    calculate_turnover,
    portfolio_returns,
    quantile_weights,
)
from robust_alpha.preprocessing import cross_sectional_transform


def file_hash(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def history_hash(panel: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(panel[list(REQUIRED_COLUMNS)], index=False)
    return hashlib.sha256(values.to_numpy().tobytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {str(name.relative_to(root)): file_hash(name) for name in sorted(root.rglob("*.py"))}


def write_json(filename: Path, values: dict[str, Any]) -> None:
    filename.write_text(
        json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )


def settings(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    cfg = load_config(config_path)
    if cfg["data"]["source"] != "parquet":
        raise ConfigError(
            "Mining needs real Parquet for date filters; convert CSV using import_csv_panel.py"
        )
    mining = cfg.get("mining", {})
    previous = None
    for name in ("train", "validation", "test"):
        dates = mining.get(name)
        if not isinstance(dates, list) or len(dates) != 2:
            raise ConfigError(f"mining.{name} must be [start, end]")
        start, end = map(pd.Timestamp, dates)
        if start > end or (previous is not None and previous >= start):
            raise ConfigError("Mining periods must be chronological and disjoint")
        previous = end
    if mining.get("test_scope") not in {"exploratory", "untouched"}:
        raise ConfigError("Declare mining.test_scope: exploratory or untouched")
    for name in ("min_train_dates", "min_validation_dates", "max_selected"):
        if type(mining.get(name)) is not int or mining[name] < 1:
            raise ConfigError(f"mining.{name} must be a positive integer")
    if not 0 < mining.get("p_threshold", 0) < 1:
        raise ConfigError("mining.p_threshold must be in (0,1)")
    if not 0 < mining.get("max_correlation", 0) < 1:
        raise ConfigError("mining.max_correlation must be in (0,1)")
    if mining.get("screen_cost_bps") not in cfg["costs"]["bps"]:
        raise ConfigError("screen_cost_bps must be one of costs.bps")
    if not 0 < mining.get("min_return_coverage", 0) <= 1:
        raise ConfigError("min_return_coverage must be in (0,1]")
    destination = project_path(cfg["output"]["directory"])
    destination.mkdir(parents=True, exist_ok=True)
    return cfg, destination


def prepare(cfg: dict[str, Any], end: str, *, labels: bool = True) -> tuple[pd.DataFrame, str]:
    source = project_path(cfg["data"]["path"])
    # Parquet predicate filters prevent loading later price/label rows during discovery.
    raw = load_panel(source, date_end=end)
    if not raw.asset.str.fullmatch(r"[0236]\d{5}").all():
        raise ValueError("Mining input must identify Shanghai/Shenzhen A shares by six digits")
    identity = history_hash(raw)
    panel = add_eligibility(raw, min_history=cfg["data"]["min_history"])
    if labels:
        panel = add_open_to_open_target(panel, **cfg["timing"])
    panel = compute_features(panel, cfg["signals"], cfg.get("custom_signals"))
    panel = cross_sectional_transform(panel, cfg["signals"], method=cfg["preprocessing"]["method"])
    return panel, identity


def period_panel(panel: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """Reserve two known grid sessions for entry/exit before each stage's end."""
    calendar = pd.Series(sorted(panel.date.unique()))
    exits = pd.Series(calendar.shift(-2).to_numpy(), index=calendar)
    safe_dates = exits.index[exits.le(pd.Timestamp(dates[1]))]
    current = panel.loc[
        panel.date.between(*map(pd.Timestamp, dates)) & panel.date.isin(safe_dates)
    ].copy()
    # Eligibility/weights stay independent of future trading states.
    target = "target_o2o" if "target_o2o" in current else None
    if target and "exit_date" in current:
        current.loc[current.exit_date.gt(pd.Timestamp(dates[1])), target] = np.nan
    return current


def weights_and_daily(panel: pd.DataFrame, column: str, cfg: dict[str, Any]) -> pd.DataFrame:
    usable = panel.copy()
    count = usable.groupby("date")[column].transform("nunique")
    usable.loc[count.lt(2), column] = np.nan
    weights = quantile_weights(
        usable,
        column,
        kind="long_only",
        quantile=cfg["evaluation"]["quantile"],
        min_assets=cfg["evaluation"]["min_assets_per_date"],
    )
    target = cfg["timing"]["target_column"]
    gross = portfolio_returns(weights, usable, target_column=target)
    turnover = calculate_turnover(weights, usable, target_column=target)
    return apply_linear_costs(gross, turnover, cost_bps=cfg["costs"]["bps"])


def summarize_period(
    panel: pd.DataFrame, cfg: dict[str, Any], stage: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if panel.empty:
        raise ValueError(f"No data in {stage} window")
    target = cfg["timing"]["target_column"]
    columns = [f"z__{name}" for name in cfg["signals"]]
    daily = daily_spearman_ic(
        panel, columns, target_column=target, min_assets=cfg["evaluation"]["min_assets_per_date"]
    )
    daily["signal"] = daily.signal.str.removeprefix("z__")
    summary = summarize_ic(
        daily,
        annualization=cfg["evaluation"]["annualization"],
        hac_maxlags=cfg["evaluation"]["hac_maxlags"],
    )
    # Include every declared candidate in the family, including undefined/null ICs.
    p = summary.ic_hac_pvalue_two_sided.fillna(1).clip(0, 1)
    summary["holm_p"] = multipletests(p, method="holm")[1]
    benchmark_weights = panel[["date", "asset"]].copy()
    visible = panel.eligible
    size = visible.groupby(panel.date).transform("sum")
    benchmark_weights["weight"] = np.where(visible & size.gt(0), 1 / size.clip(lower=1), 0)
    benchmark = portfolio_returns(benchmark_weights, panel, target_column=target)
    portfolio_frames, rows = [], []
    for name in cfg["signals"]:
        paths = weights_and_daily(panel, f"z__{name}", cfg)
        paths = paths.merge(
            benchmark[["date", "gross_return"]].rename(
                columns={"gross_return": "benchmark_return"}
            ),
            on="date",
            validate="many_to_one",
        )
        paths["active_net_return"] = paths.net_return - paths.benchmark_return
        paths["signal"], paths["stage"] = name, stage
        portfolio_frames.append(paths)
        chosen = paths.loc[paths.cost_bps.eq(cfg["mining"]["screen_cost_bps"])]
        active = chosen.n_names.gt(0)
        valid = active & chosen.active_net_return.notna()
        complete = chosen.loc[active, "net_return"]
        performance = performance_metrics(
            complete, annualization=cfg["evaluation"]["annualization"]
        )
        row = {
            "signal": name,
            "net_mean": chosen.net_return.mean(),
            "active_net_mean": chosen.active_net_return.mean(),
            "n_decision_dates": int(active.sum()),
            "active_return_coverage": float(valid.sum() / active.sum()) if active.any() else 0.0,
            "mean_half_l1_turnover": chosen.turnover.mean(),
            "net_cagr": performance["cagr"] if complete.notna().all() else np.nan,
            "net_sharpe": performance["sharpe"] if complete.notna().all() else np.nan,
            "net_max_drawdown": performance["max_drawdown"] if complete.notna().all() else np.nan,
        }
        rows.append(row)
    summary = summary.merge(pd.DataFrame(rows), on="signal", validate="one_to_one")
    summary["stage"] = stage
    return summary, daily, pd.concat(portfolio_frames, ignore_index=True)


def training_correlations(panel: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    columns = [f"z__{name}" for name in names]
    matrices = [
        group[columns].corr(method="spearman")
        for _, group in panel.loc[panel.eligible].groupby("date", sort=True)
    ]
    if not matrices:
        return pd.DataFrame(np.nan, index=names, columns=names)
    combined = pd.concat(matrices).groupby(level=0).mean().reindex(index=columns, columns=columns)
    combined.index = names
    combined.columns = names
    return combined


def screen_candidates(
    train: pd.DataFrame, validation: pd.DataFrame, correlations: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Never read test metrics when accepting, ordering or deduplicating candidates."""
    result = train.drop(columns="stage").merge(
        validation.drop(columns="stage"),
        on="signal",
        suffixes=("_train", "_validation"),
        validate="one_to_one",
    )
    settings = cfg["mining"]
    result["accepted"] = False
    result["reason"] = ""
    selected = []
    for index in result.sort_values(
        ["mean_ic_validation", "signal"], ascending=[False, True]
    ).index:
        row = result.loc[index]
        failures = []
        for stage, required_dates in (
            ("train", settings["min_train_dates"]),
            ("validation", settings["min_validation_dates"]),
        ):
            if row[f"n_dates_{stage}"] < required_dates:
                failures.append(f"{stage}:insufficient_dates")
            if not np.isfinite(row[f"mean_ic_{stage}"]) or row[f"mean_ic_{stage}"] <= 0:
                failures.append(f"{stage}:nonpositive_ic")
            if row[f"holm_p_{stage}"] > settings["p_threshold"]:
                failures.append(f"{stage}:holm_not_passed")
        if not np.isfinite(row.active_net_mean_validation) or row.active_net_mean_validation <= 0:
            failures.append("validation:nonpositive_net_active_return")
        if row.active_return_coverage_validation < settings["min_return_coverage"]:
            failures.append("validation:incomplete_return_coverage")
        if not failures:
            if any(
                not np.isfinite(correlations.loc[row.signal, old])
                or abs(correlations.loc[row.signal, old]) > settings["max_correlation"]
                for old in selected
            ):
                failures.append("redundant_or_undefined_training_correlation")
            elif len(selected) >= settings["max_selected"]:
                failures.append("max_selected_reached")
            else:
                selected.append(row.signal)
                result.loc[index, "accepted"] = True
        result.loc[index, "reason"] = (
            ";".join(failures) if failures else "accepted_research_candidate"
        )
    return result.sort_values("signal").reset_index(drop=True)


def mine(config_path: str | Path) -> Path:
    cfg, destination = settings(config_path)
    panel, prefix = prepare(cfg, cfg["mining"]["validation"][1])
    train_panel = period_panel(panel, cfg["mining"]["train"])
    validation_panel = period_panel(panel, cfg["mining"]["validation"])
    summaries, all_ic, all_returns = [], [], []
    for stage, section in (("train", train_panel), ("validation", validation_panel)):
        summary, daily, returns = summarize_period(section, cfg, stage)
        daily["stage"] = stage
        summaries.append(summary)
        all_ic.append(daily)
        all_returns.append(returns)
    correlation = training_correlations(train_panel, cfg["signals"])
    screening = screen_candidates(*summaries, correlation, cfg)
    screening.to_csv(destination / "factor_screening.csv", index=False)
    correlation.to_csv(destination / "training_factor_correlation.csv")
    pd.concat(all_ic).to_csv(destination / "daily_ic.csv", index=False)
    pd.concat(all_returns).to_csv(destination / "portfolio_diagnostics.csv", index=False)
    feature_columns = [
        "date",
        "asset",
        "eligible",
        *cfg["signals"],
        *(f"z__{name}" for name in cfg["signals"]),
    ]
    panel[feature_columns].to_parquet(destination / "factor_values.parquet", index=False)
    latest = panel.loc[panel.date.eq(panel.date.max()), feature_columns]
    latest.to_csv(destination / "candidate_latest_research.csv", index=False)
    catalog = []
    for name in cfg["signals"]:
        custom = cfg.get("custom_signals", {}).get(name)
        spec = SIGNAL_SPECS.get(name)
        catalog.append(
            {
                "signal": name,
                "expression_or_family": custom["expression"] if custom else spec.family,
                "direction": custom.get("direction", 1) if custom else 1,
                "description": custom.get("description", "") if custom else str(spec.parameters),
            }
        )
    pd.DataFrame(catalog).to_csv(destination / "factor_catalog.csv", index=False)
    selected = screening.loc[screening.accepted, "signal"].tolist()
    selection = {
        "config_sha256": config_hash(cfg),
        "code_sha256": code_hashes(),
        "source_file_sha256_at_mining": file_hash(project_path(cfg["data"]["path"])),
        "history_through": cfg["mining"]["validation"][1],
        "history_sha256": prefix,
        "selected": selected,
        "selection_uses_test": False,
        "factor_screening_sha256": file_hash(destination / "factor_screening.csv"),
        "artifact_sha256": {
            name: file_hash(destination / name)
            for name in (
                "factor_values.parquet",
                "factor_catalog.csv",
                "daily_ic.csv",
                "portfolio_diagnostics.csv",
                "training_factor_correlation.csv",
                "candidate_latest_research.csv",
            )
        },
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "config": cfg,
        "semantics": "research candidates; not alpha certification or broker orders",
    }
    write_json(destination / "selection.json", selection)
    with (destination / "trials.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    key: selection[key]
                    for key in (
                        "config_sha256",
                        "source_file_sha256_at_mining",
                        "frozen_at_utc",
                        "selected",
                    )
                }
            )
            + "\n"
        )
    lines = [
        "# 真实 A 股因子挖掘",
        "",
        f"已计算 {len(cfg['signals'])} 个声明候选，接受 {len(selected)} 个研究候选。",
        "",
        f"训练 {cfg['mining']['train']}；验证 {cfg['mining']['validation']}。"
        "筛选未加载测试期行情。",
        "",
        "每个候选的拒绝理由见 factor_screening.csv；未通过时不强行挑选赢家。",
        "",
        "这些仍是固定股票池、公开行情、完整成交与线性成本下的研究筛选，未认证历史发布时点/真实成交。",
        "",
        "输出分数的收益区间：收盘 t 后形成信号，次日开盘进入、后日开盘退出；"
        "不能直接混入收盘收益项目。",
        "",
        "```text",
        screening[
            [
                "signal",
                "mean_ic_train",
                "holm_p_train",
                "mean_ic_validation",
                "holm_p_validation",
                "active_net_mean_validation",
                "accepted",
                "reason",
            ]
        ].to_string(index=False),
        "```",
        "",
    ]
    (destination / "mining_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Real-data mining: {len(cfg['signals'])} factors, {len(selected)} accepted; {destination}"
    )
    return destination


def read_selection(cfg: dict[str, Any], destination: Path) -> dict[str, Any]:
    selection = json.loads((destination / "selection.json").read_text(encoding="utf-8"))
    if selection["config_sha256"] != config_hash(cfg) or selection["code_sha256"] != code_hashes():
        raise ValueError("Frozen selection config/code changed; create a new mining experiment")
    if selection["factor_screening_sha256"] != file_hash(destination / "factor_screening.csv"):
        raise ValueError("Frozen factor screening was modified")
    screening = pd.read_csv(destination / "factor_screening.csv")
    accepted = screening.loc[screening.accepted.eq(True), "signal"].tolist()
    if selection["selected"] != accepted:
        raise ValueError("Selected factors disagree with the frozen screening decisions")
    for name, digest in selection["artifact_sha256"].items():
        if file_hash(destination / name) != digest:
            raise ValueError(f"Frozen mining artifact changed: {name}")
    raw = load_panel(project_path(cfg["data"]["path"]), date_end=selection["history_through"])
    if history_hash(raw) != selection["history_sha256"]:
        raise ValueError("Training/validation source history was revised; review and rerun mining")
    return selection


def evaluate(config_path: str | Path, *, confirm_untouched: bool = False) -> Path:
    cfg, destination = settings(config_path)
    selection = read_selection(cfg, destination)
    if cfg["mining"]["test_scope"] == "untouched" and not confirm_untouched:
        raise ValueError(
            "Untouched test requires explicit --confirm-untouched; do not reuse viewed data"
        )
    chosen = selection["selected"]
    results = pd.DataFrame(columns=["signal", "mean_ic", "holm_p", "stage"])
    if chosen:
        panel, _ = prepare(cfg, cfg["mining"]["test"][1])
        reduced = dict(cfg, signals=chosen)
        results, daily, returns = summarize_period(
            period_panel(panel, cfg["mining"]["test"]), reduced, "test"
        )
        daily.to_csv(destination / "test_daily_ic.csv", index=False)
        returns.to_csv(destination / "test_portfolio_diagnostics.csv", index=False)
    results.to_csv(destination / "test_results.csv", index=False)
    write_json(
        destination / "test_manifest.json",
        {
            "selected": chosen,
            "test_scope": cfg["mining"]["test_scope"],
            "test": cfg["mining"]["test"],
            "prices_loaded": bool(chosen),
            "selection_sha256": file_hash(destination / "selection.json"),
            "status": "evaluated_frozen_candidates"
            if chosen
            else "no_candidate_passed_no_test_prices_loaded",
        },
    )
    print(
        f"Frozen candidate evaluation: {len(chosen)} candidates; {destination / 'test_results.csv'}"
    )
    return destination


def score(
    config_path: str | Path,
    asof: str,
    *,
    historical: bool = False,
    expected_last_session: str | None = None,
) -> Path:
    cfg, destination = settings(config_path)
    selection = read_selection(cfg, destination)
    date = pd.Timestamp(asof).normalize()
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Shanghai")).date())
    if date < pd.Timestamp(selection["history_through"]):
        raise ValueError("Cannot score dates preceding the frozen validation cutoff")
    if not historical and (
        expected_last_session is None
        or date != pd.Timestamp(expected_last_session)
        or not 0 <= (today - date).days <= 10
    ):
        raise ValueError(
            "Current scores require an explicit recent expected last session; old data refused"
        )
    if not selection["selected"]:
        raise ValueError("No candidate passed screening; refusing to manufacture a score")
    panel, _ = prepare(cfg, str(date.date()), labels=False)
    if panel.date.max() != date:
        raise ValueError("Input has no quote on the requested decision date; stale scores refused")
    columns = [f"z__{name}" for name in selection["selected"]]
    section = panel.loc[panel.date.eq(date)].copy()
    valid = section.eligible & section[columns].notna().all(axis=1)
    if valid.sum() < cfg["evaluation"]["min_assets_per_date"]:
        raise ValueError("Insufficient complete current feature cross section")
    section["alpha"] = section[columns].mean(axis=1, skipna=False)
    section.loc[~valid, "alpha"] = np.nan
    section["signal_available"] = valid
    filename = destination / f"{'historical' if historical else 'dated'}_scores_{date.date()}.csv"
    section[["date", "asset", "alpha", "signal_available"]].to_csv(filename, index=False)
    write_json(
        filename.with_suffix(".json"),
        {
            "asof": str(date.date()),
            "historical": historical,
            "selected": selection["selected"],
            "source_sha256": file_hash(project_path(cfg["data"]["path"])),
            "selection_sha256": file_hash(destination / "selection.json"),
            "scores_sha256": file_hash(filename),
            "semantics": "uncalibrated score, not target weight or executable broker order",
            "label_interval": "open(t+1) to open(t+2), signal available after close(t)",
        },
    )
    print(filename)
    return filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["mine", "evaluate", "score"])
    parser.add_argument("--config", default="configs/mine_real.yaml")
    parser.add_argument("--asof")
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--expected-last-session")
    parser.add_argument("--confirm-untouched", action="store_true")
    args = parser.parse_args()
    if args.operation == "mine":
        mine(args.config)
    elif args.operation == "evaluate":
        evaluate(args.config, confirm_untouched=args.confirm_untouched)
    else:
        if not args.asof:
            parser.error("score requires --asof")
        score(
            args.config,
            args.asof,
            historical=args.historical,
            expected_last_session=args.expected_last_session,
        )


if __name__ == "__main__":
    main()
