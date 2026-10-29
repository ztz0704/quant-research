"""End-to-end reproducible research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from robust_alpha.config import ConfigError, load_config, project_path
from robust_alpha.data import (
    add_eligibility,
    add_open_to_open_target,
    generate_synthetic_panel,
    load_panel,
    normalize_and_validate_panel,
    panel_quality_summary,
    save_panel,
)
from robust_alpha.evaluation import (
    daily_factor_returns,
    daily_spearman_ic,
    factor_return_summary,
    performance_metrics,
    summarize_ic,
)
from robust_alpha.evaluation.factor_return import factor_returns_wide
from robust_alpha.features import compute_features
from robust_alpha.portfolio import (
    apply_linear_costs,
    calculate_turnover,
    portfolio_returns,
    quantile_weights,
)
from robust_alpha.preprocessing import cross_sectional_transform
from robust_alpha.reporting import (
    plot_bootstrap_distribution,
    plot_discovery_counts,
    plot_ic_summary,
    plot_is_oos_scatter,
    plot_portfolio_costs,
    plot_regime_heatmap,
    write_manifest,
    write_research_report,
)
from robust_alpha.validation import (
    RESEARCH_PHASES,
    adjust_pvalues,
    attach_walk_forward_score,
    build_market_regimes,
    filter_evaluation_dates,
    filter_walk_forward_years,
    joint_moving_block_max_t,
    regime_ic_summary,
    resolve_research_window,
    walk_forward_ic_selection,
)
from robust_alpha.validation.splits import ResearchWindow


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _strip_signal_prefix(frame: pd.DataFrame, prefix: str = "z__") -> pd.DataFrame:
    result = frame.copy()
    result["signal"] = result["signal"].astype(str).str.removeprefix(prefix)
    return result


def _purge_targets_at_boundary(
    panel: pd.DataFrame,
    *,
    target_column: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    require_same_calendar_year: bool = False,
) -> pd.DataFrame:
    """Remove labels whose entry/exit window crosses an evaluation boundary."""

    result = panel.copy()
    valid = result["target_valid"].astype(bool)
    if start is not None:
        valid &= result["entry_date"].ge(start)
    if end is not None:
        valid &= result["exit_date"].le(end)
    if require_same_calendar_year:
        valid &= pd.to_datetime(result["exit_date"]).dt.year.eq(
            pd.to_datetime(result["date"]).dt.year
        )
    result["target_valid"] = valid
    result.loc[~valid, target_column] = pd.NA
    return result


def _build_walk_forward_portfolio_daily(
    scored_panel: pd.DataFrame,
    *,
    target_column: str,
    quantile: float,
    min_assets: int,
    cost_bps: list[int | float],
) -> pd.DataFrame:
    """Build OOS portfolios, resetting holdings at every independently scored test year."""

    active = scored_panel.loc[scored_panel["walk_forward_signal_count"].gt(0)].copy()
    active["test_year"] = pd.to_datetime(active["date"]).dt.year
    frames: list[pd.DataFrame] = []
    for portfolio_kind in ("long_short", "long_only"):
        for test_year, year_panel in active.groupby("test_year", sort=True, observed=True):
            weights = quantile_weights(
                year_panel,
                "walk_forward_score",
                quantile=quantile,
                kind=portfolio_kind,
                min_assets=min_assets,
            )
            gross = portfolio_returns(weights, year_panel, target_column=target_column)
            turnover = calculate_turnover(weights, year_panel, target_column=target_column)
            scenarios = apply_linear_costs(gross, turnover, cost_bps=cost_bps)
            scenarios["portfolio"] = portfolio_kind
            scenarios["test_year"] = int(test_year)
            frames.append(scenarios)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(
        columns=[
            "date",
            "gross_return",
            "n_names",
            "max_abs_weight",
            "missing_active_return",
            "turnover",
            "cost_bps",
            "cost",
            "net_return",
            "portfolio",
            "test_year",
        ]
    )


def _load_or_generate_data(
    config: dict[str, Any],
    *,
    root: Path,
    window: ResearchWindow,
) -> tuple[pd.DataFrame, list[Path]]:
    data_config = config["data"]
    seed = int(config["project"]["random_seed"])
    source = data_config["source"]
    artifacts: list[Path] = []
    if source == "synthetic":
        raw = generate_synthetic_panel(
            start_date=str(data_config["start_date"]),
            n_days=int(data_config["n_days"]),
            n_assets=int(data_config["n_assets"]),
            seed=seed,
        )
        raw = normalize_and_validate_panel(raw)
        synthetic_path = project_path(data_config["path"], root)
        artifacts.append(save_panel(raw, synthetic_path))
        return raw, artifacts
    if source not in {"parquet", "csv"}:
        raise ConfigError(f"Unsupported data.source: {source}")

    data_path = project_path(data_config["path"], root)
    panel = load_panel(data_path, date_end=window.load_end)
    return panel, artifacts


def run_pipeline(
    config_path: str | Path,
    *,
    include_final_holdout: bool = False,
    research_phase: str = "auto",
    stop_after: str = "report",
) -> dict[str, Path]:
    """Run the configured pipeline and return named artifact paths."""

    stages = ["panel", "features", "screen", "oos", "report"]
    if stop_after not in stages:
        raise ValueError(f"stop_after must be one of {stages}")
    root = Path.cwd().resolve()
    config_file = project_path(config_path, root)
    config = load_config(config_file)
    window = resolve_research_window(
        config,
        research_phase=research_phase,
        include_final_holdout=include_final_holdout,
    )
    output_directory = project_path(config["output"]["directory"], root)
    if window.phase != "all":
        output_directory = output_directory / window.phase
    table_directory = output_directory / "tables"
    figure_directory = output_directory / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    raw, raw_artifacts = _load_or_generate_data(config, root=root, window=window)
    if raw_artifacts:
        artifacts["source_panel"] = raw_artifacts[0]
    panel = add_eligibility(raw, min_history=int(config["data"]["min_history"]))
    panel = add_open_to_open_target(
        panel,
        target_column=config["timing"]["target_column"],
        require_entry_tradable=bool(config["timing"]["require_entry_tradable"]),
        require_exit_tradable=bool(config["timing"]["require_exit_tradable"]),
    )
    # Signal-date eligibility must not depend on future entry/exit tradeability. Target validity is
    # kept separately and only affects whether a realized return can be evaluated.
    panel["research_eligible"] = panel["eligible"]
    panel_path = project_path(config["output"]["panel_path"], root)
    if window.phase != "all":
        panel_path = panel_path.with_name(f"{panel_path.stem}_{window.phase}{panel_path.suffix}")
    artifacts["data_quality"] = _write_table(
        panel_quality_summary(panel), table_directory / "data_quality.csv"
    )
    if stop_after == "panel":
        artifacts["research_panel"] = save_panel(panel, panel_path)
        return artifacts

    signal_names = list(config["signals"])
    panel = compute_features(panel, signal_names, config.get("custom_signals"))
    panel = cross_sectional_transform(
        panel,
        signal_names,
        method=config["preprocessing"]["method"],
        eligible_column="eligible",
    )
    artifacts["research_panel"] = save_panel(panel, panel_path)
    if stop_after == "features":
        return artifacts

    target_column = config["timing"]["target_column"]
    transformed_signals = [f"z__{name}" for name in signal_names]
    evaluation = config["evaluation"]

    evaluation_panel = filter_evaluation_dates(panel, window)
    evaluation_panel = _purge_targets_at_boundary(
        evaluation_panel,
        target_column=target_column,
        start=window.evaluation_start,
        end=window.evaluation_end,
    )
    walk_panel = _purge_targets_at_boundary(
        panel,
        target_column=target_column,
        require_same_calendar_year=True,
    )
    daily_ic_all = daily_spearman_ic(
        walk_panel,
        transformed_signals,
        target_column=target_column,
        min_assets=int(evaluation["min_assets_per_date"]),
    )
    daily_ic_all = _strip_signal_prefix(daily_ic_all)
    daily_ic = daily_spearman_ic(
        evaluation_panel,
        transformed_signals,
        target_column=target_column,
        min_assets=int(evaluation["min_assets_per_date"]),
    )
    daily_ic = _strip_signal_prefix(daily_ic)
    ic_summary = summarize_ic(
        daily_ic,
        annualization=int(evaluation["annualization"]),
        hac_maxlags=int(evaluation["hac_maxlags"]),
    )
    daily_factor = daily_factor_returns(
        evaluation_panel,
        transformed_signals,
        target_column=target_column,
        min_assets=int(evaluation["min_assets_per_date"]),
    )
    daily_factor = _strip_signal_prefix(daily_factor)
    factor_summary = factor_return_summary(
        daily_factor, hac_maxlags=int(evaluation["hac_maxlags"])
    )
    artifacts["daily_ic"] = _write_table(daily_ic, table_directory / "daily_ic.csv")
    artifacts["ic_summary"] = _write_table(ic_summary, table_directory / "ic_summary.csv")
    artifacts["daily_factor_returns"] = _write_table(
        daily_factor, table_directory / "daily_factor_returns.csv"
    )
    artifacts["factor_return_summary"] = _write_table(
        factor_summary, table_directory / "factor_return_summary.csv"
    )
    if stop_after == "screen":
        return artifacts

    walk = walk_forward_ic_selection(
        daily_ic_all,
        train_years=int(config.get("walk_forward", {}).get("train_years", 5)),
        top_k=int(config.get("walk_forward", {}).get("top_k", 5)),
    )
    walk["selections"] = filter_walk_forward_years(walk["selections"], window)
    walk["combinations"] = filter_walk_forward_years(walk["combinations"], window)
    artifacts["walk_forward_selections"] = _write_table(
        walk["selections"], table_directory / "walk_forward_selections.csv"
    )
    artifacts["walk_forward_combinations"] = _write_table(
        walk["combinations"], table_directory / "walk_forward_combinations.csv"
    )

    factor_wide = factor_returns_wide(daily_factor)
    pvalues = factor_summary.set_index("signal")["hac_pvalue_two_sided"]
    multiple = adjust_pvalues(pvalues, alpha=float(config["multiple_testing"]["alpha"]))
    bootstrap = joint_moving_block_max_t(
        factor_wide,
        replications=int(config["multiple_testing"]["bootstrap_replications"]),
        block_length=int(config["multiple_testing"]["block_length"]),
        seed=int(config["project"]["random_seed"]),
        two_sided=True,
    )
    multiple = multiple.merge(bootstrap["signal_results"], on="signal", how="left")
    multiple["max_t_reject"] = multiple["max_t_adjusted_pvalue"] <= float(
        config["multiple_testing"]["alpha"]
    )
    artifacts["multiple_testing"] = _write_table(
        multiple, table_directory / "multiple_testing.csv"
    )
    artifacts["bootstrap_max_statistics"] = _write_table(
        pd.DataFrame({"max_t": bootstrap["max_statistics"]}),
        table_directory / "bootstrap_max_statistics.csv",
    )

    regime_dates = filter_evaluation_dates(build_market_regimes(panel), window)
    regime_summary = regime_ic_summary(daily_ic, regime_dates)
    artifacts["regime_dates"] = _write_table(regime_dates, table_directory / "regime_dates.csv")
    artifacts["regime_ic_summary"] = _write_table(
        regime_summary, table_directory / "regime_ic_summary.csv"
    )

    portfolio_frames: list[pd.DataFrame] = []
    for signal, transformed in zip(signal_names, transformed_signals, strict=True):
        for portfolio_kind in ("long_short", "long_only"):
            weights = quantile_weights(
                evaluation_panel,
                transformed,
                quantile=float(evaluation["quantile"]),
                kind=portfolio_kind,
                min_assets=int(evaluation["min_assets_per_date"]),
            )
            gross = portfolio_returns(weights, evaluation_panel, target_column=target_column)
            turnover = calculate_turnover(weights, evaluation_panel, target_column=target_column)
            scenarios = apply_linear_costs(
                gross, turnover, cost_bps=config["costs"]["bps"]
            )
            scenarios["signal"] = signal
            scenarios["portfolio"] = portfolio_kind
            portfolio_frames.append(scenarios)
    portfolio_daily = pd.concat(portfolio_frames, ignore_index=True)
    performance_rows: list[dict[str, Any]] = []
    for (signal, portfolio_kind, cost), group in portfolio_daily.groupby(
        ["signal", "portfolio", "cost_bps"], sort=True, observed=True
    ):
        metrics = performance_metrics(
            group.sort_values("date")["net_return"],
            annualization=int(evaluation["annualization"]),
        )
        performance_rows.append(
            {
                "signal": signal,
                "portfolio": portfolio_kind,
                "cost_bps": cost,
                "mean_turnover": group["turnover"].mean(),
                "missing_return_rate": group["missing_active_return"].mean(),
                **metrics,
            }
        )
    performance = pd.DataFrame(performance_rows)
    artifacts["portfolio_daily"] = _write_table(
        portfolio_daily, table_directory / "portfolio_daily.csv"
    )
    artifacts["portfolio_performance"] = _write_table(
        performance, table_directory / "portfolio_performance.csv"
    )

    scored_panel = attach_walk_forward_score(evaluation_panel, walk["selections"])
    # Warm-up years have no train-only selection and are not missing strategy returns.
    # Each scored calendar year is an independent OOS block because cross-year labels are
    # purged; resetting weights also prevents a gap in selected years from implying a
    # fictitious multi-year holding.
    walk_portfolio_daily = _build_walk_forward_portfolio_daily(
        scored_panel,
        target_column=target_column,
        quantile=float(evaluation["quantile"]),
        min_assets=int(evaluation["min_assets_per_date"]),
        cost_bps=config["costs"]["bps"],
    )
    walk_performance_rows: list[dict[str, Any]] = []
    for (portfolio_kind, cost), group in walk_portfolio_daily.groupby(
        ["portfolio", "cost_bps"], sort=True, observed=True
    ):
        metrics = performance_metrics(
            group.sort_values("date")["net_return"],
            annualization=int(evaluation["annualization"]),
        )
        walk_performance_rows.append(
            {
                "portfolio": portfolio_kind,
                "cost_bps": cost,
                "mean_turnover": group["turnover"].mean(),
                "missing_return_rate": group["missing_active_return"].mean(),
                "test_years": int(group.loc[group["n_names"].gt(0), "test_year"].nunique()),
                **metrics,
            }
        )
    walk_portfolio_performance = pd.DataFrame(
        walk_performance_rows,
        columns=[
            "portfolio",
            "cost_bps",
            "mean_turnover",
            "missing_return_rate",
            "test_years",
            "n_periods",
            "n_missing_periods",
            "return_coverage",
            "annualized_return",
            "cagr",
            "annualized_volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "win_rate",
            "cumulative_return",
        ],
    )
    artifacts["walk_forward_portfolio_daily"] = _write_table(
        walk_portfolio_daily, table_directory / "walk_forward_portfolio_daily.csv"
    )
    artifacts["walk_forward_portfolio_performance"] = _write_table(
        walk_portfolio_performance,
        table_directory / "walk_forward_portfolio_performance.csv",
    )
    if stop_after == "oos":
        return artifacts

    source = str(config["data"]["source"])
    is_synthetic = source == "synthetic"
    provider = str(config["data"].get("provider", source)).upper()
    dataset_label = "Synthetic demo" if is_synthetic else f"{provider} {window.phase}"
    figure_paths = [
        plot_ic_summary(
            ic_summary, figure_directory / "01_mean_ic.png", dataset_label=dataset_label
        ),
        plot_discovery_counts(
            multiple,
            figure_directory / "03_discovery_counts.png",
            dataset_label=dataset_label,
        ),
        plot_bootstrap_distribution(
            bootstrap["max_statistics"],
            bootstrap["observed_maximum"],
            figure_directory / "04_bootstrap_max_t.png",
            dataset_label=dataset_label,
        ),
    ]
    optional_figures = [
        plot_is_oos_scatter(
            walk["selections"],
            figure_directory / "02_is_oos_scatter.png",
            dataset_label=dataset_label,
        ),
        plot_regime_heatmap(
            regime_summary,
            figure_directory / "05_volatility_regime_ic.png",
            dataset_label=dataset_label,
        ),
        plot_portfolio_costs(
            portfolio_daily,
            performance,
            figure_directory / "06_portfolio_costs.png",
            dataset_label=dataset_label,
        ),
    ]
    figure_paths.extend(path for path in optional_figures if path is not None)
    for path in figure_paths:
        artifacts[path.stem] = path

    report_path = output_directory / "validation_report.md"
    artifacts["report"] = write_research_report(
        report_path,
        ic_summary=ic_summary,
        multiple_testing=multiple,
        global_pvalue=float(bootstrap["global_pvalue"]),
        performance=performance,
        walk_forward_performance=walk_portfolio_performance,
        dataset_label=dataset_label,
        result_scope=window.result_scope,
        research_phase=window.phase,
        start_date=pd.to_datetime(evaluation_panel["date"]).min(),
        end_date=pd.to_datetime(evaluation_panel["date"]).max(),
        n_assets=int(evaluation_panel["asset"].nunique()),
        is_synthetic=is_synthetic,
    )
    manifest_path = output_directory / "run_manifest.json"
    input_path = project_path(config["data"]["path"], root)
    artifacts["manifest"] = write_manifest(
        manifest_path,
        config_path=config_file,
        seed=int(config["project"]["random_seed"]),
        artifacts=list(artifacts.values()),
        source="synthetic" if is_synthetic else str(config["data"].get("provider", source)),
        research_phase=window.phase,
        result_scope=window.result_scope,
        panel=evaluation_panel,
        input_path=input_path,
    )
    return artifacts


def main() -> None:
    """Console entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--stop-after",
        choices=["panel", "features", "screen", "oos", "report"],
        default="report",
    )
    parser.add_argument(
        "--include-final-holdout",
        action="store_true",
        help="Load the configured final holdout only when the frozen protocol permits it.",
    )
    parser.add_argument(
        "--research-phase",
        choices=RESEARCH_PHASES,
        default="auto",
        help="Evaluation window; named phases require time_split in the configuration.",
    )
    arguments = parser.parse_args()
    artifacts = run_pipeline(
        arguments.config,
        include_final_holdout=arguments.include_final_holdout,
        research_phase=arguments.research_phase,
        stop_after=arguments.stop_after,
    )
    print(f"Completed stage: {arguments.stop_after}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
