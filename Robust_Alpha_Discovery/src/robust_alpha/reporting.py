"""Deterministic tables, figures, reports, and machine-readable run manifests."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_ic_summary(
    summary: pd.DataFrame, output: Path, *, dataset_label: str = "Synthetic demo"
) -> Path:
    """Plot mean IC with IID standard-error bars for descriptive diagnostics."""

    data = summary.sort_values("mean_ic").copy()
    error = data["ic_std"] / np.sqrt(data["n_dates"].clip(lower=1))
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.errorbar(data["mean_ic"], data["signal"], xerr=error, fmt="o", capsize=2)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_title(f"{dataset_label}: mean daily Spearman IC")
    axis.set_xlabel("Mean IC (error bar: IID standard error)")
    axis.set_ylabel("Signal")
    axis.grid(axis="x", alpha=0.25)
    return _save_figure(figure, output)


def plot_is_oos_scatter(
    selections: pd.DataFrame, output: Path, *, dataset_label: str = "Synthetic demo"
) -> Path | None:
    """Plot train IC against next-year test IC for selected signals."""

    data = selections.dropna(subset=["train_mean_ic", "test_mean_ic"])
    if data.empty:
        return None
    figure, axis = plt.subplots(figsize=(7, 6))
    for year, group in data.groupby("test_year", observed=True):
        axis.scatter(group["train_mean_ic"], group["test_mean_ic"], label=str(year), alpha=0.75)
    lower = float(min(data["train_mean_ic"].min(), data["test_mean_ic"].min()))
    upper = float(max(data["train_mean_ic"].max(), data["test_mean_ic"].max()))
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="grey", linewidth=1)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.axvline(0.0, color="black", linewidth=0.7)
    axis.set_title(f"{dataset_label}: train selection vs next-year OOS")
    axis.set_xlabel("Train mean IC")
    axis.set_ylabel("Test mean IC")
    axis.legend(title="Test year", fontsize=8)
    axis.grid(alpha=0.2)
    return _save_figure(figure, output)


def plot_discovery_counts(
    results: pd.DataFrame, output: Path, *, dataset_label: str = "Synthetic demo"
) -> Path:
    """Compare discovery counts across correction methods."""

    labels = ["Naive", "Bonferroni", "BH FDR", "Bootstrap maxT"]
    columns = ["naive_reject", "bonferroni_reject", "bh_reject", "max_t_reject"]
    counts = [int(results[column].fillna(False).sum()) for column in columns]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(labels, counts, color=["#5975a4", "#cc8963", "#5f9e6e", "#b55d60"])
    axis.bar_label(bars)
    axis.set_ylim(0, max(counts + [1]) * 1.2)
    axis.set_ylabel("Number of rejected hypotheses")
    axis.set_title(f"{dataset_label}: multiple-testing comparison")
    return _save_figure(figure, output)


def plot_bootstrap_distribution(
    maximum_statistics: np.ndarray,
    observed_maximum: float,
    output: Path,
    *,
    dataset_label: str = "Synthetic demo",
) -> Path:
    """Plot the joint null distribution of the maximum statistic."""

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(maximum_statistics, bins=35, alpha=0.8, color="#5975a4")
    axis.axvline(observed_maximum, color="#b55d60", linewidth=2, label="Observed maximum")
    axis.set_xlabel("Maximum absolute t-statistic")
    axis.set_ylabel("Bootstrap count")
    axis.set_title(f"{dataset_label}: joint block-bootstrap maxT null")
    axis.legend()
    return _save_figure(figure, output)


def plot_regime_heatmap(
    summary: pd.DataFrame, output: Path, *, dataset_label: str = "Synthetic demo"
) -> Path | None:
    """Plot volatility-regime mean IC by signal."""

    data = summary[summary["regime_dimension"] == "volatility"]
    pivot = data.pivot(index="signal", columns="state", values="mean_ic")
    if pivot.empty:
        return None
    pivot = pivot.sort_index()
    values = pivot.to_numpy(dtype=float)
    limit = np.nanmax(np.abs(values))
    limit = limit if np.isfinite(limit) and limit > 0 else 1.0
    figure, axis = plt.subplots(figsize=(6, 8))
    image = axis.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    axis.set_yticks(range(len(pivot.index)), labels=pivot.index)
    axis.set_title(f"{dataset_label}: mean IC by volatility regime")
    figure.colorbar(image, ax=axis, label="Mean IC")
    return _save_figure(figure, output)


def plot_portfolio_costs(
    portfolio_daily: pd.DataFrame,
    performance: pd.DataFrame,
    output: Path,
    *,
    dataset_label: str = "Synthetic demo",
) -> Path | None:
    """Plot cumulative net return at all configured costs for one diagnostic portfolio."""

    candidates = performance[
        (performance["portfolio"] == "long_short") & (performance["cost_bps"] == 0)
    ].dropna(subset=["sharpe"])
    if candidates.empty:
        return None
    selected_signal = candidates.sort_values("sharpe", ascending=False).iloc[0]["signal"]
    data = portfolio_daily[
        (portfolio_daily["signal"] == selected_signal)
        & (portfolio_daily["portfolio"] == "long_short")
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    for cost, group in data.groupby("cost_bps", sort=True, observed=True):
        ordered = group.sort_values("date").dropna(subset=["net_return"])
        wealth = (1.0 + ordered["net_return"]).cumprod()
        axis.plot(ordered["date"], wealth - 1.0, label=f"{cost:g} bps")
    axis.set_title(f"{dataset_label}: {selected_signal} cumulative return under costs")
    axis.set_ylabel("Cumulative simple return")
    axis.legend(title="Cost")
    axis.grid(alpha=0.2)
    return _save_figure(figure, output)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_versions(names: list[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def write_manifest(
    output: Path,
    *,
    config_path: Path,
    seed: int,
    artifacts: list[Path],
    source: str,
    research_phase: str,
    result_scope: str,
    panel: pd.DataFrame,
    input_path: Path | None = None,
) -> Path:
    """Write a hash-addressed machine-readable run manifest."""

    repository_root = config_path.parent.parent
    protocol_path = repository_root / "configs/protocols/factor_mining_v1.txt"
    lock_path = repository_root / "uv.lock"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_utc": datetime.now(UTC).isoformat(),
        "config": {"path": str(config_path), "sha256": _file_sha256(config_path)},
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git_state(repository_root),
        "packages": _package_versions(
            ["robust-alpha-discovery", "pandas", "numpy", "scipy", "statsmodels", "akshare"]
        ),
        "source": source,
        "research_phase": research_phase,
        "result_scope": result_scope,
        "panel": {
            "rows": len(panel),
            "assets": int(panel["asset"].nunique()),
            "dates": int(panel["date"].nunique()),
            "start_date": str(pd.to_datetime(panel["date"]).min().date()),
            "end_date": str(pd.to_datetime(panel["date"]).max().date()),
        },
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _file_sha256(path)}
            for path in artifacts
            if path.exists() and path.is_file()
        ],
    }
    if protocol_path.exists():
        payload["protocol"] = {
            "path": str(protocol_path),
            "sha256": _file_sha256(protocol_path),
        }
    if lock_path.exists():
        payload["lockfile"] = {"path": str(lock_path), "sha256": _file_sha256(lock_path)}
    if input_path is not None and input_path.exists() and input_path.is_file():
        payload["input"] = {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": _file_sha256(input_path),
        }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def write_research_report(
    output: Path,
    *,
    ic_summary: pd.DataFrame,
    multiple_testing: pd.DataFrame,
    global_pvalue: float,
    performance: pd.DataFrame,
    walk_forward_performance: pd.DataFrame,
    dataset_label: str,
    result_scope: str,
    research_phase: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_assets: int,
    is_synthetic: bool,
) -> Path:
    """Write a source-aware Markdown validation report."""

    best_ic = ic_summary.sort_values("mean_ic", ascending=False).head(5)
    discoveries = {
        "naive": int(multiple_testing["naive_reject"].sum()),
        "bonferroni": int(multiple_testing["bonferroni_reject"].sum()),
        "bh": int(multiple_testing["bh_reject"].sum()),
        "maxT": int(multiple_testing["max_t_reject"].sum()),
    }
    zero_cost = performance[
        (performance["portfolio"] == "long_short") & (performance["cost_bps"] == 0)
    ].sort_values("sharpe", ascending=False)
    best_portfolio = zero_cost.head(1)

    title = "Synthetic Pipeline Validation Report" if is_synthetic else "Research Validation Report"
    warning = (
        "This report validates code paths and research controls. It is not evidence of real, "
        "implementable alpha and must not be quoted as a market finding."
        if is_synthetic
        else "This public-data run is an integration and exploratory diagnostic. It is not a "
        "confirmatory alpha claim and is subject to the stated data and universe limitations."
    )
    lines = [
        f"# {title}",
        "",
        f"> {warning}",
        "",
        "## Run summary",
        "",
        f"- Dataset: {dataset_label}",
        f"- Research phase: `{research_phase}`",
        f"- Result scope: {result_scope}",
        f"- Signal-date range: {start_date.date()} to {end_date.date()}",
        f"- Assets represented: {n_assets}",
        f"- Candidate signals: {len(ic_summary)}",
        f"- Joint block-bootstrap global p-value: {global_pvalue:.4f}",
        f"- Discovery counts: {discoveries}",
        "",
        "## Highest descriptive mean IC values",
        "",
        "| Signal | Mean IC | ICIR | Dates |",
        "|---|---:|---:|---:|",
    ]
    for row in best_ic.itertuples(index=False):
        lines.append(f"| {row.signal} | {row.mean_ic:.6f} | {row.icir:.3f} | {row.n_dates} |")
    lines.extend(["", "## Diagnostic portfolio", ""])
    if best_portfolio.empty:
        lines.append("No valid diagnostic portfolio was produced.")
    else:
        row = best_portfolio.iloc[0]
        lines.extend(
            [
                f"The highest zero-cost single-signal long-short Sharpe was `{row['signal']}` ",
                f"with Sharpe `{row['sharpe']:.3f}`. Because this is selected from the same ",
                "evaluation table, it is descriptive and not a confirmatory result.",
            ]
        )
    lines.extend(["", "## Walk-forward selected portfolio", ""])
    walk = walk_forward_performance[
        (walk_forward_performance["portfolio"] == "long_short")
        & (walk_forward_performance["cost_bps"] == 0)
    ]
    if walk.empty:
        lines.append("No eligible walk-forward test year was available.")
    else:
        row = walk.iloc[0]
        lines.extend(
            [
                f"The train-only selected, equal-signal-weighted OOS portfolio covered "
                f"`{int(row['n_periods'])}` daily periods with zero-cost Sharpe "
                f"`{row['sharpe']:.3f}`.",
                "This is the primary selection-aware portfolio diagnostic; cost scenarios are in ",
                "`tables/walk_forward_portfolio_performance.csv`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/01_mean_ic.png`",
            "- `figures/02_is_oos_scatter.png` when enough calendar years are available",
            "- `figures/03_discovery_counts.png`",
            "- `figures/04_bootstrap_max_t.png`",
            "- `figures/05_volatility_regime_ic.png`",
            "- `figures/06_portfolio_costs.png`",
            "",
            "## Interpretation gate",
            "",
            "Review point-in-time universe, adjustment, tradeability, liquidity/capacity, and ",
            "holdout rules in `configs/protocols/factor_mining_v1.txt` before a market claim.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
