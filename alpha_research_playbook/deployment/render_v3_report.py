"""Separate report figures from verified V3 artifacts; no model or result changes."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "tmp/v3_plot_cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def render():
    experiment = ROOT / "nonstationary_financial_ml/outputs/research_v3/historical_integration"
    manifest_path = experiment / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or not (experiment / "verification.json").exists():
        raise ValueError("Report requires a complete, independently verified experiment")
    output = ROOT / "nonstationary_financial_ml/outputs/v3_validation/v3_figures"
    output.mkdir(exist_ok=True)
    performance = pd.read_csv(experiment / "portfolio/portfolio_performance.csv")
    names = ["A_REV", "B_Ridge", "C_HGB", "D_HGB_risk", "E_HGB_risk_cost"]
    sample = performance.loc[performance.cost_bps_per_full_l1.eq(10)].set_index("strategy").loc[names]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    locations = np.arange(len(names))
    axes[0].bar(locations - .17, sample.nav_valid_dates, .34, label="Valid NAV endpoints", color="#246b8e")
    axes[0].bar(locations + .17, sample.valid_return_dates, .34, label="Valid period returns", color="#d88a2d")
    axes[0].axhline(294, color="#8b8d8e", linestyle="--", linewidth=1, label="294 scheduled dates")
    axes[0].set_xticks(locations, ["A REV", "B Ridge", "C HGB", "D risk", "E risk+cost"])
    axes[0].set_ylim(0, 320)
    axes[0].set_ylabel("Dates, 10 bp scenario")
    axes[0].legend(fontsize=8, loc="upper right")
    periods = pd.read_parquet(experiment / "portfolio/portfolio_periods.parquet")
    c = periods.loc[periods.strategy.eq("C_HGB") & periods.cost_bps_per_full_l1.eq(10)]
    axes[1].plot(c.date, c.nav, color="#246b8e", linewidth=1.3)
    axes[1].scatter(c.loc[c.nav_valid, "date"], c.loc[c.nav_valid, "nav"], s=4, color="#246b8e")
    axes[1].scatter(c.loc[~c.nav_valid, "date"], np.full((~c.nav_valid).sum(), .02),
                    marker="|", transform=axes[1].get_xaxis_transform(), color="#bc4545", label="Unvalued dates")
    axes[1].set_ylabel("C HGB: confirmed scenario NAV / initial cash")
    axes[1].set_xlim(c.date.min(), c.date.max())
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=8)
    figure.suptitle("V3 viewed historical diagnostics: incomplete valuation; no PIT / full-year performance claim", fontsize=10)
    figure.savefig(output / "coverage_and_nav.png", dpi=180)
    plt.close(figure)
    comparison = pd.read_csv(experiment / "oos_model_comparison.csv")
    figure, axis = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    labels = ["Raw Pearson IC", "Raw RankIC (primary)", "Neutralized Pearson IC", "Neutralized RankIC"]
    for i, row in comparison.iterrows():
        axis.plot([row.ci_low, row.ci_high], [i, i], color="#246b8e", linewidth=2)
        axis.scatter([row["mean"]], [i], color="#d88a2d", s=35, zorder=3)
    axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
    axis.set_yticks(range(4), labels)
    axis.invert_yaxis()
    axis.set_xlabel("HGB minus Ridge: paired daily mean and HAC 95% interval")
    axis.set_title("294 viewed dates; ranking increment is not established", fontsize=11)
    figure.savefig(output / "paired_ic.png", dpi=180)
    plt.close(figure)
    lineage = {"report_generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "experiment_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.png")}}
    (output / "figure_lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    render()
