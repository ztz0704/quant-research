"""Auditable research findings with honest A-share economic interpretation."""

import hashlib
import html
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from market_microstructure.utils.paths import project_path


def markdown_table(frame: pd.DataFrame) -> str:
    """Small Markdown tables without adding a tabulate dependency."""
    def cell(value) -> str:
        if pd.isna(value):
            return "unavailable"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value)
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def reproducibility_audit(config: dict, features: pd.DataFrame, predictions: pd.DataFrame,
                          figures: list[str]) -> dict:
    """Hash actual inputs, lock/config and retained OOS artifacts; no fabricated provenance."""
    names = [config["data"]["input_path"], config["data"]["features_path"],
             "data/processed/predictions.parquet", "uv.lock"]
    digests = {}
    for name in names:
        with project_path(name).open("rb") as stream:
            digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    config_json = json.dumps(config, sort_keys=True).encode()
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "seed": config["seed"],
            "data_provider": config["data"].get("provider"), "synthetic_primary_results": False,
            "raw_scope": "000400.SZ, 2022-06-01, public messages reconstructed internally",
            "sample_rows": len(features), "oos_rows_all_models": len(predictions),
            "oos_start_utc": predictions.timestamp.min().isoformat(),
            "oos_end_utc": predictions.timestamp.max().isoformat(),
            "config_sha256": hashlib.sha256(config_json).hexdigest(), "file_sha256": digests,
            "package_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn",
                                                                 "statsmodels", "matplotlib", "pyarrow", "akshare", "easyquotation")},
            "figure_count": len(figures), "figures": figures}


def write_report(config: dict, features: pd.DataFrame, predictions: pd.DataFrame,
                  figures: list[str], aggregate: pd.DataFrame, ablations: pd.DataFrame) -> None:
    """Write Markdown, a standalone browsable HTML gallery and reproducibility manifest."""
    def csv(name: str) -> pd.DataFrame:
        return pd.read_csv(project_path(f"outputs/tables/{name}.csv"))
    execution, signal, stress = csv("execution_summary"), csv("signal_summary"), csv("robustness_summary")
    half_life, regimes, markouts = csv("signal_half_life"), csv("regime_summary"), csv("maker_markouts")
    reconstruction = json.loads(project_path("outputs/tables/reconstruction_audit.json").read_text(encoding="utf-8"))
    source_manifest_path = project_path("outputs/tables/public_sample_manifest.json")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2, allow_nan=False), encoding="utf-8")
    chosen = execution.loc[(execution.model == config["model"]["type"]) & (execution["mode"] == "taker")].iloc[0]
    passive = execution.loc[execution.model == "always_long"].iloc[0]
    ic = signal.loc[signal.model == config["model"]["type"]].iloc[0]
    importance = csv("feature_importance")
    important = (importance.loc[importance.model == "forest"].groupby("feature").permutation_mean
                 .mean().nlargest(3))
    maker_mean = markouts.signed_markout_bps.mean()
    major = execution[["model", "mode", "gross_pnl", "net_pnl", "total_fees", "trade_count", "residual_inventory_abs", "closed_lot_count"]]
    narrative = [
        "# Market Microstructure & Execution — real A-share research report",
        "## Scope and provenance",
        f"Primary data: **000400.SZ (Xu Ji Electric), 2022-06-01**, {len(features):,} reconstructed five-level snapshots. "
        f"The adapter replayed {reconstruction['selected_message_rows']:,} messages from a pinned public sample. "
        f"{reconstruction['crossed_snapshot']} crossed states were counted and excluded; no missing-order reference or over-reduction was recorded. "
        "There is no official exchange snapshot for independent verification. These are reconstructed public historical observations, not certified vendor L2 snapshots.",
        "[Pinned raw-data source](https://github.com/Agnes252/LOB-Reconstruction/tree/12c60567d4a20aece2dd6e27f63e30339ec233e7/data_test). "
        "See tables/public_sample_manifest.json for SHA-256, tables/reconstruction_audit.json for replay accounting, and tables/ashare_source_probe.json for live-provider failures/staleness. "
        "AKShare and Easyquotation Python adapters were probed; live holiday quotes were not substituted into historical research.",
        f"OOS interval: {predictions.timestamp.min().tz_convert('Asia/Shanghai').isoformat()} through "
        f"{predictions.timestamp.max().tz_convert('Asia/Shanghai').isoformat()}. Complete chronological test windows are retained; "
        "this endpoint is the last OOS quote, not an official closing auction. Labels and rolling features reset at lunch/session boundaries.",
        "## Main findings",
        f"The selected logistic signal has within-day OOS IC **{ic.ic:.3f}**, with timestamp-block bootstrap interval "
        f"[{ic.ic_low:.3f}, {ic.ic_high:.3f}]. This interval measures conditional uncertainty within one day; it does not establish cross-day or cross-stock stability.",
        f"Selected taker net **marked** PnL is **CNY {chosen.net_pnl:.2f}**, with {int(chosen.trade_count)} fills and "
        f"{int(chosen.residual_inventory_abs)} shares left. **No lots close** under same-day T+1; positive PnL is held-stock appreciation, not realized intraday round-trip profit. "
        f"The identical-rules constant-long benchmark marks **CNY {passive.net_pnl:.2f}**. Thus prediction and gross return must be assessed against directional exposure, fees and delayed liquidation.",
        "The taker and benchmark hold different final share quantities and invested notionals because of capacity and lot constraints. "
        "Raw PnL differences are not exposure-normalized excess returns and cannot independently establish signal underperformance or alpha.",
        "## Statistical evaluation",
        markdown_table(aggregate),
        "Linear metrics are in bps; classification is {-1, 0, +1} with a 0.5 bps neutral band. "
        "AUC is macro one-vs-rest. Balanced accuracy/macro precision and recall complement accuracy. "
        "All preprocessing fits training data only. Expanding folds purge immature labels and embargo the boundaries. "
        "Validation permutation importance never uses test labels. Overlapping labels remain dependent despite purging.",
        "## Fixed feature ablations",
        markdown_table(ablations.drop(columns=["model", "rows", "labeled_rows"], errors="ignore")),
        "These feature sets were fixed before the report and share fold boundaries, regularization and label rules. "
        "They assess incremental predictive content on this day; they are not a feature search or an independent discovery sample. "
        f"Top mean validation permutation features: {', '.join(important.index)}. "
        "OLS HAC t statistics and coefficient drift are in feature_importance.csv. Highly related imbalance, spread and microprice variables make joint coefficients unstable; importance does not establish causality.",
        "## Prediction versus execution",
        markdown_table(major),
        "Mid-price execution is labeled **UNREALISTIC UPPER BOUND** and charges no costs. Different fill paths and capacities mean it is not a mathematical upper bound on every maker result. "
        "Takers buy at ask and sell at bid with capacity, adverse slippage and temporary impact. "
        "Maker results use fixed-price resting limits, future-observation eligibility, approximate queue pressure and seeded probabilistic fills. "
        "Their apparent ranking is sensitive to the uncalibrated probability model and different inventory exposure; it is not exchange-matching evidence.",
        "## PnL accounting and A-share constraints",
        "Net marked PnL = realized gross alpha + gross inventory mark-to-market − spread cost − fees − slippage − impact. "
        "The event ledger reconciles this bridge against cash + marked inventory − initial capital at every timestamp. "
        "Cash-stock mode enforces no naked shorts, positive cash, 100-share buy submissions and T+1 settled-stock sales. Partial fills are integer shares. "
        "Commission: configured 3 bps with CNY 5 minimum per submitted order, charged incrementally across partial maker fills. "
        "Historical sell stamp tax is 10 bps and transfer fee is 0.1 bps; the separate live config uses 5 bps stamp tax after the 2023 cut. "
        "Broker commission is an assumption, not a universal tariff. Same-day sell taxes remain uncharged where T+1 blocks every sale. "
        "Closed-trade hit rate and completed holding periods are unavailable here. Sharpe is fixed-clock, unannualized, and has little inferential value for one day.",
        "[Official stamp-tax reduction](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html). "
        "[SZSE rules applicable to the 2022 sample](https://docs.static.szse.cn/www/lawrules/rule/trade/W020210331716400647184.pdf). Fee dates and account scope are explicit in configs/base.yaml.",
        "## Maker markouts, decay and regimes",
        f"Mean signed 10-observation maker markout: **{maker_mean:.3f} bps**, from {len(markouts)} simplified-model fills. "
        "Negative signed markout denotes adverse selection; spread capture and markout are recorded separately from total marked PnL. Future prices are used only for diagnostics.",
        markdown_table(half_life),
        "Half-life is the first sampled horizon after peak |IC| reaching half of that peak, rather than an exponential fitted parameter. "
        "Retained observations are approximately one second, with gaps where invalid books are excluded. "
        "Training-only median cutoffs define six marginal regimes. Each low/high pair partitions events; regimes across dimensions overlap, so their PnL must not be summed together. "
        "Maker fill rate is attributed to order-arrival cohorts, even if fills occur later in another regime.",
        markdown_table(regimes.loc[regimes["mode"] == "taker", ["regime", "rows", "signal_ic", "prediction_accuracy", "net_pnl", "spread_cost"]]),
        "## Conditional execution stresses",
        markdown_table(stress[["scenario", "parameter", "net_pnl", "trade_count", "residual_inventory_abs", "max_drawdown"]]),
        "Signals are frozen OOS predictions except explicit added noise. The volatility shock scales the observed price path, so it also scales its directional appreciation; "
        "a larger PnL under that shock is not evidence of robustness to a new stochastic market regime. Spread stresses may create prices off the exchange tick grid and are mathematical cost sensitivities. "
        "Liquidity, commission and latency stresses are separate scenarios; no monotone latency result is assumed.",
        "## Optimal execution",
        "160 schedule/parameter cases compare TWAP, historical-flow forecast VWAP-style, front/back loading and exact discrete Almgren–Chriss. "
        "Volatility and half-spread are estimated only from the initial training period. Eta and gamma are explicit uncalibrated sensitivity inputs. "
        "A tridiagonal solver minimizes expected shortfall + lambda × variance. With a = eta/tau − gamma/2 > 0, "
        "E[C] = half_spread × X + gamma × X²/2 + a × sum(n_k²); Var(C) = sigma² × tau × sum(x_k²), with post-trade inventory. "
        "Continuous optimal schedules and conserving 100-share rounded schedules are both reported. Rounding is not an integer optimizer, and no observed execution-quality claim is made. "
        "The optimizer assumes already-settled stock for liquidation; this is a separate mandate from the empty-account signal backtest. "
        "[Almgren and Chriss](https://doi.org/10.21314/JOR.2001.041).",
        "## Answers to the research questions",
        "1. Imbalance has measurable within-day predictive content in the fixed ablation; one stock-day cannot establish a durable edge.\n"
        "2. Spread/depth/microprice add joint information as measured by the ablation and validation importance, subject to collinearity and sampling uncertainty.\n"
        "3. Permutation and HAC coefficient tables identify descriptive associations, not causal drivers.\n"
        "4. Accuracy and IC do not establish executable intraday profitability; T+1 and the passive benchmark matter.\n"
        "5. Spread, commissions, slippage and impact are separately measured and reconciled.\n"
        "6. Maker/taker rankings depend on inventory exposure and uncertain fill assumptions.\n"
        "7. Regime tables show conditional variation within this day, not cross-day robustness.\n"
        "8. Sampled IC decay and the explicitly defined half-life summarize horizon sensitivity.\n"
        "9. Latency sensitivity is empirical and non-monotone under this constrained one-day mandate.\n"
        "10. AC traces a cost-risk frontier under its assumed impact model; it is not a calibrated execution service.",
        "## Remaining limitations and next research steps",
        "Obtain licensed multi-day, multi-stock historical L2 and official snapshots; independently reconcile reconstruction; calibrate queue fills with order-level own-trade data; "
        "include initial settled inventories or multi-day liquidation to study cash-stock intraday timing; model corporate actions, price limits, auctions, suspensions, ST-specific rules and broker routing; "
        "validate stability on a later untouched time period. Public library snapshots cannot supply a certified historical five-level tape. "
        "Synthetic artifacts live under engineering_validation and are excluded from this report. No live orders were placed.",
        "## Figures",
    ]
    for figure in figures:
        title = Path(figure).stem.replace("_", " ")
        narrative.append(f"### {title}\n\n![{title}]({figure})")
    report = "\n\n".join(narrative) + "\n"
    project_path("outputs/research_report.md").write_text(report, encoding="utf-8")
    audit = reproducibility_audit(config, features, predictions, figures)
    project_path("outputs/tables/reproducibility_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    gallery = "\n".join(f'<figure><img loading="lazy" src="{figure}"><figcaption>{html.escape(Path(figure).stem.replace("_", " "))}</figcaption></figure>' for figure in figures)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Real A-share microstructure report</title>
    <style>body{{max-width:1100px;margin:40px auto;font:16px/1.6 Segoe UI,Arial;color:#193040;padding:0 24px}}
    h1,h2{{color:#126b85}}table{{border-collapse:collapse;font-size:13px;display:block;overflow:auto}}
    th,td{{padding:7px 10px;border-bottom:1px solid #dfe6eb}}img{{width:100%;height:auto}}figure{{margin:28px 0}}
    .note{{background:#eff4f6;padding:20px;border-left:4px solid #126b85}}pre{{white-space:pre-wrap;font:14px/1.6 Consolas}}
    </style><h1>Market Microstructure & Execution</h1><p>000400.SZ · 2022-06-01 · real public messages · reconstructed L2</p>
    <div class="note">Logistic taker net marked PnL: CNY {chosen.net_pnl:.2f}. Constant-long benchmark: CNY {passive.net_pnl:.2f}.
    {int(chosen.residual_inventory_abs)} shares remain. No same-day round trips close under T+1. One day and an uncalibrated maker model do not establish a tradable edge.</div>
    <p><a href="research_report.md">Complete Markdown research report</a> · <a href="../README.md">Methodology and reproduction</a></p>
    <h2>Prediction metrics</h2>{aggregate.to_html(index=False, float_format=lambda x: f'{x:.4f}', na_rep='unavailable')}
    <h2>Executed marked results</h2>{major.to_html(index=False, float_format=lambda x: f'{x:.2f}', na_rep='unavailable')}
    <h2>Full research text</h2><pre>{html.escape(report[:report.index('## Figures')])}</pre>
    <h2>All figures</h2>{gallery}</html>'''
    project_path("outputs/research_report.html").write_text(page, encoding="utf-8")
