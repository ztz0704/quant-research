"""Auditable real-data validation using the existing risk/backtest/report modules."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from risk_aware_portfolio.backtest.comparisons import STRATEGIES
from risk_aware_portfolio.backtest.engine import BacktestResult, run_backtest
from risk_aware_portfolio.reporting.research_report import markdown_table
from risk_aware_portfolio.risk.model import RiskHistory
from risk_aware_portfolio.utils.config import Config, variant
from risk_aware_portfolio.utils.manifest import file_hash
from risk_aware_portfolio.utils.paths import resolve_path


def verify_real_data(config: Config) -> dict:
    """Reject changed raw payloads, calendar, declarations or panel after preparation."""
    settings = config["real_data"]
    cache = resolve_path(settings["raw_dir"])
    manifest = json.loads((cache / "data_manifest.json").read_text(encoding="utf-8"))
    paths = {
        "panel_sha256": resolve_path(config["data"]["input_path"]),
        "quotes_sha256": cache / "quotes.parquet",
        "valuation_quotes_sha256": cache / "valuation_quotes.parquet",
        "universe_sha256": resolve_path(settings["universe_path"]),
        "suspensions_sha256": resolve_path(settings["suspension_path"]),
    }
    if manifest["config"] != config:
        raise ValueError("Real-data preparation config changed; rerun fetch_ashare_data")
    for key, path in paths.items():
        if file_hash(path) != manifest[key]:
            raise ValueError(f"Modified real input: {path}")
    for name, digest in manifest["raw_sha256"].items():
        if file_hash(cache / name) != digest:
            raise ValueError(f"Modified raw payload: {name}")
    return manifest


def run_primary_comparisons(
    panel: pd.DataFrame, history: RiskHistory, config: Config, output: Path
) -> dict[str, BacktestResult]:
    """Run all declared portfolios on the same calendar and costs."""
    results = {}
    for name, slug, baseline, neutral, penalty in STRATEGIES:
        settings = variant(
            config,
            portfolio={
                "cost_penalty": config["portfolio"]["cost_penalty"] if penalty is None else penalty
            },
            alpha={"neutralize": config["alpha"]["neutralize"] if neutral is None else neutral},
        )
        result = run_backtest(panel, settings, name, baseline, history)
        result.save(output / slug)
        results[slug] = result
        print(
            f"Real-data {name}: {len(result.returns)} sessions, NAV {result.returns['nav'].iloc[-1]:.6f}",
            flush=True,
        )
    return results


def real_no_lookahead_checks(
    panel: pd.DataFrame, config: Config, reference: BacktestResult
) -> pd.DataFrame:
    """Run actual-panel prefix and adversarial future-return/signal perturbation checks."""
    cutoff = reference.weights.index[min(30, len(reference.weights) - 2)]
    next_date = pd.DatetimeIndex(panel["date"].unique())
    next_date = next_date[next_date > cutoff][0]
    prefix = run_backtest(panel[panel["date"] <= next_date], config)
    changed = panel.copy()
    mask = changed["date"] > cutoff
    changed.loc[mask, "return"] *= -0.5
    changed.loc[mask, "alpha"] += 100.0
    changed.loc[mask, "size"] *= -2.0
    altered = run_backtest(changed, config)
    rows = []
    for name, candidate in (("prefix_invariance", prefix), ("future_perturbation", altered)):
        a, b = reference.weights.loc[:cutoff], candidate.weights.loc[:cutoff]
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        rows.append(
            {
                "check": name,
                "cutoff": cutoff,
                "passed": True,
                "maximum_weight_difference": float(np.max(np.abs(a.to_numpy() - b.to_numpy()))),
            }
        )
    return pd.DataFrame(rows)


def write_real_report(
    frames: dict[str, pd.DataFrame],
    results: dict[str, BacktestResult],
    config: Config,
    checks: pd.DataFrame,
    output: Path,
) -> None:
    """Measured empirical narrative without simulated truth or alpha profitability claims."""
    main = results["risk_aware_tc"]
    performance = frames["performance_summary"]
    columns = [
        "portfolio",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "average_turnover",
        "total_transaction_cost",
    ]
    risk = frames["risk_summary"].set_index("portfolio").loc[main.name]
    coverage = pd.read_csv(resolve_path(config["real_data"]["raw_dir"]) / "coverage.csv")
    failures = frames["robustness_summary"]["status"].eq("failed").sum()
    statuses = main.solver_diagnostics["status"].value_counts().to_dict()
    content = f"""# A-share Real Data Validation

Actual Sina A-share prices, decoded with AKShare. This is an engineering and risk-model
validation on a **retrospectively declared fixed basket**, not a validated trading strategy
or a survivorship-free market study. The synthetic results remain in their original directories.

## Declared sample and data audit

{len(main.weights.columns)} assets; {len(main.returns)} realized sessions,
{main.returns.index.min().date()} through {main.returns.index.max().date()}.
2022 is the fixed feature/risk warmup; 2023 is the predeclared evaluation year.
No asset was removed after download and no parameters were selected by realized Sharpe.
The native coverage audit contains {int(coverage["missing_sessions"].sum())} absent quote rows.
Only dates in the linked suspension declarations are marked at the last valid price;
unknown holes stop preparation. Native raw quotes remain separate from valuation marks.

## Timing, exposures and execution

Same-date returns are adjusted-close ratios with no implicit fill. Signals and exposures
have one complete session of lag: features observed at $t-1$ enter $w_t$, which earns
$r_{{t+1}}$. Covariance moments use regressions available through decision date $t$.
The close execution and full-fill assumption remains idealized; this is not an intraday
market-on-close implementation or a limit-order simulation.

Style exposures are log float-cap **proxy**, 126-minus-21-session log momentum,
63-session return volatility and log 21-session float turnover. Share counts and
backward adjustment factors are merged backward on vendor effective dates. Historical
publication/revision timestamps are unavailable: **vendor-level PIT certification is false**.
The six business groups are manually supplied coarse classifications, not certified
historical industry snapshots. Value/quality are omitted rather than populated with
current financial statements. Market cap is raw price times vendor float shares, not total cap.

The default input alpha is fixed five-session reversal, cross-sectionally standardized
and scaled by a predeclared 0.0006 expected-return coefficient. It has not passed
Robust Alpha Discovery and its forecast units are not empirically calibrated. An external
`date, asset, alpha` input can replace it through `data.alpha_path`, with complete date coverage.

## Costs and suspension handling

The same ex-post convex impact model applies to all four portfolios, including the
rank baseline. The objective penalty distinguishes the two optimized raw-alpha cases.
Sell-side stamp duty is 10 bps before 2023-08-28 and 5 bps from that date; the dated
rate enters both the objective and exact self-financing fee accounting. Generic symmetric
linear costs are 5 bps; impact is an assumed NAV-based approximation, not calibrated ADV capacity.
Broker minimum fees, board lots, transfer/handling fee details and dividend taxes are not modeled.
See [the tax authority's original rate](https://tianjin.chinatax.gov.cn/11200000000/0300/030005/20220725150235434.shtml),
[seller-only treatment](https://liaoning.chinatax.gov.cn/art/2022/8/3/art_99_86476.html) and
[the dated reduction](https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm).

If any constituent is suspended on a scheduled rebalance, **all trades are deferred**
until the basket is tradable, using updated as-of information at that date. This conservative
policy avoids fee-funded implicit trades in frozen positions. Held positions earn marked
zero return during halts and the actual price change on reopening. Halted rows are excluded
from the factor-return regression, while their residuals remain in accounting. There were
{int(main.returns["deferred_rebalance"].sum())} deferred sessions in the main portfolio.
Daily bar data does not certify fillability at price limits; T+1 settlement, cash dividend
timing, rights subscriptions and delisting cash flows still require production event accounting.

## Measured results

{markdown_table(performance[columns])}

Geometric annualized return; arithmetic daily Sharpe with zero risk-free rate; one-way
L1 turnover including initial deployment. Total cost is the sum of fee fractions, not
additive cumulative NAV drag. The attribution ledger links contributions to actual NAV.
These outcomes do not establish deployable alpha. The constrained policy and unconstrained
rank basket differ in several ways, so their comparison is not a pure optimizer causal effect.

{markdown_table(frames["pairwise_comparisons"])}

HAC tests concern paired mean daily returns, not Sharpe differences. Holm adjustment
controls the three primary comparisons, not the entire exploratory robustness grid.
Block-bootstrap intervals are reported in performance_summary.csv. One calendar year
and a selected basket cannot establish broad market robustness.

## Risk diagnostics and attribution

Mean predicted annual volatility: **{risk["average_predicted_volatility"]:.2%}**;
realized gross volatility: **{risk["realized_gross_volatility"]:.2%}**;
mean squared return / forecast variance: **{risk["mean_squared_return_to_variance"]:.3f}**.
Forward frozen-weight calibration is descriptive and uses overlapping 21-session windows.
Covariance diagnostics include warmup forecasts as well as evaluation forecasts.

Main solver status counts: `{statuses}`. `optimal_inaccurate` solutions are retained
only after independent constraint checks, with warnings in the log. Feasibility checks
do not certify optimality or correct sensitivity to estimation error.

{markdown_table(frames["covariance_method_comparison"])}

Factor and specific components reconcile to gross realized return. `alpha_forecast`
is an ex-ante accounting allocation, **not identified true alpha**. There is no synthetic
ground truth in this real-data report. Cumulative prior-NAV-scaled attribution reconciles
to net NAV change, including the cost component.

{markdown_table(frames["attribution_summary"])}

## Robustness and integrity checks

All {len(frames["robustness_summary"])} predeclared experiments/stresses were attempted;
{int(failures)} failures retained. Every case has its own weight/return ledgers and config.
No best case was promoted to the main result.

{markdown_table(checks)}

Raw payloads, basket, halt declarations, processed input, source files and lockfile have
SHA256 provenance. This verifies computational temporal integrity; it does not supply
missing vendor publication timestamps or fix retrospective universe selection.

## Artifacts and reproduction

Run from the project root in Windows CMD:

```cmd
uv sync --extra real-data
uv run pytest
uv run --extra real-data python scripts\\fetch_ashare_data.py
uv run --extra real-data python scripts\\validate_ashare.py
```

All empirical figures, tables, logs and backtests are under `outputs/ashare`.
Raw downloads are cached; `--refresh` deliberately replaces them and invalidates old results.
Read [AKShare's documented data interface](https://akshare.akfamily.xyz/data/stock/stock.html#id29)
and the raw payload metadata for source URLs, effective dates and retrieval times.
"""
    (output / "research_report.md").write_text(content, encoding="utf-8")
