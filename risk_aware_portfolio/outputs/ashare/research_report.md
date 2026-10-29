# A-share Real Data Validation

Actual Sina A-share prices, decoded with AKShare. This is an engineering and risk-model
validation on a **retrospectively declared fixed basket**, not a validated trading strategy
or a survivorship-free market study. The synthetic results remain in their original directories.

## Declared sample and data audit

60 assets; 241 realized sessions,
2023-01-04 through 2023-12-29.
2022 is the fixed feature/risk warmup; 2023 is the predeclared evaluation year.
No asset was removed after download and no parameters were selected by realized Sharpe.
The native coverage audit contains 10 absent quote rows.
Only dates in the linked suspension declarations are marked at the last valid price;
unknown holes stop preparation. Native raw quotes remain separate from valuation marks.

## Timing, exposures and execution

Same-date returns are adjusted-close ratios with no implicit fill. Signals and exposures
have one complete session of lag: features observed at $t-1$ enter $w_t$, which earns
$r_{t+1}$. Covariance moments use regressions available through decision date $t$.
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
1 deferred sessions in the main portfolio.
Daily bar data does not certify fillability at price limits; T+1 settlement, cash dividend
timing, rights subscriptions and delisting cash flows still require production event accounting.

## Measured results

| portfolio | annualized_return | annualized_volatility | sharpe | max_drawdown | average_turnover | total_transaction_cost |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | -0.0650 | 0.1646 | -0.3264 | -0.2045 | 0.3114 | 0.1112 |
| Risk-Aware Portfolio | 0.0642 | 0.1231 | 0.5667 | -0.1172 | 0.0652 | 0.0189 |
| Risk-Aware + TC | 0.0913 | 0.1238 | 0.7673 | -0.1112 | 0.0165 | 0.0042 |
| Neutralized Alpha Portfolio | 0.1165 | 0.1255 | 0.9407 | -0.1062 | 0.0172 | 0.0043 |

Geometric annualized return; arithmetic daily Sharpe with zero risk-free rate; one-way
L1 turnover including initial deployment. Total cost is the sum of fee fractions, not
additive cumulative NAV drag. The attribution ledger links contributions to actual NAV.
These outcomes do not establish deployable alpha. The constrained policy and unconstrained
rank basket differ in several ways, so their comparison is not a pure optimizer causal effect.

| portfolio | comparison | mean | hac_standard_error | hac_t_stat | p_value | holm_p_value |
| --- | --- | --- | --- | --- | --- | --- |
| Risk-Aware Portfolio | Baseline | 0.0005 | 0.0003 | 1.5014 | 0.1332 | 0.2360 |
| Risk-Aware + TC | Baseline | 0.0006 | 0.0004 | 1.5697 | 0.1165 | 0.2360 |
| Neutralized Alpha Portfolio | Baseline | 0.0007 | 0.0004 | 1.7584 | 0.0787 | 0.2360 |

HAC tests concern paired mean daily returns, not Sharpe differences. Holm adjustment
controls the three primary comparisons, not the entire exploratory robustness grid.
Block-bootstrap intervals are reported in performance_summary.csv. One calendar year
and a selected basket cannot establish broad market robustness.

## Risk diagnostics and attribution

Mean predicted annual volatility: **12.35%**;
realized gross volatility: **12.38%**;
mean squared return / forecast variance: **1.066**.
Forward frozen-weight calibration is descriptive and uses overlapping 21-session windows.
Covariance diagnostics include warmup forecasts as well as evaluation forecasts.

Main solver status counts: `{'held': 191, 'optimal': 42, 'optimal_inaccurate': 8}`. `optimal_inaccurate` solutions are retained
only after independent constraint checks, with warnings in the log. Feasibility checks
do not certify optimality or correct sensitivity to estimation error.

| method | n_forecasts | mean_effective_observations | mean_asset_condition_number | mean_factor_condition_number | minimum_asset_eigenvalue | maximum_asset_eigenvalue | minimum_factor_eigenvalue | mean_relative_covariance_change | mean_relative_factor_covariance_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample | 294.0000 | 119.1429 | 195.0377 | 115.0706 | 0.0000 | 0.0087 | 0.0000 | 0.0471 | nan |
| ewma | 294.0000 | 32.2404 | 211.9482 | 170.2184 | 0.0000 | 0.0142 | 0.0000 | 0.1123 | nan |

Factor and specific components reconcile to gross realized return. `alpha_forecast`
is an ex-ante accounting allocation, **not identified true alpha**. There is no synthetic
ground truth in this real-data report. Cumulative prior-NAV-scaled attribution reconciles
to net NAV change, including the cost component.

| portfolio | factor | alpha_forecast | specific_after_alpha | transaction_cost | total |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.0262 | 0.1190 | -0.0914 | -0.1160 | -0.0622 |
| Risk-Aware Portfolio | 0.0937 | 0.0337 | -0.0456 | -0.0205 | 0.0613 |
| Risk-Aware + TC | 0.0995 | 0.0057 | -0.0136 | -0.0046 | 0.0871 |
| Neutralized Alpha Portfolio | 0.0958 | 0.0070 | 0.0131 | -0.0048 | 0.1111 |

## Robustness and integrity checks

All 21 predeclared experiments/stresses were attempted;
0 failures retained. Every case has its own weight/return ledgers and config.
No best case was promoted to the main result.

| check | cutoff | passed | maximum_weight_difference |
| --- | --- | --- | --- |
| prefix_invariance | 2023-02-21 00:00:00 | True | 0.0000 |
| future_perturbation | 2023-02-21 00:00:00 | True | 0.0000 |

Raw payloads, basket, halt declarations, processed input, source files and lockfile have
SHA256 provenance. This verifies computational temporal integrity; it does not supply
missing vendor publication timestamps or fix retrospective universe selection.

## Artifacts and reproduction

Run from the project root in Windows CMD:

```cmd
uv sync --extra real-data
uv run pytest
uv run --extra real-data python scripts\fetch_ashare_data.py
uv run --extra real-data python scripts\validate_ashare.py
```

All empirical figures, tables, logs and backtests are under `outputs/ashare`.
Raw downloads are cached; `--refresh` deliberately replaces them and invalidates old results.
Read [AKShare's documented data interface](https://akshare.akfamily.xyz/data/stock/stock.html#id29)
and the raw payload metadata for source URLs, effective dates and retrieval times.
