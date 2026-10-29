"""Create four reproducible notebooks that inspect actual research artifacts."""

import subprocess
import sys

import nbformat as nbf

from market_microstructure.utils.paths import project_path

SETUP = '''from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
from market_microstructure.utils.config import load_config
from market_microstructure.utils.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
config = load_config()
def table(name):
    return pd.read_csv(ROOT / 'outputs' / 'tables' / f'{name}.csv')
def figure(name):
    display(Image(filename=str(ROOT / 'outputs' / 'figures' / f'{name}.png')))
features = pd.read_parquet(ROOT / config['data']['features_path'])
assert set(features.asset) == {'000400.SZ'}
assert features.timestamp.dt.tz is not None
'''


def markdown(text: str):
    """Create an explanation cell."""
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    """Create a deterministic code cell."""
    return nbf.v4.new_code_cell(text)


def write(name: str, cells: list) -> None:
    """Notebooks use the project environment and no live API dependency."""
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3.12 (project uv)", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    path = project_path(f"notebooks/{name}.ipynb")
    nbf.write(notebook, path)
    lint = subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(path)],
                          capture_output=True, text=True, check=False)
    if lint.returncode:
        raise RuntimeError(lint.stdout + lint.stderr)


def main() -> None:
    """Build sources only; verify_notebooks.py executes every cell and saves outputs."""
    write("01_orderbook_diagnostics", [
        markdown("# 01 · Real A-share order book diagnostics\n\n000400.SZ, 2022-06-01. Published messages reconstructed internally; no official reference snapshot. This notebook reads the actual primary artifacts, with no generated replacement data."),
        code(SETUP),
        code("audit = json.loads((ROOT / 'outputs/tables/reconstruction_audit.json').read_text())\ndisplay(pd.Series(audit, name='replay audit'))\nsource = json.loads((ROOT / 'outputs/tables/public_sample_manifest.json').read_text())\ndisplay(pd.DataFrame(source['files'])[['file', 'bytes', 'sha256']])"),
        markdown("## Schema and data quality\n\nValidation rejects nonfinite prices/sizes, duplicate keys and crossed books. The 41 rejected crossed reconstruction endpoints were counted upstream. Their exclusion is not evidence of official accuracy."),
        code("from market_microstructure.data.validation import validate_orderbook\nbook = pd.read_parquet(ROOT / config['data']['input_path'])\nvalidated, audit_rows = validate_orderbook(book)\nassert audit_rows.removed == 0\ndisplay(book[['timestamp','asset','bid_price_1','ask_price_1','bid_size_1','ask_size_1']].head())\ndisplay(features[['spread_bps','imbalance_1','imbalance_k','total_depth','microprice_deviation_bps']].describe())"),
        code("figure('order_imbalance_distribution')\nfigure('spread_distribution')\nfigure('depth_distribution')\nfigure('microprice_deviation_distribution')"),
        markdown("## Causal feature check\n\nModify only future real quotes and compare the unchanged prefix. Labels are future outcomes and are excluded from model features."),
        code("from market_microstructure.features.feature_pipeline import build_features, feature_columns\ncutoff = book.timestamp.iloc[1000]\nchanged = book.copy()\nfor column in book.columns:\n    if column.startswith(('bid_price_', 'ask_price_')):\n        changed.loc[book.timestamp > cutoff, column] *= 1.1\nfirst = build_features(book, levels=5, weighting='distance_decay', rolling_window=50, min_periods=10)\nsecond = build_features(changed, levels=5, weighting='distance_decay', rolling_window=50, min_periods=10)\nmask = book.timestamp <= cutoff\npd.testing.assert_frame_equal(first.loc[mask, feature_columns(first)], second.loc[mask, feature_columns(second)])\nprint('Future quote mutation leaves all prefix predictors unchanged.')"),
        markdown("Trade intensity describes active snapshot intervals, not a precise message arrival rate. Sampling is one second and resets at lunch; finer exchange latency is not identified."),
    ])
    write("02_signal_research", [
        markdown("# 02 · Out-of-sample signal research\n\nPurged chronological folds; all preprocessing fits training history. This is one stock-day, not evidence of a stable market edge."),
        code(SETUP),
        code("predictions = pd.read_parquet(ROOT / 'data/processed/predictions.parquet')\nassert not predictions.duplicated(['model','asset','timestamp']).any()\ndisplay(table('model_oos_summary'))\ndisplay(table('signal_summary'))\ndisplay(table('feature_ablation_summary'))"),
        markdown("## Train-only fit and overlap boundaries\n\nA real first-fold logistic model is fitted below using exactly the purged training indices. It is an inspectable example, not a backfill of saved OOS predictions."),
        code("from market_microstructure.models.validation import WalkForwardConfig, walk_forward_splits\nfrom market_microstructure.models.estimators import make_model\nfrom market_microstructure.features.feature_pipeline import feature_columns\nfold = next(walk_forward_splits(features, WalkForwardConfig(**config['validation'])))\ntrain = features.iloc[fold.train].dropna(subset=['direction'])\nassert train.label_end.max() < fold.validation_start\nassert features.iloc[fold.validation].label_end.max() < fold.test_start\ncolumns = feature_columns(features)\nassert not {'future_return_bps','direction','label_end'} & set(columns)\nmodel = make_model('logistic', config['model'], config['seed'])\nmodel.fit(train[columns], train.direction)\nfrozen_mean = model.named_steps['scaler'].mean_.copy()\nmodel.predict_proba(features.iloc[fold.test][columns])\nnp.testing.assert_array_equal(frozen_mean, model.named_steps['scaler'].mean_)\ndisplay(table('fold_audit').head(6))\nprint('Training scaler remains frozen on unseen data.')"),
        code("figure('signal_quantile_vs_future_return')\nfigure('roc_curve')\nfigure('prediction_vs_realized_return')"),
        markdown("## Decay and stability\n\nOverlapping labels remain dependent. The bootstrap is within-day timestamp-block uncertainty. Half-life is a sampled half-peak crossing rather than an exponential fit."),
        code("display(table('signal_half_life'))\nfigure('signal_decay_curve')\nfigure('coefficient_stability')\nfigure('feature_importance')\nfigure('rolling_ic')"),
    ])
    write("03_execution_analysis", [
        markdown("# 03 · Execution, costs and A-share inventory\n\nStarting empty, T+1 prevents same-day stock round trips. Marked PnL and realized closed-lot outcomes are distinct. The maker model is explicitly simplified and uncalibrated."),
        code(SETUP),
        code("execution = table('execution_summary')\ndisplay(execution[['model','mode','gross_pnl','net_pnl','trade_count','residual_inventory_abs','closed_lot_count']])\ndisplay(table('pnl_decomposition'))"),
        markdown("## Every-event reconciliation\n\nThe cost bridge must equal cash plus marked stock minus initial capital. Same-day sales cannot erase the residual position."),
        code("selected = config['model']['type']\nledger = pd.read_parquet(ROOT / f'outputs/backtests/{selected}_taker_events.parquet')\nbridge = ledger.gross_pnl - ledger.spread_cost - ledger.fees - ledger.slippage - ledger.impact\nnp.testing.assert_allclose(ledger.net_pnl, bridge, atol=1e-7)\nnp.testing.assert_allclose(ledger.equity, ledger.cash + ledger.position * ledger.mid, atol=1e-7)\nassert ledger.position.min() >= 0\nassert ledger.cash.min() >= 0\nclosed = pd.read_parquet(ROOT / f'outputs/backtests/{selected}_taker_closed_lots.parquet')\nassert closed.empty\nprint('All events reconcile; closed-trade hit rate is unavailable under this mandate.')"),
        code("figure('cumulative_net_pnl')\nfigure('pnl_decomposition')\nfigure('inventory_path')\nfigure('rolling_hit_rate')"),
        markdown("## Fill and markout assumptions\n\nFuture mid labels are used for diagnostics only. Queue proxy, interval aggregation and cancellation timing prevent interpreting these as calibrated exchange fills."),
        code("display(table('maker_markouts')[['timestamp','quantity','spread_capture','signed_markout_bps']])\nfigure('fill_probability')\nfigure('maker_adverse_selection')"),
        markdown("## All predeclared sensitivities\n\nFrozen OOS signals except explicit noise. Scaling the observed path scales its trend too; it is not a new empirical volatility regime."),
        code("display(table('robustness_summary')[['scenario','net_pnl','trade_count','residual_inventory_abs','max_drawdown']])\ndisplay(table('regime_summary'))\nfigure('pnl_vs_latency')\nfigure('all_stress_scenarios')"),
    ])
    write("04_final_report", [
        markdown("# 04 · Final research report and optimal execution\n\nReal-data findings with explicit market/account scope and uncalibrated assumptions. The optimal-execution mandate starts with settled inventory and is separate from the empty-account signal strategy."),
        code(SETUP),
        code("report = (ROOT / 'outputs/research_report.md').read_text(encoding='utf-8')\ndisplay(Markdown(report.split('## Figures')[0]))"),
        markdown("## Optimal execution grid\n\nSigma/spread/volume forecast use real training history. Eta/gamma are assumed sensitivity inputs. Continuous optima and conserving 100-share rounding are reported separately; rounding is not integer optimization."),
        code("optimal = table('optimal_execution_summary')\nassert len(optimal) == 160\nbase = optimal[(optimal.sigma_multiplier == 1) & (optimal.eta_multiplier == 1) & (optimal.size_multiplier == 1)]\ndisplay(base[['schedule','risk_aversion_per_cny','expected_cost_cny','cost_variance_cny2','rounded_expected_cost_cny']])\nfigure('almgren_chriss_execution_trajectory')\nfigure('almgren_chriss_inventory_trajectory')\nfigure('execution_cost_vs_risk_aversion')\nfigure('execution_cost_risk_frontier')"),
        code("manifest = json.loads((ROOT / 'outputs/tables/reproducibility_manifest.json').read_text())\nassert manifest['synthetic_primary_results'] is False\nassert manifest['oos_rows_all_models'] == 31500\nassert all((ROOT / 'outputs' / file).is_file() for file in manifest['figures'])\ndisplay(pd.Series(manifest['package_versions'], name='locked package versions'))\nprint(f\"Verified {manifest['figure_count']} figure files and real-data provenance.\")"),
        markdown("## Interview conclusion\n\nA statistical signal must be evaluated through an executable account mandate. Here T+1, passive directional exposure, minimum commission and uncertain maker fills prevent treating marked gains as realized high-frequency profitability. Next work: official snapshot reconciliation, licensed multi-day holdouts, own-order fill calibration and settled/multi-day inventories."),
    ])
    print("Created four real-data research notebooks.")


if __name__ == "__main__":
    main()
