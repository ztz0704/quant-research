"""Thin empirical notebook; computation remains in the tested core modules."""

from risk_aware_portfolio.reporting.notebooks import _write


def write_real_notebook(config_path: str) -> None:
    """Read the isolated, provenance-checked real-data artifacts for visual review."""
    setup = f"""import pandas as pd
from IPython.display import Image, Markdown, display
from risk_aware_portfolio.utils.paths import resolve_path
from risk_aware_portfolio.utils.config import load_config
from risk_aware_portfolio.utils.manifest import verify_manifest
from risk_aware_portfolio.reporting.real_validation import verify_real_data
config = load_config({config_path!r})
output = resolve_path(config['real_data']['output_dir'])
verify_real_data(config)
verify_manifest(output / 'manifest.json', config, resolve_path(config['data']['input_path']))
"""
    _write(
        "05_ashare_validation",
        [
            (
                "markdown",
                "# A-share empirical validation\n\nActual historical prices on a retrospectively specified fixed basket. The default reversal input has not passed independent alpha discovery. This notebook consumes the entire declared evaluation, with no selection of favorable subperiods.",
            ),
            (
                "code",
                setup
                + "display(pd.read_csv(output / 'tables/performance_summary.csv'))\ndisplay(pd.read_csv(output / 'tables/pairwise_comparisons.csv'))",
            ),
            (
                "markdown",
                "## Risk, attribution and temporal integrity\n\nForecast alpha is an accounting proxy, not identified true alpha. Effective-date data joins do not certify historical publication availability.",
            ),
            (
                "code",
                "for name in ['risk_summary', 'attribution_summary', 'covariance_method_comparison', 'no_lookahead_checks', 'risk_aware_tc_solver_status']:\n    display(pd.read_csv(output / f'tables/{name}.csv'))",
            ),
            (
                "code",
                "for name in ['cumulative_net_comparison', 'predicted_vs_realized_risk', 'factor_attribution']:\n    display(Image(filename=str(output / f'figures/{name}.png'), width=850))",
            ),
            ("code", "display(pd.read_csv(output / 'tables/robustness_summary.csv'))"),
            (
                "markdown",
                "## Limits\n\nManual business groups, retrospective basket selection, incomplete publication/revision timestamps, idealized close fills and incomplete A-share event/settlement rules remain material limits. Read outputs/ashare/research_report.md and docs/ASHARE_VALIDATION.md for the full audit and reproduction commands.",
            ),
        ],
    )
