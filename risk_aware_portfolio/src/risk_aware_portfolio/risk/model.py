"""Sequential risk model. All snapshots are assembled using observations <= as-of."""

from dataclasses import dataclass, field

import pandas as pd

from risk_aware_portfolio.data.universe import require_balanced_universe
from risk_aware_portfolio.data.validation import validate_panel
from risk_aware_portfolio.risk.covariance import (
    CovarianceAudit,
    build_asset_covariance,
    ensure_psd,
)
from risk_aware_portfolio.risk.exposures import ExposureBuilder, build_exposures
from risk_aware_portfolio.risk.factor_covariance import estimate_covariance
from risk_aware_portfolio.risk.factor_returns import estimate_factor_returns
from risk_aware_portfolio.risk.specific_risk import estimate_specific_variance
from risk_aware_portfolio.utils.config import Config


@dataclass(frozen=True)
class RiskSnapshot:
    date: pd.Timestamp
    exposures: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variance: pd.Series
    covariance: pd.DataFrame
    audit: CovarianceAudit
    factor_audit: CovarianceAudit


@dataclass
class RiskHistory:
    factor_returns: pd.DataFrame
    specific_returns: pd.DataFrame
    regressions: pd.DataFrame
    snapshots: dict[pd.Timestamp, RiskSnapshot] = field(default_factory=dict)


def _snapshot(
    date: pd.Timestamp,
    x: pd.DataFrame,
    factors: pd.DataFrame,
    residuals: pd.DataFrame,
    config: Config,
) -> RiskSnapshot:
    risk = config["risk_model"]
    factors, residuals = factors.tail(risk["lookback"]), residuals.tail(risk["lookback"])
    factor_cov = estimate_covariance(factors, risk["covariance_method"], risk["ewma_lambda"])
    values, factor_audit = ensure_psd(factor_cov.to_numpy(), risk["eigenvalue_floor"])
    factor_cov = pd.DataFrame(values, index=factor_cov.index, columns=factor_cov.columns)
    specific = estimate_specific_variance(
        residuals,
        risk["specific_method"],
        risk["ewma_lambda"],
        risk["variance_floor"],
        risk["specific_shrinkage"],
    )
    covariance, audit = build_asset_covariance(x, factor_cov, specific, risk["eigenvalue_floor"])
    return RiskSnapshot(date, x, factor_cov, specific, covariance, audit, factor_audit)


def estimate_risk_history(
    panel: pd.DataFrame, config: Config, builder: ExposureBuilder = build_exposures
) -> RiskHistory:
    """Fit return_t to X_(t-1), then forecast risk for t+1 with X_t.

    Warmup requires min_observations actual regressions. Iteration never consults
    future categories, future membership, future caps, or full-sample moments.
    """
    validate_panel(panel, config["data"]["max_abs_return"])
    require_balanced_universe(panel)
    risk = config["risk_model"]
    factor_rows, residual_rows, diagnostics, dates = [], [], [], []
    snapshots: dict[pd.Timestamp, RiskSnapshot] = {}
    previous_x, previous_cap = None, None
    for date, cross_section in panel.groupby("date", sort=True):
        current_x = builder(cross_section, config)
        indexed = cross_section.set_index("asset")
        if previous_x is not None:
            available = (
                indexed["tradable"]
                if "tradable" in indexed
                else pd.Series(True, index=indexed.index)
            )
            fit = estimate_factor_returns(
                previous_x.loc[available],
                indexed.loc[available, "return"],
                previous_cap.loc[available],
                risk["regression"],
                risk["weight_power"],
            )
            factor_rows.append(fit.factor_returns)
            # Retain every asset in accounting, even when excluded from the daily fit.
            residual_rows.append(indexed["return"] - previous_x @ fit.factor_returns)
            diagnostics.append(fit.diagnostics)
            dates.append(date)
            if len(dates) >= risk["min_observations"]:
                factors = pd.DataFrame(factor_rows[-risk["lookback"] :])
                residuals = pd.DataFrame(residual_rows[-risk["lookback"] :])
                snapshots[date] = _snapshot(date, current_x, factors, residuals, config)
        previous_x, previous_cap = current_x, indexed["market_cap"]
    return RiskHistory(
        pd.DataFrame(factor_rows, index=dates),
        pd.DataFrame(residual_rows, index=dates),
        pd.DataFrame(diagnostics, index=dates),
        snapshots,
    )
