"""Open-interval risk from X_s and labels matured by close(t), with dynamic assets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from risk_aware_portfolio.backtest.open_events import at_time
from risk_aware_portfolio.risk.covariance import build_asset_covariance, ensure_psd
from risk_aware_portfolio.risk.exposures import build_exposures
from risk_aware_portfolio.risk.factor_covariance import estimate_covariance
from risk_aware_portfolio.risk.factor_returns import estimate_factor_returns
from risk_aware_portfolio.risk.model import RiskSnapshot
from risk_aware_portfolio.risk.specific_risk import estimate_specific_variance

OPEN_CONVENTION = "close_decision_next_open_to_following_open_simple"


@dataclass
class OpenRiskHistory:
    exposures: dict
    factor_returns: pd.DataFrame
    specific_returns: pd.DataFrame
    available_at: pd.Series
    diagnostics: pd.DataFrame


def fit_open_history(panel: pd.DataFrame, config: dict) -> OpenRiskHistory:
    """Ex-post regression archive; ONLY as-of slices are allowed into a snapshot.

    Label s is open(s+1)->open(s+2). X_s and cap_s predate entry;
    the regression becomes available at the recorded label_available_at.
    No shifted close return series or future industry vocabulary is used.
    """
    if panel.duplicated(["date", "asset"]).any():
        raise ValueError("Duplicate open risk input")
    if (
        not panel.return_convention.eq(OPEN_CONVENTION).all()
        or not panel.return_unit.eq("decimal_simple").all()
    ):
        raise ValueError("Open risk model requires the exact simple open interval")
    if (
        panel.feature_asof.isna().any()
        or panel.decision_time.isna().any()
        or panel.feature_asof.gt(panel.decision_time).any()
    ):
        raise ValueError("Future risk exposure information")
    exposures, factors, residuals, availability, diagnostics = {}, {}, {}, {}, []
    risk = config["risk_model"]
    for date, section in panel.groupby("date", sort=True):
        current = section.loc[section.eligible].sort_values("asset").copy()
        if current.empty:
            continue
        x = build_exposures(current, config)
        exposures[date] = x
        valid = current.target_valid & current.target.notna()
        fitted = current.loc[valid].set_index("asset")
        record = {
            "date": date,
            "status": "insufficient_or_unidentified",
            "valid_assets": len(fitted),
            "exposure_asof": at_time(date, 18),
            "return_convention": OPEN_CONVENTION,
        }
        if not fitted.empty:
            # One decision-date cross section must describe one actual interval.
            if fitted.label_start.nunique() != 1 or fitted.label_end.nunique() != 1:
                record["status"] = "blocked_mixed_exchange_intervals"
                diagnostics.append(record)
                continue
            record.update(
                label_start=fitted.label_start.iloc[0],
                label_end=fitted.label_end.iloc[0],
                available_at=fitted.label_available_at.max(),
            )
            if not (
                fitted.label_available_at.notna().all()
                and fitted.label_start.gt(date).all()
                and fitted.label_available_at.ge(fitted.label_end).all()
            ):
                raise ValueError("Invalid open-label maturity / interval")
            cap = fitted.total_market_cap.where(
                fitted.total_market_cap.notna(), fitted.float_market_cap
            )
            try:
                fit = estimate_factor_returns(
                    x.loc[fitted.index],
                    fitted.target,
                    cap,
                    risk["regression"],
                    risk["weight_power"],
                )
                factors[date], residuals[date] = fit.factor_returns, fit.specific_returns
                availability[date] = fitted.label_available_at.max()
                record.update(status="ok", **fit.diagnostics)
            except ValueError as error:
                record["reason"] = str(error)
        diagnostics.append(record)
    return OpenRiskHistory(
        exposures,
        pd.DataFrame.from_dict(factors, orient="index").sort_index(),
        pd.DataFrame.from_dict(residuals, orient="index").sort_index(),
        pd.Series(availability, dtype="datetime64[ns]").sort_index(),
        pd.DataFrame(diagnostics),
    )


def open_snapshot(
    history: OpenRiskHistory, date: pd.Timestamp, assets: pd.Index, config: dict
) -> tuple[RiskSnapshot, dict]:
    """Use labels available <= close(t); never use target_valid of future decisions."""
    cutoff = at_time(date, 18)
    risk = config["risk_model"]
    if date not in history.exposures or not assets.isin(history.exposures[date].index).all():
        raise ValueError("Missing contemporaneous risk exposures")
    mature = history.available_at.index[history.available_at.le(cutoff)]
    mature = mature[-risk["lookback"] :]
    if len(mature) < risk["min_observations"]:
        raise ValueError("Insufficient matured open factor regressions")
    x = history.exposures[date].loc[assets]
    factor_cov = estimate_covariance(
        history.factor_returns.loc[mature], risk["covariance_method"], risk["ewma_lambda"]
    )
    values, factor_audit = ensure_psd(factor_cov.to_numpy(), risk["eigenvalue_floor"])
    factor_cov = pd.DataFrame(values, index=x.columns, columns=x.columns)
    past_residuals = history.specific_returns.loc[mature]
    estimates, fallback = {}, []
    past_variances = past_residuals.var(ddof=1).dropna()
    if past_variances.empty:
        raise ValueError("No historical residual risk estimates")
    conservative_prior = max(float(past_variances.quantile(0.9)), risk["variance_floor"])
    for asset in assets:
        sample = (
            past_residuals[asset].dropna() if asset in past_residuals else pd.Series(dtype=float)
        )
        if len(sample) >= risk.get("min_specific_observations", 20):
            estimates[asset] = estimate_specific_variance(
                sample.to_frame(asset),
                risk["specific_method"],
                risk["ewma_lambda"],
                risk["variance_floor"],
                0.0,
            ).iloc[0]
        elif risk.get("allow_specific_prior", False):
            estimates[asset] = conservative_prior
            fallback.append(asset)
        else:
            raise ValueError(f"Insufficient historical specific risk: {asset}")
    specific = pd.Series(estimates).reindex(assets)
    shrink = risk["specific_shrinkage"]
    specific = ((1 - shrink) * specific + shrink * specific.median()).clip(
        lower=risk["variance_floor"]
    )
    covariance, audit = build_asset_covariance(x, factor_cov, specific, risk["eigenvalue_floor"])
    diagnostic = {
        "date": date,
        "risk_cutoff": cutoff,
        "max_label_available": history.available_at.loc[mature].max(),
        "mature_regressions": len(mature),
        "specific_prior_assets": ";".join(fallback),
        "return_convention": OPEN_CONVENTION,
        "ewma_clock": "valid_matured_regression_observations",
        **audit.metrics(),
    }
    return RiskSnapshot(date, x, factor_cov, specific, covariance, audit, factor_audit), diagnostic
