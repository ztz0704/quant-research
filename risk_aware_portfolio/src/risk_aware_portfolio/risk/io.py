"""Inspectable as-of factor covariance, specific variance and exposure artifacts."""

from pathlib import Path

import pandas as pd

from risk_aware_portfolio.risk.model import RiskHistory


def save_risk_snapshots(history: RiskHistory, directory: Path) -> None:
    """Save labeled long-format forecasts; full asset covariance is reconstructible."""
    directory.mkdir(parents=True, exist_ok=True)
    factors, specific, exposures = [], [], []
    for date, snapshot in history.snapshots.items():
        covariance = snapshot.factor_covariance.rename_axis(index="factor_i", columns="factor_j")
        factors.append(
            covariance.stack(future_stack=True).rename("covariance").reset_index().assign(date=date)
        )
        specific.append(
            snapshot.specific_variance.rename_axis("asset").reset_index().assign(date=date)
        )
        exposures.append(snapshot.exposures.rename_axis("asset").reset_index().assign(date=date))
    for name, rows in (
        ("factor_covariance", factors),
        ("specific_variance", specific),
        ("factor_exposures", exposures),
    ):
        pd.concat(rows, ignore_index=True).to_parquet(directory / f"{name}.parquet", index=False)
