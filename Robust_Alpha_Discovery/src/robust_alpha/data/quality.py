"""Compact, machine-readable panel quality diagnostics."""

from __future__ import annotations

import pandas as pd


def panel_quality_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize coverage and tradeability without inspecting research outcomes."""

    dates = pd.to_datetime(panel["date"])
    member = panel["universe_member"].astype(bool)
    tradable = panel["tradable"].astype(bool)
    cross_section = panel.loc[member].groupby("date", observed=True)["asset"].nunique()
    metrics: list[tuple[str, object]] = [
        ("rows", len(panel)),
        ("assets", panel["asset"].nunique()),
        ("dates", dates.nunique()),
        ("start_date", dates.min().date().isoformat()),
        ("end_date", dates.max().date().isoformat()),
        ("duplicate_asset_dates", panel.duplicated(["date", "asset"]).sum()),
        ("member_rows", member.sum()),
        ("tradable_member_rows", (member & tradable).sum()),
        ("nontradable_member_rows", (member & ~tradable).sum()),
        ("nonmember_rows", (~member).sum()),
        (
            "member_price_missing_cells",
            panel.loc[member, ["open", "high", "low", "close"]].isna().sum().sum(),
        ),
        ("median_members_per_date", cross_section.median()),
        ("minimum_members_per_date", cross_section.min()),
        ("maximum_members_per_date", cross_section.max()),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])
