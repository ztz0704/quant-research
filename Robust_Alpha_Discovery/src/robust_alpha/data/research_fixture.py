"""Separate deterministic dynamic fixture, never an empirical A-share dataset."""

import numpy as np
import pandas as pd

from robust_alpha.data.research_contract import (
    ASOF_FIELDS,
    FORMAL_FIELDS,
    VERSION,
    local_session_time,
)
from robust_alpha.data.synthetic import generate_synthetic_panel


def engineering_sample(
    days: int = 260, assets: int = 40, seed: int = 8, continuous: bool = False
) -> tuple:
    old = generate_synthetic_panel(n_days=days, n_assets=assets, seed=seed)
    if continuous:
        # A separate, explicitly declared no-halt/no-action accounting oracle.
        # Raw generated prices already exist; no empirical quote is filled.
        old["universe_member"] = True
        old["tradable"] = True
    panel = old.rename(columns={p: f"raw_{p}" for p in ("open", "high", "low", "close")})
    panel["asset"] = "SYN_" + panel.asset
    panel["exchange"] = "SSE"
    panel["adj_factor"] = 1.0
    panel["is_st"] = False
    panel["suspended"] = ~panel.tradable & panel.universe_member
    panel["buyable"] = panel.tradable.astype(bool)
    panel["sellable"] = panel.tradable.astype(bool)
    panel["industry"] = panel.asset.str[-2:].astype(int).mod(4).map(lambda n: f"SYN_G{n}")
    panel["total_market_cap"] = panel.raw_close * 1e8
    panel["float_market_cap"] = panel.total_market_cap * 0.7
    panel["limit_up"], panel["limit_down"] = panel.raw_close * 1.1, panel.raw_close * 0.9
    panel["list_date"] = panel.groupby("asset").date.transform("min")
    panel["delist_date"] = pd.NaT
    panel["corporate_actions_version"] = "synthetic_no_actions"
    for field in ASOF_FIELDS:
        panel[field] = local_session_time(panel.date, 18)
    absent = ~panel.universe_member | panel.suspended
    panel.loc[
        absent, [f"raw_{p}" for p in ("open", "high", "low", "close")] + ["volume", "amount"]
    ] = np.nan
    calendar = panel[["date", "exchange"]].drop_duplicates()
    metadata = {
        "schema_version": VERSION,
        "provider": "synthetic_engineering_only",
        "official_calendar": False,
        "survivorship_free": False,
        "publication_history_verified": False,
        "units": {"price": "CNY", "volume": "shares", "amount": "CNY"},
        "fields": {
            field: {
                "status": "synthetic",
                "source": "controlled_fixture",
                "version": f"seed_{seed}",
            }
            for field in FORMAL_FIELDS
        },
        "limitations": ["SYN assets and business-day calendar, no empirical evidence"],
        "continuous_accounting_oracle": continuous,
    }
    return panel, calendar, metadata
