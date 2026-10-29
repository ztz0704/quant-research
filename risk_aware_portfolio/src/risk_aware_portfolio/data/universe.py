"""Conservative initial universe policy: explicit rejection of unsupported events."""

import pandas as pd


def require_balanced_universe(panel: pd.DataFrame) -> tuple[str, ...]:
    """Require a balanced universe; never silently drop held delisted assets.

    Production delistings, suspensions and stale quotes require an event ledger.
    This prototype rejects such panels instead of inducing survivorship/accounting bias.
    """
    reference = tuple(panel.loc[panel["date"].eq(panel["date"].iloc[0]), "asset"])
    for date, group in panel.groupby("date", sort=True):
        if tuple(group["asset"]) != reference:
            raise ValueError(f"Unbalanced universe at {date}; supply an event policy")
    return reference
