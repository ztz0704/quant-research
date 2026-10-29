"""Explicit known-halt valuation; unclassified holes always stop acquisition."""

import pandas as pd


def insert_suspension_marks(
    prices: pd.DataFrame, calendar: pd.DatetimeIndex, asset: str, declarations: pd.DataFrame
) -> pd.DataFrame:
    """Carry only valuation prices on dated, announced halts; never carry volume.

    This is stale-price marking, not an observation of the security's fair value.
    The original quote payload and native coverage audit remain available.
    """
    known = declarations[declarations["asset"].eq(asset)].copy()
    if (known["announcement_date"] > known["date"]).any():
        raise ValueError("Suspension information was announced after the marked date")
    dates = pd.DatetimeIndex(prices["date"])
    missing = calendar.difference(dates)
    permitted = pd.DatetimeIndex(known["date"])
    if not missing.difference(permitted).empty:
        raise ValueError(
            f"Unclassified quote gap: {asset} {missing.difference(permitted).tolist()}"
        )
    if not permitted.intersection(calendar).difference(missing).empty:
        raise ValueError(f"Halt declaration conflicts with native quotes: {asset}")
    if dates.duplicated().any() or not dates.difference(calendar).empty:
        raise ValueError("Duplicate or unexpected native quote dates")
    if prices["volume"].le(0).any() or prices["close"].le(0).any():
        raise ValueError("Invalid native price/volume record")
    result = prices.set_index("date").reindex(calendar)
    columns = [name for name in ("open", "high", "low", "close") if name in result]
    marked = result[columns].ffill().loc[missing]
    if marked.isna().any().any():
        raise ValueError("No earlier valid price for halt valuation")
    result.loc[missing, columns] = marked
    result.loc[missing, ["volume", "amount"]] = 0.0
    result["tradable"] = ~result.index.isin(missing)
    result["native_quote"] = result["tradable"]
    return result.rename_axis("date").reset_index()
