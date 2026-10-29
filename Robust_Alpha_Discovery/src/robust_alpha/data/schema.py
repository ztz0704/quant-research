"""Canonical panel schema and defensive validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COLUMNS = ("date", "asset")
PRICE_COLUMNS = ("open", "high", "low", "close")
REQUIRED_COLUMNS = (
    "date",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "universe_member",
    "tradable",
)


class PanelValidationError(ValueError):
    """Raised when raw market data violate the canonical panel contract."""


def _strict_boolean(series: pd.Series, column: str) -> pd.Series:
    """Parse booleans without treating non-empty strings such as 'False' as true."""

    string_mapping = {"true": True, "false": False, "1": True, "0": False}

    def parse(value: object) -> bool:
        if pd.isna(value):
            raise PanelValidationError(
                f"{column} cannot contain missing values; encode the state explicitly."
            )
        if isinstance(value, bool | np.bool_):
            return bool(value)
        if isinstance(value, int | np.integer) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in string_mapping:
            return string_mapping[value.strip().lower()]
        raise PanelValidationError(
            f"{column} must contain only booleans or 0/1; received {value!r}."
        )

    return series.map(parse).astype(bool)


def normalize_and_validate_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, typed copy after checking the point-in-time panel contract."""

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PanelValidationError(f"Missing required columns: {missing}")

    panel = frame.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="raise").dt.normalize()
    panel["asset"] = panel["asset"].astype("string")
    for column in (*PRICE_COLUMNS, "volume", "amount"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    for column in ("universe_member", "tradable"):
        panel[column] = _strict_boolean(panel[column], column)

    if panel[list(KEY_COLUMNS)].isna().any().any():
        raise PanelValidationError("date and asset cannot contain missing values.")
    duplicate = panel.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate.any():
        examples = panel.loc[duplicate, list(KEY_COLUMNS)].head(5).to_dict("records")
        raise PanelValidationError(f"Duplicate (date, asset) keys, examples: {examples}")

    global_dates = pd.Index(sorted(panel["date"].unique()))
    date_position = pd.Series(range(len(global_dates)), index=global_dates)
    positions = panel["date"].map(date_position)
    spans = positions.groupby(panel["asset"], observed=True).agg(["min", "max", "count"])
    incomplete = spans["count"].ne(spans["max"] - spans["min"] + 1)
    if incomplete.any():
        examples = spans.index[incomplete].astype(str).tolist()[:5]
        raise PanelValidationError(
            "Each asset must have a row for every market date inside its panel lifespan; "
            f"incomplete examples: {examples}"
        )

    numeric = panel[[*PRICE_COLUMNS, "volume", "amount"]]
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise PanelValidationError("Price and volume fields cannot contain infinity.")

    listed = panel["universe_member"]
    if (panel["tradable"] & ~listed).any():
        raise PanelValidationError("tradable=true requires universe_member=true.")
    if panel.loc[listed, list(PRICE_COLUMNS)].isna().any().any():
        raise PanelValidationError("Universe-member rows require complete OHLC prices.")
    if (panel.loc[listed, list(PRICE_COLUMNS)] <= 0).any().any():
        raise PanelValidationError("Universe-member prices must be strictly positive.")
    if (panel.loc[listed, ["volume", "amount"]].fillna(-1) < 0).any().any():
        raise PanelValidationError("Universe-member volume and amount must be non-negative.")

    valid_ohlc = panel.loc[listed, list(PRICE_COLUMNS)]
    if (valid_ohlc["low"] > valid_ohlc[["open", "close", "high"]].min(axis=1)).any():
        raise PanelValidationError("OHLC invariant failed: low exceeds open, close, or high.")
    if (valid_ohlc["high"] < valid_ohlc[["open", "close", "low"]].max(axis=1)).any():
        raise PanelValidationError("OHLC invariant failed: high is below open, close, or low.")

    panel = panel.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    return panel
