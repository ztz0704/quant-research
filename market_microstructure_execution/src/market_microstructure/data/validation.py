"""Strict validation with explicit, bounded removal audit reports."""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    """Reason counts can overlap; removed counts unique rows."""

    input_rows: int
    output_rows: int
    removed: int
    removed_percentage: float
    reasons: dict[str, int]

    def to_dict(self) -> dict:
        """Return a JSON-serializable audit record."""
        return asdict(self)


class ValidationError(ValueError):
    """Invalid schema/data, retaining a machine-readable row audit."""

    def __init__(self, message: str, report: ValidationReport | None = None):
        super().__init__(message)
        self.report = report


def infer_levels(frame: pd.DataFrame) -> int:
    """Require complete, contiguous book levels; never ignore partial levels."""
    levels = 0
    for level in range(1, 6):
        columns = [f"{side}_{field}_{level}" for side in ("bid", "ask")
                   for field in ("price", "size")]
        count = sum(column in frame for column in columns)
        if count and (count != 4 or level != levels + 1):
            raise ValidationError(f"Incomplete/noncontiguous Level-{level} schema")
        if count:
            levels = level
    if not levels:
        raise ValidationError("Level-1 book columns required")
    return levels


def validate_orderbook(
    frame: pd.DataFrame,
    mode: Literal["raise", "drop"] = "raise",
    max_removed_fraction: float = 0.01,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Check UTC time, uniqueness, uncrossed ordered quotes, sizes and optional trades.

    Naive timestamps explicitly mean UTC. Out-of-order data always raises: sorting
    ingestion errors is not treated as deletion. Missing trade prices are permitted
    only on rows without trades. Empty inputs are rejected.
    """
    if mode not in {"raise", "drop"} or not 0 <= max_removed_fraction <= 1:
        raise ValueError("Invalid cleaning policy")
    if frame.empty or not {"timestamp", "asset"}.issubset(frame):
        raise ValidationError("Nonempty timestamp/asset data required")
    levels = infer_levels(frame)
    data = frame.copy().reset_index(drop=True)
    reasons: dict[str, pd.Series] = {}
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce", format="mixed")
    reasons["invalid_timestamp"] = data["timestamp"].isna()
    reasons["invalid_asset"] = data["asset"].isna() | data["asset"].astype(str).str.strip().eq("")
    reasons["duplicate_timestamp_asset"] = data.duplicated(["timestamp", "asset"], keep="first")
    for level in range(1, levels + 1):
        for side in ("bid", "ask"):
            for field in ("price", "size"):
                column = f"{side}_{field}_{level}"
                data[column] = pd.to_numeric(data[column], errors="coerce")
                invalid = ~np.isfinite(data[column])
                invalid |= data[column] <= 0 if field == "price" else data[column] < 0
                reasons[f"invalid_{column}"] = invalid
        reasons[f"crossed_level_{level}"] = data[f"bid_price_{level}"] >= data[f"ask_price_{level}"]
        if level > 1:
            reasons[f"unordered_level_{level}"] = (
                (data[f"bid_price_{level}"] >= data[f"bid_price_{level - 1}"])
                | (data[f"ask_price_{level}"] <= data[f"ask_price_{level - 1}"])
            )
    if "trade_size" in data:
        data["trade_size"] = pd.to_numeric(data["trade_size"], errors="coerce")
        reasons["invalid_trade_size"] = ~np.isfinite(data.trade_size) | (data.trade_size < 0)
    active = data.get("trade_size", pd.Series(0.0, index=data.index)) > 0
    if "trade_price" in data:
        data["trade_price"] = pd.to_numeric(data["trade_price"], errors="coerce")
        reasons["invalid_trade_price"] = (
            (active & (~np.isfinite(data.trade_price) | (data.trade_price <= 0)))
            | (data.trade_price.notna() & ~np.isfinite(data.trade_price))
        )
    if "trade_side" in data:
        reasons["invalid_trade_side"] = ~data.trade_side.isin([-1, 0, 1])
    masks = pd.DataFrame(reasons)
    invalid = masks.any(axis=1)
    removed = int(invalid.sum())
    counts = {name: int(mask.sum()) for name, mask in reasons.items() if mask.any()}
    report = ValidationReport(len(data), len(data) - removed, removed,
                              100 * removed / len(data), counts)
    if not data.loc[~reasons["invalid_timestamp"], "timestamp"].is_monotonic_increasing:
        raise ValidationError("Timestamps must be globally sorted (per asset strictly unique)", report)
    if removed and (mode == "raise" or removed / len(data) > max_removed_fraction):
        raise ValidationError(f"Invalid book rows: {report.to_dict()}", report)
    clean = data.loc[~invalid].reset_index(drop=True)
    if clean.empty:
        raise ValidationError("No usable rows remain", report)
    return clean, report
