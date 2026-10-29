"""Shared V2 daily research contract. Timestamps are naive Asia/Shanghai local time.

Daily state snapshots are supplied, never inferred from a missing quote or a
current security list. Unknown states remain nullable. Formal acceptance checks
provenance as well as values. Calendar expansion creates no prices or states.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "ashare-research-v2"
RETURN_CONVENTION = "close_decision_next_open_to_following_open_simple"
PRICE_FIELDS = ["raw_open", "raw_high", "raw_low", "raw_close"]
BOOLEAN_FIELDS = ["universe_member", "is_st", "suspended", "buyable", "sellable"]
ASOF_FIELDS = [
    "quote_asof",
    "universe_asof",
    "status_asof",
    "industry_asof",
    "capitalization_asof",
    "adjustment_asof",
    "listing_asof",
]
OPTIONAL_FIELDS = [
    "adj_factor",
    "list_date",
    "delist_date",
    "industry",
    "total_market_cap",
    "float_market_cap",
    "limit_up",
    "limit_down",
    "corporate_actions_version",
    *BOOLEAN_FIELDS,
    *ASOF_FIELDS,
]
FORMAL_FIELDS = [
    *PRICE_FIELDS,
    "volume",
    "amount",
    *OPTIONAL_FIELDS,
]


def local_session_time(dates, hour: int, minute: int = 0) -> pd.DatetimeIndex:
    """Explicit NumPy time units, including NaT; identical local wall times."""
    values = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    return pd.to_datetime(values + np.timedelta64(hour * 60 + minute, "m"))


def digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def calendar_frame(calendar: pd.DataFrame) -> pd.DataFrame:
    """Require separate exchange calendar keys; quotes cannot silently define it."""
    if {"date", "exchange"} - set(calendar):
        raise ValueError("Calendar needs date and exchange")
    result = calendar[["date", "exchange"]].copy()
    result["date"] = pd.to_datetime(result.date, errors="raise")
    if result.empty or result.isna().any().any():
        raise ValueError("Empty or missing calendar keys")
    if result.date.dt.tz is not None or not result.date.eq(result.date.dt.normalize()).all():
        raise ValueError("Calendar dates must be normalized local dates")
    if result.duplicated(["date", "exchange"]).any():
        raise ValueError("Duplicate exchange calendar sessions")
    return result.sort_values(["exchange", "date"]).reset_index(drop=True)


def normalize_dynamic_panel(frame: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Expand each observed security onto its exchange calendar, preserving gaps.

    No complete stock lifetime is required. IPO/post-delist/missing-quote rows
    have unknown states unless a daily PIT snapshot supplied them explicitly.
    Quotes outside the separate calendar, conflicting exchanges and duplicates
    are errors. Generated rows never claim suspension or non-membership.
    """
    required = {"date", "asset", "exchange", *PRICE_FIELDS, "volume", "amount"}
    if required - set(frame):
        raise ValueError(f"Missing raw fields: {sorted(required - set(frame))}")
    source = frame.copy()
    source["date"] = pd.to_datetime(source.date, errors="raise")
    if source.empty or source[["date", "asset", "exchange"]].isna().any().any():
        raise ValueError("Empty or missing panel keys")
    if source.date.dt.tz is not None or not source.date.eq(source.date.dt.normalize()).all():
        raise ValueError("Panel dates must be normalized local dates")
    if not source.asset.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("Security identifiers must be nonempty strings")
    if source.duplicated(["date", "asset"]).any():
        raise ValueError("Duplicate date-asset keys")
    if source.groupby("asset").exchange.nunique().gt(1).any():
        raise ValueError("A security cannot change exchange under the same identifier")
    sessions = calendar_frame(calendar)
    if (
        not pd.MultiIndex.from_frame(source[["date", "exchange"]])
        .isin(pd.MultiIndex.from_frame(sessions[["date", "exchange"]]))
        .all()
    ):
        raise ValueError("Quote outside supplied exchange calendar")
    securities = source[["asset", "exchange"]].drop_duplicates()
    grid = securities.merge(sessions, on="exchange", how="inner")
    panel = grid.merge(
        source,
        on=["date", "asset", "exchange"],
        how="left",
        validate="one_to_one",
        indicator="_supplied",
    )
    panel["supplied_row"] = panel.pop("_supplied").eq("both")
    for field in OPTIONAL_FIELDS:
        if field not in panel:
            panel[field] = pd.NA
    for field in BOOLEAN_FIELDS:
        # Nullable booleans, no bool('False') or null->false parsing.
        values = panel[field].dropna()
        if not values.map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f"{field} must contain booleans or null")
        panel[field] = panel[field].astype("boolean")
    for field in [*ASOF_FIELDS, "list_date", "delist_date"]:
        panel[field] = pd.to_datetime(panel[field], errors="raise")
    for field in [
        *PRICE_FIELDS,
        "volume",
        "amount",
        "adj_factor",
        "limit_up",
        "limit_down",
        "total_market_cap",
        "float_market_cap",
    ]:
        panel[field] = pd.to_numeric(panel[field], errors="raise").astype(float)
        if np.isinf(panel[field]).any():
            raise ValueError(f"Infinite field: {field}")
    prices = panel[PRICE_FIELDS]
    complete = prices.notna().all(axis=1)
    if ((prices.notna().any(axis=1)) & ~complete).any():
        raise ValueError("Partial OHLC quote")
    if (prices.loc[complete] <= 0).any().any():
        raise ValueError("Nonpositive raw quote")
    if (panel.loc[complete, "raw_low"] > prices.loc[complete].min(axis=1)).any() or (
        panel.loc[complete, "raw_high"] < prices.loc[complete].max(axis=1)
    ).any():
        raise ValueError("OHLC invariant violation")
    if (panel[["volume", "amount"]].dropna() < 0).any().any():
        raise ValueError("Negative volume/amount")
    if panel.loc[complete, ["volume", "amount"]].isna().any().any():
        raise ValueError("Observed quote requires volume and amount")
    if panel.adj_factor.dropna().le(0).any():
        raise ValueError("Nonpositive adjustment factor")
    for field in ("total_market_cap", "float_market_cap"):
        if panel[field].dropna().le(0).any():
            raise ValueError(f"Nonpositive {field}")
    panel["decision_time"] = local_session_time(panel.date, 18)
    # A row is a daily as-of snapshot; later revisions must be materialized by
    # the upstream bitemporal adapter, never used here as the earlier snapshot.
    for field in ASOF_FIELDS:
        if panel[field].gt(panel.decision_time).any():
            raise ValueError(f"Future information in {field}")
    panel["quote_observed"] = complete & panel.quote_asof.notna()
    panel["research_close"] = panel.raw_close * panel.adj_factor
    panel["research_open"] = panel.raw_open * panel.adj_factor
    return panel.sort_values(["asset", "date"]).reset_index(drop=True)


def add_decision_eligibility(
    panel: pd.DataFrame, min_history: int = 60, allow_unknown_status: bool = False
) -> pd.DataFrame:
    """Use only current known inputs; no target, next quote or next state lookup."""
    if min_history < 1:
        raise ValueError("min_history must be positive")
    result = panel.copy()
    observed = result.quote_observed & result.research_close.notna()
    observed &= result.list_date.isna() | result.list_date.le(result.date)
    result["history_count"] = observed.astype(int).groupby(result.asset).cumsum()
    member = result.universe_member.eq(True) & result.universe_asof.notna()
    known_state = result.status_asof.notna() & result.is_st.notna() & result.suspended.notna()
    state_ok = result.is_st.eq(False) & result.suspended.eq(False)
    if allow_unknown_status:
        # This exemption is explicit and allowed ONLY by an engineering runner.
        state_ok = state_ok.fillna(True)
    else:
        state_ok &= known_state
    listed = result.list_date.le(result.date) | result.list_date.isna()
    not_delisted = result.delist_date.isna() | result.delist_date.gt(result.date)
    result["eligible"] = (
        (
            observed
            & member
            & state_ok
            & listed
            & not_delisted
            & result.history_count.ge(min_history)
        )
        .fillna(False)
        .astype(bool)
    )
    return result


def add_calendar_labels(panel: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Simple adjusted next-open return, exact exchange sessions, never a fill.

    Adjusted return is a corporate-action research proxy. Raw entry/exit prices
    are retained; a live cash/dividend engine is not implied. Outcome validity
    requires both quotes and directional trading states, separate from eligibility.
    """
    result = panel.copy()
    sessions = calendar_frame(calendar)
    by_exchange = sessions.groupby("exchange", sort=False).date
    sessions["label_start"] = by_exchange.shift(-1)
    sessions["label_end"] = by_exchange.shift(-2)
    result = result.merge(sessions, on=["date", "exchange"], validate="many_to_one")
    quote_fields = ["raw_open", "research_open", "buyable", "sellable"]
    for prefix, key in (("entry", "label_start"), ("exit", "label_end")):
        quotes = panel[["asset", "date", *quote_fields]].rename(
            columns={"date": key, **{f: f"{prefix}_{f}" for f in quote_fields}}
        )
        result = result.merge(quotes, on=["asset", key], how="left", validate="many_to_one")
    result["target"] = result.exit_research_open / result.entry_research_open - 1
    valid = (
        result.entry_raw_open.gt(0)
        & result.exit_raw_open.gt(0)
        & result.entry_buyable.eq(True)
        & result.exit_sellable.eq(True)
        & np.isfinite(result.target)
    )
    result["target_valid"] = valid.fillna(False).astype(bool)
    result.loc[~result.target_valid, "target"] = np.nan
    # EOD-only feed conservatively makes the exit label usable after exit close.
    result["label_available_at"] = local_session_time(result.label_end, 18)
    result["entry_time"] = local_session_time(result.label_start, 9, 30)
    result["exit_time"] = local_session_time(result.label_end, 9, 30)
    result["return_convention"] = RETURN_CONVENTION
    result["return_unit"] = "decimal_simple"
    return result.sort_values(["date", "asset"]).reset_index(drop=True)


def quality_report(panel: pd.DataFrame, metadata: dict) -> dict:
    """Field lineage includes absent values; null delist_date means still listed
    only when an authenticated listing-status source explicitly provides it.
    """
    lineage = metadata.get("fields", {})
    fields = []
    for field in FORMAL_FIELDS:
        item = lineage.get(field, {})
        fields.append(
            {
                "field": field,
                "status": item.get("status", "missing"),
                "source": item.get("source"),
                "version": item.get("version"),
                "missing_rows": int(panel[field].isna().sum()) if field in panel else len(panel),
            }
        )
    blockers = [
        f"unverified_field:{r['field']}"
        for r in fields
        if r["status"] != "verified_pit" or not r["source"] or not r["version"]
    ]
    for flag in ("official_calendar", "survivorship_free", "publication_history_verified"):
        if metadata.get(flag) is not True:
            blockers.append(flag)
    if metadata.get("schema_version") != VERSION:
        blockers.append("schema_version")
    units = metadata.get("units", {})
    if units != {"price": "CNY", "volume": "shares", "amount": "CNY"}:
        blockers.append("explicit_units")
    members = panel.universe_member.eq(True).fillna(False)
    required_on_members = [
        "list_date",
        "industry",
        "total_market_cap",
        "float_market_cap",
        "is_st",
        "suspended",
        "buyable",
        "sellable",
        "limit_up",
        "limit_down",
        "adj_factor",
        "corporate_actions_version",
        *ASOF_FIELDS,
    ]
    for field in required_on_members:
        if panel.loc[members, field].isna().any():
            blockers.append(f"missing_member_field:{field}")
    missing_quotes = members & ~panel.quote_observed
    declared_halt = panel.suspended.eq(True).fillna(False)
    if (missing_quotes & ~declared_halt).any():
        blockers.append("unexplained_member_quote_gap")
    return {
        "schema_version": VERSION,
        "rows": len(panel),
        "assets": int(panel.asset.nunique()),
        "dates": int(panel.date.nunique()),
        "observed_quotes": int(panel.quote_observed.sum()),
        "generated_grid_rows": int((~panel.supplied_row).sum()),
        "fields": fields,
        "formal_ready": not blockers,
        "blockers": blockers,
        "pit_certification": "upstream declarations plus internal checks; not vendor attestation",
    }


def require_formal_quality(report: dict) -> None:
    if not report["formal_ready"]:
        raise ValueError(f"Formal PIT research blocked: {report['blockers']}")


def require_frozen(cfg: dict, actual_protocol_sha256: str, include_holdout: bool = False) -> None:
    """Check BEFORE reading research data. No automatic untouched attestation."""
    protocol = cfg.get("protocol", {})
    if protocol.get("status") != "frozen" or protocol.get("sha256") != actual_protocol_sha256:
        raise ValueError("Research protocol is not frozen or its hash changed")
    if cfg.get("phase") == "holdout" and (
        not include_holdout or protocol.get("holdout_confirmed_untouched") is not True
    ):
        raise ValueError("Final holdout remains sealed")
