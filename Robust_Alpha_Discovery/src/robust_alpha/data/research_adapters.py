"""Offline vendor adapters; no credential, API entitlement or network assumed."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from robust_alpha.data.research_contract import (
    ASOF_FIELDS,
    FORMAL_FIELDS,
    VERSION,
    digest_file,
)


def read_bundle(directory: str | Path, date_end: str | None = None) -> tuple:
    """Canonical exports from Tushare/Wind/CSMAR must provide their own PIT states.

    Every bundle file is hash bound. Parquet is filtered before loading data past
    a research boundary. Metadata must supply hashes for calendar and panel.
    """
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    panel_name = metadata.get("panel_file", "panel.parquet")
    if Path(panel_name).name != panel_name:
        raise ValueError("Bundle panel filename must stay inside its directory")
    for name in (panel_name, "calendar.parquet"):
        if metadata.get("files", {}).get(name) != digest_file(root / name):
            raise ValueError(f"Bundle identity mismatch: {name}")
    filters = [("date", "<=", pd.Timestamp(date_end))] if date_end else None
    panel = pd.read_parquet(root / panel_name, filters=filters)
    calendar = pd.read_parquet(root / "calendar.parquet", filters=filters)
    return panel, calendar, metadata


def normalize_tushare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Convert documented unadjusted daily export units: hands→shares, kCNY→CNY.

    No stock_basic.industry is promoted to historical industry; adjustment and
    actual information timestamps must be joined from separately versioned data.
    """
    required = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}
    if required - set(daily):
        raise ValueError(f"Tushare daily missing {sorted(required - set(daily))}")
    result = daily.rename(
        columns={
            "ts_code": "asset",
            "trade_date": "date",
            "vol": "volume",
            **{f: f"raw_{f}" for f in ("open", "high", "low", "close")},
        }
    ).copy()
    suffix = result.asset.str.rsplit(".", n=1).str[-1]
    result["exchange"] = suffix.map({"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"})
    if result.exchange.isna().any():
        raise ValueError("Unsupported Tushare security suffix")
    result["date"] = pd.to_datetime(result.date, format="%Y%m%d")
    result["volume"] = pd.to_numeric(result.volume) * 100
    result["amount"] = pd.to_numeric(result.amount) * 1000
    return result[
        [
            "date",
            "asset",
            "exchange",
            *[f"raw_{p}" for p in ("open", "high", "low", "close")],
            "volume",
            "amount",
        ]
    ]


def cached_sina_sample(risk_root: str | Path) -> tuple:
    """Use actual native quotes; remove legacy valuation marks from price fields.

    Retrospective basket, dated adjustment/share proxies, and assumed EOD
    availability are explicitly labeled engineering-only. Unknown ST/limits and
    listing dates stay null. Native coverage is NOT proof of tradability.
    """
    root = Path(risk_root)
    cache = root / "data/raw/ashare"
    manifest_path = cache / "data_manifest.json"
    upstream = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {
        "quotes.parquet": upstream["quotes_sha256"],
        "valuation_quotes.parquet": upstream["valuation_quotes_sha256"],
    }
    for name, expected in files.items():
        if digest_file(cache / name) != expected:
            raise ValueError(f"Upstream cache changed: {name}")
    # Verify original payload identities, basket and the declared halt ledger.
    for name, expected in upstream["raw_sha256"].items():
        if digest_file(cache / name) != expected:
            raise ValueError(f"Upstream payload changed: {name}")
    for key, name in (
        ("universe_sha256", "ashare_universe.csv"),
        ("suspensions_sha256", "ashare_suspensions.csv"),
    ):
        if digest_file(root / "configs" / name) != upstream[key]:
            raise ValueError(f"Upstream declaration changed: {name}")
    native = pd.read_parquet(cache / "quotes.parquet")
    valuation = pd.read_parquet(cache / "valuation_quotes.parquet")
    reference = pd.read_parquet(cache / "calendar.parquet")
    basket = pd.read_csv(root / "configs/ashare_universe.csv", dtype={"asset": str})
    extra = ["date", "asset", "hfq_factor", "outstanding_share", "native_quote", "tradable"]
    panel = valuation[extra].merge(
        native[["date", "asset", "open", "high", "low", "close", "volume", "amount"]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(basket[["asset", "industry"]], on="asset", validate="many_to_one")
    panel = panel.rename(
        columns={
            **{p: f"raw_{p}" for p in ("open", "high", "low", "close")},
            "hfq_factor": "adj_factor",
        }
    )
    panel["exchange"] = panel.asset.str[:2].map({"sh": "SSE", "sz": "SZSE"})
    panel["float_market_cap"] = panel.raw_close * panel.outstanding_share
    # These are scenario assumptions, never certified historical trading states.
    panel["buyable"] = panel.tradable.astype(bool)
    panel["sellable"] = panel.tradable.astype(bool)
    panel["universe_member"] = True
    for field in ASOF_FIELDS:
        panel[field] = panel.date + pd.Timedelta(hours=18)
    panel["listing_asof"] = pd.NaT
    panel["corporate_actions_version"] = digest_file(manifest_path)
    calendar = pd.concat(
        [
            reference[["date"]].assign(exchange=exchange)
            for exchange in sorted(panel.exchange.unique())
        ],
        ignore_index=True,
    )
    lineage = {
        field: {"status": "missing", "source": None, "version": None} for field in FORMAL_FIELDS
    }
    for field in panel:
        if field in lineage and panel[field].notna().any():
            lineage[field] = {
                "status": "unverified_historical",
                "source": "local_sina_native_quotes_or_dated_events",
                "version": digest_file(manifest_path),
            }
    for field in ("industry", "universe_member", "universe_asof"):
        lineage[field] = {
            "status": "researcher_supplied_retrospective",
            "source": "configs/ashare_universe.csv; retrospective fixed basket",
            "version": upstream["universe_sha256"],
        }
    for field in ASOF_FIELDS:
        if panel[field].notna().any():
            lineage[field] = {
                "status": "assumed_eod_availability",
                "source": "engineering assumption: date + 18:00 Asia/Shanghai",
                "version": "v2_engineering_assumption_1",
            }
    for field in ("buyable", "sellable"):
        lineage[field] = {
            "status": "assumed_execution_scenario",
            "source": "legacy tradable/declaration proxy; no price-limit fill proof",
            "version": upstream["suspensions_sha256"],
        }
    lineage["float_market_cap"] = {
        "status": "derived_unverified_historical",
        "source": "native raw close x cached dated float shares",
        "version": digest_file(manifest_path),
    }
    metadata = {
        "schema_version": VERSION,
        "provider": "cached_sina_engineering_sample",
        "official_calendar": False,
        "survivorship_free": False,
        "publication_history_verified": False,
        "fields": lineage,
        "units": {"price": "CNY", "volume": "shares", "amount": "CNY"},
        "source_files": {
            str(cache / name): digest_file(cache / name)
            for name in (*files, "calendar.parquet", "data_manifest.json")
        },
        "limitations": upstream["limitations"]
        + [
            "ST/price limits/listing dates missing; tradability is an engineering assumption",
            "EOD availability assumed, not supplier-certified",
            "Adjusted open return proxy does not account for actual dividend cash",
        ],
    }
    metadata["source_files"].update(
        {str(cache / name): expected for name, expected in upstream["raw_sha256"].items()}
    )
    metadata["source_files"].update(
        {
            str(root / "configs/ashare_universe.csv"): upstream["universe_sha256"],
            str(root / "configs/ashare_suspensions.csv"): upstream["suspensions_sha256"],
        }
    )
    return panel, calendar, metadata
