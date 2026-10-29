"""Historical A-share acquisition for the existing shared V3 research contract.

SDK history is retrospective evidence, never an authenticated publication archive.
Missing availability timestamps, auction permissions and cash/share actions stay null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from robust_alpha.data.history_api import BAO_FIELDS, TOKEN_NAMES, token
from robust_alpha.data.history_store import Archive, RequestFailed, write_json
from robust_alpha.data.pit_audit_v3 import audit_pit_v3
from robust_alpha.data.research_adapters import read_bundle
from robust_alpha.data.research_contract import (
    ASOF_FIELDS,
    FORMAL_FIELDS,
    PRICE_FIELDS,
    RETURN_CONVENTION,
    VERSION,
    add_calendar_labels,
    add_decision_eligibility,
    digest_file,
    normalize_dynamic_panel,
    require_formal_quality,
)

PROJECT = Path(__file__).resolve().parents[3]
PROJECTS = PROJECT.parent
SAMPLE_CODES = [
    "sh.600000",
    "sh.600004",
    "sh.600009",
    "sh.600016",
    "sh.600036",
    "sh.600050",
    "sz.000001",
    "sz.000024",
    "sz.000562",
    "sz.000594",
]


def asset_codes(codes: pd.Series) -> pd.Series:
    return codes.str[:2] + codes.str[3:]


def unique(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Remove identical retransmissions; conflicting revisions require explicit review."""
    result = frame.drop_duplicates().copy()
    if result.duplicated(keys).any():
        raise ValueError(f"Conflicting records for {keys}")
    return result


def normalize_bao(frame: pd.DataFrame) -> pd.DataFrame:
    raw = unique(frame, ["date", "code"])
    result = raw.rename(columns={f: f"raw_{f}" for f in ("open", "high", "low", "close")})
    result["date"] = pd.to_datetime(result.date)
    result["asset"] = asset_codes(result.code)
    result["exchange"] = result.code.str[:2].map({"sh": "SSE", "sz": "SZSE"})
    for field in [*PRICE_FIELDS, "preclose", "volume", "amount", "turn"]:
        result[field] = pd.to_numeric(result[field].replace("", np.nan), errors="raise")
    result["turnover_rate"] = result.turn / 100  # BaoStock percent -> decimal.
    for original, target in (("tradestatus", "suspended"), ("isST", "is_st")):
        if not result[original].isin(["0", "1", ""]).all():
            raise ValueError(f"Unknown BaoStock {original} state")
        result[target] = result[original].map({"0": False, "1": True}).astype("boolean")
    result["suspended"] = ~result.suspended
    halted = result.suspended.eq(True).fillna(False)
    result["vendor_halt_price_present"] = halted & result.raw_close.notna()
    # Preserve vendor marks in the raw archive; a suspended carry mark is not a new trade.
    result.loc[halted, PRICE_FIELDS] = np.nan
    result["buyable"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["sellable"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result.loc[halted, ["buyable", "sellable"]] = False
    result["source"] = "baostock"
    return result.drop(columns=["turn", "tradestatus", "isST"])


def normalize_lifecycle(frame: pd.DataFrame) -> pd.DataFrame:
    result = unique(frame, ["code"]).copy()
    result["asset"] = asset_codes(result.code)
    result["list_date"] = pd.to_datetime(result.ipoDate.replace("", None))
    result["delist_date"] = pd.to_datetime(result.outDate.replace("", None))
    result["listing_status"] = result.status
    return result[["asset", "code", "code_name", "list_date", "delist_date", "listing_status"]]


def calendar_from_bao(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    raw = unique(raw, ["calendar_date"])
    dates = pd.to_datetime(raw.calendar_date)
    expected = pd.date_range(start, end, freq="D")
    if set(dates) != set(expected) or not raw.is_trading_day.isin(["0", "1"]).all():
        raise ValueError("Calendar response truncated or contains invalid states")
    sessions = dates[raw.is_trading_day.eq("1")]
    return pd.concat(
        [pd.DataFrame({"date": sessions, "exchange": exchange}) for exchange in ("SSE", "SZSE")],
        ignore_index=True,
    )


def construct_panel(
    quotes: pd.DataFrame, calendar: pd.DataFrame, lifecycle: pd.DataFrame, factors: pd.DataFrame
) -> pd.DataFrame:
    panel = normalize_bao(quotes).merge(lifecycle, on=["asset", "code"], validate="many_to_one")
    panel["universe_member"] = pd.Series(pd.NA, index=panel.index, dtype="boolean")
    known = panel.list_date.notna()
    panel.loc[known, "universe_member"] = (
        panel.loc[known, "date"].ge(panel.loc[known, "list_date"])
        & (
            panel.loc[known, "delist_date"].isna()
            | panel.loc[known, "date"].lt(panel.loc[known, "delist_date"])
        )
    ).to_numpy()
    panel["adj_factor"] = np.nan
    # Sparse effective-date factors are carried only forwards, with an actual earlier anchor.
    # Never backfill the first event or manufacture a factor of 1 for missing histories.
    for code, group in panel.groupby("code"):
        events = unique(factors.loc[factors.code.eq(code)], ["code", "dividOperateDate"])
        if events.empty:
            continue
        events = events.assign(
            date=pd.to_datetime(events.dividOperateDate),
            factor=pd.to_numeric(events.backAdjustFactor.replace("", np.nan)),
        ).sort_values("date")
        joined = pd.merge_asof(
            group[["date"]].sort_values("date"),
            events[["date", "factor"]],
            on="date",
            direction="backward",
        )
        panel.loc[group.sort_values("date").index, "adj_factor"] = joined.factor.to_numpy()
    for field in ASOF_FIELDS:
        panel[field] = pd.NaT  # Date-of-event != actual publication/availability timestamp.
    expanded = normalize_dynamic_panel(panel, calendar)
    # Supply static lifecycle on generated grid rows, then retrospective date qualification.
    static = lifecycle.set_index("asset")
    for field in ("list_date", "delist_date", "listing_status"):
        expanded[field] = expanded.asset.map(static[field])
    known = expanded.list_date.notna()
    expanded.loc[known, "universe_member"] = (
        expanded.loc[known, "date"].ge(expanded.loc[known, "list_date"])
        & (
            expanded.loc[known, "delist_date"].isna()
            | expanded.loc[known, "date"].lt(expanded.loc[known, "delist_date"])
        )
    ).to_numpy()
    outside = expanded.universe_member.eq(False).fillna(False)
    expanded.loc[outside, ["buyable", "sellable"]] = False
    return expanded


def local_inventory() -> list[dict]:
    inventory = []
    for name in ("Robust_Alpha_Discovery", "nonstationary_financial_ml", "risk_aware_portfolio"):
        for path in sorted((PROJECTS / name / "data").rglob("*")):
            if path.suffix not in (".parquet", ".csv") or "historical_v3" in path.parts:
                continue
            item = {"path": str(path), "sha256": digest_file(path), "bytes": path.stat().st_size}
            try:
                if path.suffix == ".parquet":
                    parquet = pq.ParquetFile(path)
                    item.update(rows=parquet.metadata.num_rows, columns=parquet.schema.names)
                    if "date" in parquet.schema.names:
                        dates = pd.to_datetime(pd.read_parquet(path, columns=["date"]).date)
                        item.update(date_min=str(dates.min()), date_max=str(dates.max()))
                else:
                    frame = pd.read_csv(path, dtype=str)
                    item.update(rows=len(frame), columns=list(frame))
                item["pit_evidence"] = (
                    "unverified_local_cache; not used to fabricate missing fields"
                )
            except (ValueError, OSError) as exc:
                item["error"] = type(exc).__name__
            inventory.append(item)
    return inventory


class Collector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = (PROJECT / cfg["output"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive = Archive(self.root / "archive", **cfg["requests"])

    def probe(self) -> dict:
        records = []
        tasks = []
        if token():
            tasks.extend(
                ("tushare", name, params)
                for name, params in (
                    ("stock_basic", {"list_status": "D"}),
                    (
                        "trade_cal",
                        {"exchange": "SSE", "start_date": "20150101", "end_date": "20150109"},
                    ),
                    (
                        "daily",
                        {"ts_code": "600000.SH", "start_date": "20150105", "end_date": "20150109"},
                    ),
                    (
                        "adj_factor",
                        {"ts_code": "600000.SH", "start_date": "20150105", "end_date": "20150109"},
                    ),
                    ("daily_basic", {"ts_code": "600000.SH", "trade_date": "20150105"}),
                    ("suspend_d", {"trade_date": "20150105"}),
                    ("stk_limit", {"trade_date": "20150105"}),
                    ("dividend", {"ts_code": "600000.SH"}),
                    (
                        "index_weight",
                        {
                            "index_code": "000300.SH",
                            "start_date": "20150101",
                            "end_date": "20150109",
                        },
                    ),
                    ("index_member_all", {"ts_code": "600000.SH"}),
                    ("namechange", {"ts_code": "600000.SH"}),
                )
            )
        else:
            records.append(
                {
                    "source": "tushare",
                    "status": "not_configured",
                    "permissions": "not_tested; no environment credential",
                    "token_names": TOKEN_NAMES,
                }
            )
        tasks.extend(
            [
                ("baostock", "calendar", {"start_date": "2015-01-01", "end_date": "2015-01-09"}),
                ("baostock", "basic", {}),
                ("baostock", "universe", {"day": "2015-01-05"}),
                (
                    "baostock",
                    "quotes",
                    {
                        "code": "sh.600000",
                        "fields": BAO_FIELDS,
                        "start_date": "2015-01-05",
                        "end_date": "2015-01-09",
                        "adjustflag": "3",
                    },
                ),
                (
                    "baostock",
                    "adjustment",
                    {"code": "sh.600000", "start_date": "1990-01-01", "end_date": "2025-12-31"},
                ),
                (
                    "baostock",
                    "dividends",
                    {"code": "sh.600000", "year": "2015", "yearType": "operate"},
                ),
                ("baostock", "industry", {"date": "2015-01-05"}),
                ("baostock", "hs300", {"date": "2015-01-05"}),
                ("baostock", "terminated", {"date": "2015-12-31"}),
                ("baostock", "suspended", {"date": "2015-01-05"}),
                ("baostock", "st", {"date": "2015-01-05"}),
                (
                    "akshare",
                    "eastmoney",
                    {
                        "symbol": "600000",
                        "start_date": "20150105",
                        "end_date": "20150109",
                        "adjust": "",
                        "timeout": 25,
                    },
                ),
                (
                    "akshare",
                    "tencent",
                    {
                        "symbol": "sh600000",
                        "start_date": "20150105",
                        "end_date": "20150109",
                        "adjust": "",
                        "timeout": 25,
                    },
                ),
                ("akshare", "calendar", {}),
                ("akshare", "delisted_sz", {"symbol": "终止上市公司"}),
                ("akshare", "delisted_sh", {"symbol": "全部"}),
                (
                    "akshare",
                    "share_changes",
                    {"symbol": "600000", "start_date": "20150101", "end_date": "20151231"},
                ),
            ]
        )
        for source, endpoint, params in tasks:
            info = {
                "source": source,
                "endpoint": endpoint,
                "params": params,
                "pit_evidence": "vendor history; publication/revision archive not certified",
            }
            try:
                frame = self.archive.get(source, endpoint, **params)
                info.update(
                    status="accessible" if len(frame) else "empty",
                    rows=len(frame),
                    columns=list(frame),
                    permissions="observed_this_call_only",
                )
                date_columns = [c for c in frame if "date" in c.lower() or "日期" in c]
                info["returned_date_range"] = {
                    c: {"min": str(frame[c].min()), "max": str(frame[c].max())}
                    for c in date_columns
                    if len(frame)
                }
            except RequestFailed as exc:
                info.update(
                    status="failed", failure=str(exc), permissions="unavailable; see failure"
                )
            records.append(info)
            print(f"PROBE {source}/{endpoint}: {info['status']}", flush=True)
        inventory = local_inventory()
        report = {
            "sources": records,
            "local_inventory": inventory,
            "scope": [self.cfg["start"], self.cfg["end"]],
            "status": "EXPLORATORY_ONLY",
            "formal_ready": False,
            "credentials_logged": False,
        }
        write_json(self.root / "source_access.json", report)
        return report

    def collect(self, phase: str) -> dict:
        start, end = self.cfg["start"], self.cfg["end"]
        raw_cal = self.archive.get("baostock", "calendar", start_date=start, end_date=end)
        target_calendar = calendar_from_bao(raw_cal, start, end)
        registry_raw = self.archive.get("baostock", "basic")
        registry_raw = registry_raw.loc[
            registry_raw.type.eq("1") & registry_raw.code.str.match(r"^(sh\.6|sz\.[03])")
        ]
        lifecycle = normalize_lifecycle(registry_raw)
        discovery = self.archive.get("baostock", "universe", day="2015-01-05")
        # Registry contains inactive securities. Historical discovery is an independent check.
        if phase == "sample":
            dates = sorted(target_calendar.date.unique())[:5]
            selected = lifecycle.loc[lifecycle.code.isin(SAMPLE_CODES)].copy()
            calendar = target_calendar.loc[target_calendar.date.isin(dates)]
        elif phase == "year":
            self.check_stage("sample")
            dates = sorted(
                target_calendar.loc[
                    target_calendar.date.dt.year.eq(self.cfg["test_year"]), "date"
                ].unique()
            )
            selected = lifecycle.loc[lifecycle.code.isin([*SAMPLE_CODES, "sz.300431"])].copy()
            calendar = target_calendar.loc[target_calendar.date.isin(dates)]
        elif phase == "range":
            self.check_stage("sample")
            self.check_stage("year")
            selected = lifecycle.loc[lifecycle.code.isin([*SAMPLE_CODES, "sz.300431"])].copy()
            dates = sorted(target_calendar.date.unique())
            calendar = target_calendar
        else:
            self.check_stage("sample")
            self.check_stage("year")
            if not self.cfg["allow_exploratory_bulk"]:
                previous = json.loads(
                    (self.root / "year" / "acceptance.json").read_text(encoding="utf-8")
                )
                if not previous["formal_ready"]:
                    write_json(
                        self.root / "bulk_status.json",
                        {
                            "status": "BLOCKED_FORMAL_PIT",
                            "reason": "Year lacks authenticated PIT; exploratory bulk disabled",
                            "target": [start, end],
                            "data_preserved": True,
                        },
                    )
                    return {"status": "BLOCKED_FORMAL_PIT"}
            overlap = lifecycle.list_date.le(pd.Timestamp(end)) & (
                lifecycle.delist_date.isna() | lifecycle.delist_date.ge(pd.Timestamp(start))
            )
            # Reserve delisted stocks without market cap or future return ranking.
            selected = lifecycle.loc[overlap].sort_values(
                ["delist_date", "code"], na_position="last"
            )
            selected = selected.head(self.cfg["max_assets"])
            dates = sorted(target_calendar.date.unique())
            calendar = target_calendar
        if phase == "sample" and len(selected) != 10:
            raise ValueError("The historical registry did not resolve all 10 sample stocks")
        begin, finish = (
            pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
            pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        )
        frames, adjustments, failures = [], [], []
        planned = []
        for row in selected.itertuples():
            for year in sorted({pd.Timestamp(d).year for d in dates}):
                lo, hi = max(begin, f"{year}-01-01"), min(finish, f"{year}-12-31")
                if row.list_date > pd.Timestamp(hi) or (
                    pd.notna(row.delist_date) and row.delist_date <= pd.Timestamp(lo)
                ):
                    continue
                planned.append({"code": row.code, "start_date": lo, "end_date": hi})
        write_json(
            self.root / phase / "plan.json",
            {
                "partitions": planned,
                "assets": selected.code.tolist(),
                "dates": len(dates),
                "scope": [begin, finish],
                "selection": "historical lifecycle; delisted retained",
            },
        )
        for index, part in enumerate(planned):
            try:
                frame = self.archive.get(
                    "baostock", "quotes", **part, fields=BAO_FIELDS, adjustflag="3"
                )
                if frame.empty:
                    raise RequestFailed("Empty quote partition during registered listing lifetime")
                info = self.archive.last_info
                frame["downloaded_at"] = info["downloaded_at"]
                frame["raw_partition_sha256"] = info["raw_sha256"]
                frames.append(frame)
            except RequestFailed as exc:
                failures.append({**part, "failure": str(exc)})
            if (index + 1) % 10 == 0 or index + 1 == len(planned):
                print(f"{phase.upper()} {index + 1}/{len(planned)} {part['code']}", flush=True)
        for code in selected.code:
            try:
                adjustments.append(
                    self.archive.get(
                        "baostock",
                        "adjustment",
                        code=code,
                        start_date="1990-01-01",
                        end_date=finish,
                    )
                )
            except RequestFailed as exc:
                failures.append({"code": code, "kind": "adjustment", "failure": str(exc)})
        write_json(self.root / phase / "incomplete_partitions.json", {"failures": failures})
        if not frames:
            raise RequestFailed("No successful quote partitions; preserved all failures")
        factors = (
            pd.concat(adjustments, ignore_index=True)
            if adjustments
            else pd.DataFrame(columns=["code", "dividOperateDate", "backAdjustFactor"])
        )
        quotes = pd.concat(frames, ignore_index=True)
        panel = construct_panel(quotes, calendar, selected, factors)
        panel["download_source"] = "baostock"
        panel["source_archive"] = str(self.archive.root)
        return self.export(phase, panel, calendar, selected, factors, discovery, failures)

    def check_stage(self, phase: str) -> None:
        report = json.loads((self.root / phase / "acceptance.json").read_text(encoding="utf-8"))
        if not report["engineering_passed"]:
            raise ValueError(f"Previous {phase} acceptance failed; no bulk acquisition")

    def crosscheck(self) -> dict:
        results = []
        for code in ("sh.600000", "sz.000001"):
            bao = normalize_bao(
                self.archive.get(
                    "baostock",
                    "quotes",
                    code=code,
                    fields=BAO_FIELDS,
                    start_date="2015-01-05",
                    end_date="2015-01-09",
                    adjustflag="3",
                )
            )
            for endpoint in ("eastmoney", "tencent"):
                params = {
                    "symbol": code[3:] if endpoint == "eastmoney" else code.replace(".", ""),
                    "start_date": "20150105",
                    "end_date": "20150109",
                    "adjust": "",
                    "timeout": 25,
                }
                try:
                    other = self.archive.get("akshare", endpoint, **params)
                    if endpoint == "eastmoney":
                        other = other.rename(
                            columns={
                                "日期": "date",
                                "开盘": "open",
                                "最高": "high",
                                "最低": "low",
                                "收盘": "close",
                                "成交量": "volume",
                                "成交额": "amount",
                            }
                        )
                        other["volume"] = pd.to_numeric(other.volume) * 100
                    other["date"] = pd.to_datetime(other.date)
                    other = unique(other, ["date"])
                    joined = bao.merge(
                        other, on="date", suffixes=("_bao", "_ak"), validate="one_to_one"
                    )
                    price_ok = all(
                        np.allclose(joined["raw_" + f], joined[f], atol=0.02, rtol=0)
                        for f in ("open", "high", "low", "close")
                    )
                    volume_ok = np.allclose(joined.volume_bao, joined.volume_ak, rtol=0.0001)
                    amount_ok = np.allclose(joined.amount_bao, joined.amount_ak, rtol=0.002)
                    results.append(
                        {
                            "code": code,
                            "endpoint": endpoint,
                            "overlap_rows": len(joined),
                            "price_match": bool(price_ok),
                            "volume_match": bool(volume_ok),
                            "amount_match": bool(amount_ok),
                            "volume_bao_to_ak_median": float(
                                (joined.volume_bao / joined.volume_ak).median()
                            ),
                            "accepted_as_primary": False,
                        }
                    )
                except RequestFailed as exc:
                    results.append({"code": code, "endpoint": endpoint, "failure": str(exc)})
        bao_cal = calendar_from_bao(
            self.archive.get(
                "baostock", "calendar", start_date=self.cfg["start"], end_date=self.cfg["end"]
            ),
            self.cfg["start"],
            self.cfg["end"],
        )
        ak_cal = self.archive.get("akshare", "calendar")
        dates = pd.to_datetime(ak_cal.trade_date)
        ak_dates = set(dates[dates.between(self.cfg["start"], self.cfg["end"])])
        bao_dates = set(bao_cal.date)
        report = {
            "quotes": results,
            "calendar_symmetric_difference": [str(d) for d in sorted(ak_dates ^ bao_dates)],
            "calendar_sessions": len(bao_dates),
            "official_pit_certified": False,
            "policy": "Mismatched supplements excluded; never silently rescale a supplier",
            "primary_ohlcv_crosscheck_passed": all(
                x.get("overlap_rows") == 5
                and x.get("price_match")
                and x.get("volume_match")
                and x.get("amount_match")
                for x in results
                if x["endpoint"] == "eastmoney"
            ),
        }
        write_json(self.root / "crosscheck.json", report)
        print(json.dumps(report), flush=True)
        return report

    def export(
        self,
        phase: str,
        panel: pd.DataFrame,
        calendar: pd.DataFrame,
        lifecycle: pd.DataFrame,
        factors: pd.DataFrame,
        discovery: pd.DataFrame,
        failures: list,
    ) -> dict:
        output = self.root / phase
        output.mkdir(parents=True, exist_ok=True)
        references = self.reference_tables(output)
        panel.to_parquet(output / "canonical_market_panel.parquet", index=False)
        (output / "panel.parquet").unlink(missing_ok=True)  # Remove this collector's legacy alias.
        calendar.to_parquet(output / "calendar.parquet", index=False)
        lifecycle.to_parquet(output / "lifecycle_registry.parquet", index=False)
        factors.to_parquet(output / "adjustment_events.parquet", index=False)
        discovery.to_parquet(output / "historical_discovery.parquet", index=False)
        panel[["date", "asset", "exchange", "universe_member", "list_date", "delist_date"]].assign(
            evidence="retrospective_lifecycle_not_publication_verified"
        ).to_parquet(output / "historical_membership.parquet", index=False)
        checkpoint = self.archive.state
        write_json(
            output / "download_manifest.json",
            {
                "archive": str(self.archive.root),
                "requests": checkpoint,
                "raw_format": "immutable SDK response JSON before normalization, not wire capture",
            },
        )
        version = digest_file(output / "download_manifest.json")
        metadata = {
            "schema_version": VERSION,
            "panel_file": "canonical_market_panel.parquet",
            "status": "EXPLORATORY_ONLY",
            "formal_ready": False,
            "units": {"price": "CNY", "volume": "shares", "amount": "CNY"},
            "return_convention": RETURN_CONVENTION,
            "return_unit": "decimal_simple",
            "timezone": "Asia/Shanghai",
            "official_calendar": False,
            "survivorship_free": False,
            "publication_history_verified": False,
            "calendar_evidence": "BaoStock sessions without primary exchange attestation",
            "membership_evidence": "retrospective lifecycle; IPO inclusive, delist exclusive",
            "industry_evidence": "vendor snapshot kept separately; no projection over dates",
            "fields": {
                f: {
                    "status": "unverified_historical" if panel[f].notna().any() else "missing",
                    "source": "baostock" if panel[f].notna().any() else None,
                    "version": version if panel[f].notna().any() else None,
                }
                for f in FORMAL_FIELDS
            },
            "reference_files": references,
            "files": {p.name: digest_file(p) for p in output.glob("*.parquet")},
            "audit_files": {
                "lifecycle_registry": {
                    "filename": "lifecycle_registry.parquet",
                    "sha256": digest_file(output / "lifecycle_registry.parquet"),
                }
            },
        }
        write_json(output / "metadata.json", metadata)
        audited = audit_pit_v3(panel, metadata, lifecycle_registry=lifecycle)
        write_json(output / "v3_pit_audit.json", audited.pop("report"))
        for name, frame in audited.items():
            frame.to_parquet(output / f"{name}.parquet", index=False)
        native = panel[PRICE_FIELDS].notna().all(axis=1)
        members = panel.universe_member.eq(True).fillna(False)
        halt = panel.suspended.eq(True).fillna(False)
        missing = members & ~native & ~halt
        gaps = panel.loc[missing, ["date", "asset", "list_date", "delist_date", "suspended"]]
        gaps.to_parquet(output / "unexplained_quote_gaps.parquet", index=False)
        traded = native & panel.volume.gt(0) & panel.amount.gt(0)
        vwap = panel.amount / panel.volume
        unit_errors = traded & (vwap.lt(panel.raw_low * 0.95) | vwap.gt(panel.raw_high * 1.05))
        panel.loc[
            unit_errors, ["date", "asset", "volume", "amount", "raw_low", "raw_high"]
        ].to_parquet(output / "unit_anomalies.parquet", index=False)
        live = lifecycle.loc[
            lifecycle.delist_date.notna()
            & lifecycle.delist_date.between(calendar.date.min(), calendar.date.max())
        ]
        passed = not failures and gaps.empty and not unit_errors.any()
        if phase == "year":
            passed &= len(live) > 0 and bool(halt.any())
        report = {
            "phase": phase,
            "status": "EXPLORATORY_ONLY",
            "engineering_passed": bool(passed),
            "formal_ready": False,
            "assets": int(panel.asset.nunique()),
            "trading_days": int(panel.date.nunique()),
            "stock_date_rows": len(panel),
            "native_price_rows": int(native.sum()),
            "calendar_range": [str(calendar.date.min().date()), str(calendar.date.max().date())],
            "registered_delisted_assets": int(lifecycle.delist_date.notna().sum()),
            "delisted_in_test_window": live.asset.tolist(),
            "known_halt_rows": int(halt.sum()),
            "unexplained_member_gaps": len(gaps),
            "amount_volume_unit_anomalies": int(unit_errors.sum()),
            "failed_partitions": failures,
            "historical_listing_registry_total": len(lifecycle),
            "field_missing_rates": {
                f: float(panel[f].isna().mean()) for f in [*FORMAL_FIELDS, "turnover_rate"]
            },
            "pit_eligible_rows": int(add_decision_eligibility(panel, min_history=1).eligible.sum()),
            "valid_return_labels": int(add_calendar_labels(panel, calendar).target_valid.sum()),
            "unresolved": [
                "Publication/revision times not available; ASOF fields remain NaT",
                "Auction buy/sell permissions and official price limits unavailable",
                "No authenticated corporate-action cash/share/payment/settlement ledger",
                "Historical industry/index records are vendor snapshots, not certified PIT",
                "Historic cap data unavailable; no shares inferred from turnover",
                "BaoStock covers SH/SZ sample; Beijing market not covered",
                "Sample lifecycle selection does not establish whole-market survivorship freedom",
            ],
        }
        write_json(output / "acceptance.json", report)
        write_json(
            output / "manifest.json",
            {
                "files": {
                    p.name: digest_file(p)
                    for p in output.iterdir()
                    if p.is_file() and p.name != "manifest.json"
                },
                "status": "EXPLORATORY_ONLY",
                "source_manifest_sha256": version,
            },
        )
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "phase",
                        "assets",
                        "trading_days",
                        "stock_date_rows",
                        "native_price_rows",
                        "engineering_passed",
                        "formal_ready",
                    )
                }
            ),
            flush=True,
        )
        return report

    def reference_tables(self, output: Path) -> dict:
        kinds = {
            "industry",
            "hs300",
            "dividends",
            "terminated",
            "suspended",
            "st",
            "delisted_sz",
            "delisted_sh",
            "share_changes",
        }
        grouped = {}
        for info in self.archive.state.values():
            if info["status"] != "complete" or info["endpoint"] not in kinds:
                continue
            key = info["source"] + "_" + info["endpoint"]
            frame = pd.read_parquet(self.archive.root / info["parquet"])
            frame["downloaded_at"] = info["downloaded_at"]
            frame["source_request"] = json.dumps(info["params"], sort_keys=True)
            frame["evidence_grade"] = "retrospective_vendor; no revision/publication authentication"
            grouped.setdefault(key, []).append(frame)
        files = {}
        for key, frames in grouped.items():
            name = key + "_reference.parquet"
            pd.concat(frames, ignore_index=True).to_parquet(output / name, index=False)
            files[name] = digest_file(output / name)
        return files


def verify_bundle(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["files"].items():
        if digest_file(root / name) != sha:
            raise ValueError(f"Changed bundle file: {name}")
    raw, calendar, metadata = read_bundle(root)
    panel = normalize_dynamic_panel(raw, calendar)
    report = audit_pit_v3(
        panel, metadata, lifecycle_registry=pd.read_parquet(root / "lifecycle_registry.parquet")
    )["report"]
    if report["formal_ready"]:
        raise ValueError("Vendor-only acquisition unexpectedly passed PIT certification")
    try:
        require_formal_quality(report)
    except ValueError:
        blocked = True
    else:
        blocked = False
    return {
        "schema_valid": True,
        "hashes_valid": True,
        "formal_research_blocked": blocked,
        "rows": len(panel),
        "assets": int(panel.asset.nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["probe", "sample", "year", "range", "bulk", "all", "verify", "crosscheck"]
    )
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/historical_data.yaml")
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.stage == "verify":
        print(json.dumps(verify_bundle(args.bundle.resolve())))
        return
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    collector = Collector(cfg)
    if args.stage == "crosscheck":
        collector.crosscheck()
        return
    if args.stage in ("probe", "all"):
        collector.probe()
    for stage in ["sample", "year", "range", "bulk"] if args.stage == "all" else [args.stage]:
        if stage != "probe":
            collector.collect(stage)


if __name__ == "__main__":
    main()
