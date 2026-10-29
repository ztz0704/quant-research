"""Fetch recent REAL unadjusted bars with AKShare; audit instead of inventing data.

Run using Robust_Alpha_Discovery's environment. Worker processes bound vendor calls
that may contain internal requests without a timeout. This is a source probe, not
a certified market calendar, a PIT universe, or a current signal generator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["date", "open", "high", "low", "close", "volume", "amount"]
EM_COLUMNS = dict(
    zip(["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额"], FIELDS, strict=True)
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tencent_volume_multiplier(symbol: str, version: str, source: str) -> int:
    # Actual 1.19.1 source exempts sz000 (ordinary stocks) together with index
    # prefixes from the lot-to-share conversion. Confirmed against Eastmoney.
    # Match the faulty implementation as well as the version, avoiding a double
    # conversion if that same version has been patched locally.
    faulty_clause = 'not symbol.startswith(("sh688", "sz399", "sh000", "sz000"))'
    return (
        100 if version == "1.19.1" and symbol.startswith("000") and faulty_clause in source else 1
    )


def normalize_bars(frame: pd.DataFrame, provider: str, *, tx_multiplier: int = 1) -> pd.DataFrame:
    if provider == "eastmoney":
        frame = frame.rename(columns=EM_COLUMNS)
    if not set(FIELDS).issubset(frame.columns) or frame.empty:
        raise ValueError("Empty response or missing OHLCV/amount columns")
    result = frame[FIELDS].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    for name in FIELDS[1:]:
        result[name] = pd.to_numeric(result[name], errors="raise")
    if provider == "eastmoney":
        result["volume"] *= 100  # documented lot unit; canonical unit is shares
    elif provider == "tencent":
        result["volume"] *= tx_multiplier
    return result.sort_values("date").reset_index(drop=True)


def audit_bars(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    reasons = []
    if frame.empty:
        return ["empty_response"]
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        reasons.append("invalid_or_duplicate_dates")
    if not frame["date"].between(start, end).all():
        reasons.append("outside_requested_dates")
    if frame["date"].max() != end:
        reasons.append("not_at_expected_last_session")
    values = frame[FIELDS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        reasons.append("nonfinite_market_fields")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        reasons.append("nonpositive_prices")
    if (frame[["volume", "amount"]] < 0).any().any():
        reasons.append("negative_volume_or_amount")
    if not (
        (frame["low"] <= frame[["open", "close"]].min(axis=1))
        & (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame["high"])
    ).all():
        reasons.append("invalid_ohlc_geometry")
    active = frame["volume"].gt(0) & frame["amount"].gt(0)
    if not active.any():
        reasons.append("no_active_quotes")
    if (frame["volume"].eq(0) != frame["amount"].eq(0)).any():
        reasons.append("volume_amount_zero_mismatch")
    # This catches 100x share/lot or 10000x currency-unit errors. A tolerance
    # handles vendor rounding; it is not a proof of the exchange's original tape.
    vwap = frame.loc[active, "amount"] / frame.loc[active, "volume"]
    if not vwap.between(frame.loc[active, "low"] * 0.98, frame.loc[active, "high"] * 1.02).all():
        reasons.append("amount_volume_implied_price_outside_bar_check_units")
    return reasons


def worker(args: argparse.Namespace) -> int:
    import akshare as ak

    try:
        volume_multiplier = 100 if args.provider == "eastmoney" else 1
        vendor_source_sha256 = None
        if args.provider == "tencent":
            source = inspect.getsource(ak.stock_zh_a_hist_tx)
            vendor_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
            volume_multiplier = tencent_volume_multiplier(
                args.symbol, importlib.metadata.version("akshare"), source
            )
            prefix = "sh" if args.symbol.startswith("6") else "sz"
            raw = ak.stock_zh_a_hist_tx(
                symbol=prefix + args.symbol,
                start_date=args.start.replace("-", ""),
                end_date=args.end.replace("-", ""),
                adjust="",
                timeout=args.timeout,
            )
        else:
            raw = ak.stock_zh_a_hist(
                symbol=args.symbol,
                period="daily",
                start_date=args.start.replace("-", ""),
                end_date=args.end.replace("-", ""),
                adjust="",
                timeout=args.timeout,
            )
        # Preserve vendor values before normalization, including suspect units.
        raw.to_parquet(Path(args.destination).with_suffix(".vendor.parquet"), index=False)
        normalize_bars(raw, args.provider, tx_multiplier=volume_multiplier).to_parquet(
            args.destination, index=False
        )
        print(
            json.dumps(
                {
                    "volume_multiplier": volume_multiplier,
                    "vendor_function_sha256": vendor_source_sha256,
                }
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        # Avoid printing proxy URLs/credentials embedded in transport exceptions.
        print(json.dumps({"error_type": type(exc).__name__}), flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-last-session")
    parser.add_argument("--symbols", nargs="+", default=["600000", "000001"])
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=12)
    parser.add_argument("--worker-deadline", type=float, default=35)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--provider", choices=["tencent", "eastmoney"], help=argparse.SUPPRESS)
    for name in ["symbol", "start", "end", "destination"]:
        parser.add_argument("--" + name, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if not args.expected_last_session:
        parser.error("Confirm the official last completed session with --expected-last-session")
    end = pd.Timestamp(args.expected_last_session).normalize()
    now = datetime.now(UTC)
    local_date = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    if end > local_date or end < local_date - pd.Timedelta(days=14):
        parser.error("Expected session must be recent and cannot be in the future")
    if args.lookback_days < 1 or args.timeout <= 0 or args.worker_deadline <= 0:
        parser.error("Lookback and timeout parameters must be positive")
    if any(
        not re.fullmatch(r"(?:000|001|002|003|300|301|600|601|603|605|688)\d{3}", x)
        for x in args.symbols
    ):
        parser.error("Use six-digit Shanghai/Shenzhen A-share symbols")
    start = end - pd.Timedelta(days=args.lookback_days)
    output = args.output or Path(__file__).resolve().parents[1] / "outputs" / (
        "recent_data_" + now.strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit_code.py").write_bytes(Path(__file__).read_bytes())
    results, valid = [], {}
    for symbol in sorted(set(args.symbols)):
        for provider in ["tencent", "eastmoney"]:
            destination = output / f"{provider}_{symbol}.parquet"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--provider",
                provider,
                "--symbol",
                symbol,
                "--start",
                str(start.date()),
                "--end",
                str(end.date()),
                "--destination",
                str(destination),
                "--timeout",
                str(args.timeout),
            ]
            record = {"provider": provider, "asset": symbol, "passed": False}
            try:
                run = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=args.worker_deadline,
                )
                if run.returncode:
                    record["reasons"] = ["vendor_worker_failed"]
                    for line in run.stdout.splitlines():
                        if line.startswith('{"error_type":'):
                            record.update(json.loads(line))
                else:
                    for line in run.stdout.splitlines():
                        if line.startswith('{"volume_multiplier":'):
                            record.update(json.loads(line))
                    bars = pd.read_parquet(destination)
                    reasons = audit_bars(bars, start, end)
                    record.update(
                        rows=len(bars),
                        last_date=str(bars["date"].max().date()),
                        reasons=reasons,
                        passed=not reasons,
                    )
                    valid[(provider, symbol)] = bars
            except subprocess.TimeoutExpired:
                record["reasons"] = ["vendor_worker_deadline_exceeded"]
            record["files"] = {
                p.name: sha256(p)
                for p in [destination, destination.with_suffix(".vendor.parquet")]
                if p.is_file()
            }
            results.append(record)
            print(f"{provider} {symbol}: {record.get('reasons', [])}", flush=True)
    comparisons = []
    for symbol in sorted(set(args.symbols)):
        pair = [valid.get((p, symbol)) for p in ["tencent", "eastmoney"]]
        if any(x is None for x in pair):
            comparisons.append({"asset": symbol, "status": "unavailable_two_sources_required"})
            continue
        merged = pair[0].merge(pair[1], on="date", suffixes=("_tx", "_em"))
        same_dates = set(pair[0]["date"]) == set(pair[1]["date"])
        max_difference = (
            max(
                float((merged[x + "_tx"] - merged[x + "_em"]).abs().max())
                for x in ["open", "high", "low", "close"]
            )
            if len(merged)
            else None
        )
        comparisons.append(
            {
                "asset": symbol,
                "shared_dates": len(merged),
                "same_dates": same_dates,
                "max_ohlc_difference": max_difference,
                "scope": "OHLC only; volume/amount identities checked separately",
                "status": "passed"
                if same_dates and max_difference is not None and max_difference <= 0.01000001
                else "mismatch",
            }
        )
    audit = {
        "retrieved_at_utc": now.isoformat(),
        "expected_last_session": str(end.date()),
        "start_date": str(start.date()),
        "adjustment": "none",
        "volume_unit": "shares",
        "amount_unit": "CNY",
        "akshare_version": importlib.metadata.version("akshare"),
        "script_sha256": sha256(Path(__file__)),
        "results": results,
        "comparisons": comparisons,
        "all_passed": all(x["passed"] for x in results)
        and all(x["status"] == "passed" for x in comparisons),
        "limits": [
            "Small source probe, not full-universe coverage or PIT certification",
            "Expected last session is caller-confirmed, not inferred from weekdays",
            "Vendor requests may internally fetch more history; it is not evaluated",
            "No strategy selection, orders, brokerage access or synthetic fallback",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# 近期真实行情接口核验",
        "",
        f"核验截止：{end.date()}；价格：未复权。",
        "",
        "通过代表本次指定样本的接口检查通过；不代表股票池/PIT/交易部署已完成。",
        "",
        "```json",
        json.dumps(audit, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Manifest: {output / 'manifest.json'}", flush=True)
    return 0 if audit["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
