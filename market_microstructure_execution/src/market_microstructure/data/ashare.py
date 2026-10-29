"""Real A-share Level-5 library adapters; never infer a book from bar/trade data."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from market_microstructure.utils.paths import PROJECT_ROOT


def asset_code(symbol: str) -> str:
    """Canonical mainland main-board code; preserve leading zeros."""
    code = symbol[-6:]
    if not code.isdigit() or len(code) != 6:
        raise ValueError("A-share code must contain six digits")
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    raise ValueError("This adapter supports Shanghai/Shenzhen A shares only")


def normalize_sina(snapshot: dict[str, dict], received_at: pd.Timestamp) -> pd.DataFrame:
    """Easyquotation's Sina bid/ask volumes are already shares, not lots.

    Model timestamp is receipt time. Exchange time is stored separately for freshness
    and provenance; no trade direction is fabricated from cumulative volume.
    """
    rows = []
    for symbol, values in snapshot.items():
        exchange = pd.Timestamp(f"{values['date']} {values['time']}", tz="Asia/Shanghai").tz_convert("UTC")
        row = {"timestamp": received_at, "exchange_timestamp": exchange,
               "received_timestamp": received_at, "asset": asset_code(symbol),
               "source": "easyquotation_sina", "timestamp_quality": "receipt_and_exchange",
               "name": values["name"], "last_trade_price": float(values["now"]),
               "cumulative_trade_size": float(values["turnover"]),
               "cumulative_trade_amount": float(values["volume"]),
               "stale_seconds": (received_at - exchange).total_seconds()}
        for level in range(1, 6):
            for side in ("bid", "ask"):
                row[f"{side}_price_{level}"] = float(values[f"{side}{level}"])
                row[f"{side}_size_{level}"] = float(values[f"{side}{level}_volume"])
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_akshare(snapshot: pd.DataFrame, symbol: str, received_at: pd.Timestamp) -> pd.DataFrame:
    """AKShare 1.19.1 stock_bid_ask_em already multiplies raw lots by 100.

    This API omits exchange timestamps; receipt time is explicitly not proof of a fresh quote.
    """
    values = snapshot.set_index("item").value
    row = {"timestamp": received_at, "received_timestamp": received_at, "asset": asset_code(symbol),
           "source": "akshare_eastmoney", "timestamp_quality": "receipt_only",
           "exchange_timestamp": pd.NaT, "stale_seconds": float("nan")}
    for level in range(1, 6):
        for side, provider_side in (("bid", "buy"), ("ask", "sell")):
            row[f"{side}_price_{level}"] = float(values[f"{provider_side}_{level}"])
            row[f"{side}_size_{level}"] = float(values[f"{provider_side}_{level}_vol"])
    return pd.DataFrame([row])


def quote_fingerprint(row: dict) -> str:
    """Deduplicate unchanged books independently of polling/receipt timestamp."""
    columns = [f"{side}_{field}_{level}" for level in range(1, 6)
               for side in ("bid", "ask") for field in ("price", "size")]
    values = [row["asset"], *[row[c] for c in columns], str(row.get("exchange_timestamp"))]
    return hashlib.sha256(json.dumps(values, allow_nan=False).encode()).hexdigest()


def fetch_library_snapshot(provider: str, symbols: list[str], timeout_seconds: int = 20,
                           bypass_proxy: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bound provider calls in an isolated subprocess; avoid global library monkeypatches."""
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "probe_ashare_sources.py"),
               "--worker", provider, "--symbols", *symbols]
    if bypass_proxy:
        command.append("--bypass-proxy")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(f"{provider} failed: {completed.stderr[-500:]}")
    payload = json.loads(completed.stdout)
    receipt = pd.Timestamp(datetime.now(timezone.utc))
    if provider == "sina":
        frame = normalize_sina(payload["quotes"], receipt)
    elif provider == "akshare":
        parts = [normalize_akshare(pd.DataFrame(records), symbol, receipt)
                 for symbol, records in payload["quotes"].items()]
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        raise ValueError("Use sina or akshare Level-5 snapshots")
    return frame, payload
