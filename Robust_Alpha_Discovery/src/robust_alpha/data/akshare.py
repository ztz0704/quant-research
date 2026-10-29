"""AKShare-backed A-share integration helpers.

AKShare is treated here as an integration/prototyping source. The completed market-date grid
prevents a suspension or missing quote from silently turning a later observation into the next
trading day's label.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from robust_alpha.config import ConfigError, load_config, project_path
from robust_alpha.data.loader import save_panel
from robust_alpha.data.schema import normalize_and_validate_panel

AKSHARE_COLUMNS = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}
CANONICAL_MARKET_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
]


def normalize_akshare_history(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Map one AKShare history response into observed canonical rows."""

    renamed = frame.rename(columns=AKSHARE_COLUMNS)
    missing = sorted(set(CANONICAL_MARKET_COLUMNS) - set(renamed.columns))
    if missing:
        raise ValueError(f"AKShare response for {symbol} is missing columns: {missing}")
    result = renamed[CANONICAL_MARKET_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["asset"] = str(symbol).zfill(6)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["date", "open", "high", "low", "close", "volume", "amount"]].isna().any().any():
        raise ValueError(f"AKShare response for {symbol} contains missing required market fields.")
    result["universe_member"] = True
    result["tradable"] = result["volume"].gt(0) & result["amount"].gt(0)
    return result[
        [
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
        ]
    ]


def complete_market_date_grid(observed: pd.DataFrame) -> pd.DataFrame:
    """Complete each asset onto the union market calendar."""

    if observed.empty:
        raise ValueError("Cannot construct a panel from an empty AKShare response.")
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(observed["date"]).unique()), name="date")
    completed: list[pd.DataFrame] = []
    for asset, group in observed.groupby("asset", sort=True, observed=True):
        quotes = group.sort_values("date").set_index("date")
        if quotes.index.has_duplicates:
            raise ValueError(f"AKShare returned duplicate dates for {asset}.")
        first_date = quotes.index.min()
        last_date = quotes.index.max()
        member = dates.to_series(index=dates).between(first_date, last_date)
        asset_frame = quotes.reindex(dates)
        was_observed = asset_frame["close"].notna()
        previous_close = asset_frame["close"].ffill()
        missing_member = member & ~was_observed
        for column in ("open", "high", "low", "close"):
            asset_frame.loc[missing_member, column] = previous_close.loc[missing_member]
        asset_frame.loc[missing_member, ["volume", "amount"]] = 0.0
        asset_frame["asset"] = str(asset)
        asset_frame["universe_member"] = member.to_numpy()
        asset_frame["tradable"] = (
            was_observed & asset_frame["volume"].gt(0) & asset_frame["amount"].gt(0)
        )
        completed.append(asset_frame.reset_index())
    return normalize_and_validate_panel(pd.concat(completed, ignore_index=True))


def download_akshare_panel(
    symbols: list[str],
    *,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    endpoint: str = "tencent",
    timeout: float = 20.0,
    retries: int = 3,
    pause_seconds: float = 0.15,
    fetcher: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Download and normalize a deterministic fixed-symbol A-share panel."""

    if not symbols:
        raise ValueError("At least one AKShare symbol is required.")
    if retries < 1:
        raise ValueError("retries must be positive.")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date.")
    if adjust not in {"", "qfq", "hfq"}:
        raise ValueError("adjust must be '', 'qfq', or 'hfq'.")
    if fetcher is None:
        import akshare as ak

        if endpoint == "eastmoney":
            fetcher = ak.stock_zh_a_hist
        elif endpoint == "tencent":

            def fetcher(**kwargs: Any) -> pd.DataFrame:
                raw_symbol = str(kwargs["symbol"])
                market_symbol = (
                    f"sh{raw_symbol}" if raw_symbol.startswith("6") else f"sz{raw_symbol}"
                )
                return ak.stock_zh_a_hist_tx(
                    symbol=market_symbol,
                    start_date=str(kwargs["start_date"]),
                    end_date=str(kwargs["end_date"]),
                    adjust=str(kwargs["adjust"]),
                    timeout=float(kwargs["timeout"]),
                )

        else:
            raise ValueError("endpoint must be 'tencent' or 'eastmoney'.")

    histories: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    ordered_symbols = sorted({str(symbol).zfill(6) for symbol in symbols})
    unsupported = [
        symbol for symbol in ordered_symbols if not symbol.startswith(("0", "2", "3", "6"))
    ]
    if unsupported:
        raise ValueError(f"Only Shanghai/Shenzhen A-share symbols are supported: {unsupported}")
    for position, symbol in enumerate(ordered_symbols, start=1):
        error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                raw = fetcher(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust=adjust,
                    timeout=timeout,
                )
                if raw.empty:
                    raise ValueError("empty response")
                histories.append(normalize_akshare_history(raw, symbol))
                error = None
                break
            except Exception as exc:  # provider errors vary by upstream endpoint
                error = exc
                if attempt < retries:
                    time.sleep(float(attempt))
        if error is not None:
            failures[symbol] = f"{type(error).__name__}: {error}"
        print(f"AKShare {position}/{len(ordered_symbols)}: {symbol}", flush=True)
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    if failures:
        details = "; ".join(f"{symbol}={message}" for symbol, message in failures.items())
        raise RuntimeError(f"AKShare download failed for {len(failures)} symbols: {details}")
    return complete_market_date_grid(pd.concat(histories, ignore_index=True))


def _disable_proxy_environment() -> None:
    os.environ["NO_PROXY"] = "*"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(name, None)


def fetch_from_config(
    config_path: str | Path,
    *,
    bypass_proxy: bool = False,
) -> tuple[Path, Path]:
    """Download the configured AKShare integration panel and local manifest."""

    if bypass_proxy:
        _disable_proxy_environment()
    root = Path.cwd().resolve()
    config_file = project_path(config_path, root)
    config = load_config(config_file)
    data = config["data"]
    if data.get("source") != "parquet" or data.get("provider") != "akshare":
        raise ConfigError("AKShare fetch config requires data.source=parquet and provider=akshare.")
    symbols = [str(symbol).zfill(6) for symbol in data.get("symbols", [])]
    panel = download_akshare_panel(
        symbols,
        start_date=str(data["start_date"]),
        end_date=str(data["end_date"]),
        adjust=str(data.get("adjust", "qfq")),
        endpoint=str(data.get("endpoint", "tencent")),
        timeout=float(data.get("timeout_seconds", 20)),
        retries=int(data.get("retries", 3)),
        pause_seconds=float(data.get("pause_seconds", 0.15)),
    )
    destination = save_panel(panel, project_path(data["path"], root))
    manifest_path = destination.with_name(f"{destination.stem}_manifest.json")
    manifest: dict[str, Any] = {
        "provider": "AKShare",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "config": str(config_file),
        "start_date": data["start_date"],
        "end_date": data["end_date"],
        "adjust": data.get("adjust", "qfq"),
        "endpoint": data.get("endpoint", "tencent"),
        "symbols": symbols,
        "rows": len(panel),
        "dates": int(panel["date"].nunique()),
        "assets": int(panel["asset"].nunique()),
        "caveats": [
            "Fixed hand-curated smoke universe; not point-in-time index membership history.",
            "Historical ST flags and official suspension reasons are unavailable from this feed.",
            "AKShare wraps third-party public endpoints without an institutional data SLA.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/akshare_smoke.yaml")
    parser.add_argument(
        "--bypass-proxy",
        action="store_true",
        help="Ignore system proxy variables for public market-data requests.",
    )
    arguments = parser.parse_args()
    panel_path, manifest_path = fetch_from_config(
        arguments.config, bypass_proxy=arguments.bypass_proxy
    )
    print(f"panel: {panel_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
