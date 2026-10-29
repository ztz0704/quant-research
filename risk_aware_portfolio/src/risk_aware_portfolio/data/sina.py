"""Sina raw endpoints decoded with AKShare, without its implicit OHLC forward-fill.

AKShare is an optional extra. Imports stay inside provider methods so the core
and offline tests work without any data vendor installed.
"""

import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

from risk_aware_portfolio.utils.manifest import file_hash

logger = logging.getLogger("risk_aware_portfolio")


class SinaClient:
    """Serial, rate-limited acquisition with raw payload hashes and explicit errors."""

    def __init__(
        self, cache: Path, timeout: float, retries: int, pause: float, bypass_proxy: bool = False
    ) -> None:
        import requests

        self.session = requests.Session()
        self.session.trust_env = not bypass_proxy
        self.cache, self.timeout, self.retries, self.pause = cache, timeout, retries, pause
        self.cache.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Release sockets after the batch."""
        self.session.close()

    def payload(self, symbol: str, kind: str, refresh: bool = False) -> str:
        """Persist exact upstream text and query metadata; cached data is hash checked."""
        from akshare.stock.cons import (
            zh_sina_a_stock_amount_url,
            zh_sina_a_stock_hfq_url,
            zh_sina_a_stock_hist_url,
        )

        urls = {
            "prices": zh_sina_a_stock_hist_url.format(symbol),
            "shares": zh_sina_a_stock_amount_url.format(symbol, symbol),
            "adjustments": zh_sina_a_stock_hfq_url.format(symbol),
            "shares_history": (
                "https://money.finance.sina.com.cn/corp/go.php/vCI_StockStructureHistory/"
                f"stockid/{symbol[2:]}/stocktype/LiuTongA.phtml"
            ),
        }
        path = self.cache / f"{symbol}_{kind}.txt"
        metadata_path = path.with_suffix(".json")
        if path.exists() and not refresh:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["sha256"] != file_hash(path):
                raise ValueError(f"Modified cached payload: {path}")
            return path.read_text(encoding="utf-8")
        text = self._download(urls[kind])
        path.write_text(text, encoding="utf-8")
        metadata = {
            "url": urls[kind],
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "sha256": file_hash(path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return text

    def _download(self, url: str) -> str:
        """Retry bounded network failures; never substitute an empty successful result."""
        import requests

        for attempt in range(self.retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                response.encoding = "gb18030" if "StockStructureHistory" in url else "utf-8"
                if not response.text.strip():
                    raise ValueError(f"Empty response: {url}")
                time.sleep(self.pause)
                return response.text
            except requests.RequestException:
                logger.exception("Acquisition failed: %s attempt %d", url, attempt + 1)
                if attempt + 1 == self.retries:
                    raise
                time.sleep(min(2**attempt, 5))
        raise RuntimeError("Unreachable retry state")


def decode_prices(text: str) -> pd.DataFrame:
    """Decode native quote rows; preserve their dates and detect duplicate records."""
    from akshare.stock.cons import hk_js_decode
    from py_mini_racer import MiniRacer

    encoded = text.split("=", 1)[1].split(";", 1)[0].strip().strip('"')
    with MiniRacer() as decoder:
        decoder.eval(hk_js_decode)
        records = decoder.call("d", encoded)
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None).dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate native quote dates")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.sort_values("date").reset_index(drop=True)


def decode_events(
    text: str, kind: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Decode dated share-count or backward adjustment records without executing eval."""
    from akshare.utils import demjson

    if kind == "shares":
        records = demjson.decode(text[text.find("[") : text.rfind("]") + 1])
        value = "outstanding_share"
    elif kind == "adjustments":
        body = text.split("=", 1)[1].strip().rstrip(";")
        records = demjson.decode(body)["data"]
        value = "hfq_factor"
    else:
        raise ValueError("Unknown event kind")
    frame = pd.DataFrame(records)
    if frame.empty or frame.shape[1] != 2:
        raise ValueError(f"Missing or unexpected {kind} records")
    frame.columns = ["date", value]
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame[value] = pd.to_numeric(frame[value], errors="raise")
    if kind == "shares":
        frame[value] *= 10000  # upstream unit is ten thousand shares
    return restrict_events(frame, value, start, end)


def restrict_events(
    frame: pd.DataFrame, value: str, start: str | None, end: str | None
) -> pd.DataFrame:
    """Retain the last pre-window event and all in-window events, checking conflicts."""
    if end is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    if start is not None:
        prior = frame.loc[frame["date"] <= pd.Timestamp(start), "date"]
        if not prior.empty:
            # Older overwritten events cannot affect any observation in this window.
            # Conflicts at the last prior date or inside the window still fail below.
            frame = frame[frame["date"] >= prior.max()]
    if frame.empty or frame["date"].duplicated().any() or frame[value].le(0).any():
        raise ValueError(f"Invalid {value} event records")
    return frame.sort_values("date").reset_index(drop=True)


def decode_share_history(text: str, start: str, end: str) -> pd.DataFrame:
    """Parse the archived dated float-A chart; units are explicitly ten thousand shares."""
    match = re.search(r"<graph\b.*?</graph>", text, flags=re.DOTALL)
    if match is None or "流通A股" not in match.group(0) or "万股" not in text:
        raise ValueError("Historical float-A share chart or unit missing")
    graph = ElementTree.fromstring(match.group(0).replace("&nbsp;", " "))
    events = pd.DataFrame(
        [
            {
                "date": item.attrib["hoverText"],
                "outstanding_share": float(item.attrib["value"]) * 10000,
            }
            for item in graph.findall("set")
        ]
    )
    if events.empty:
        raise ValueError("Empty historical float-A share chart")
    events["date"] = pd.to_datetime(events["date"], errors="raise")
    return restrict_events(events, "outstanding_share", start, end)


def merge_events(prices: pd.DataFrame, events: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Backward as-of join on vendor effective dates; no quotes are filled or created.

    Effective dates are available in this source; historical publication/revision
    timestamps are not. This distinction is recorded as a data limitation.
    """
    dated = events.rename(columns={"date": f"{kind}_effective_date"})
    merged = pd.merge_asof(
        prices.sort_values("date"),
        dated.sort_values(f"{kind}_effective_date"),
        left_on="date",
        right_on=f"{kind}_effective_date",
        direction="backward",
        allow_exact_matches=True,
    )
    if merged[f"{kind}_effective_date"].isna().any():
        raise ValueError(f"No earlier {kind} observation for a native quote")
    if (merged[f"{kind}_effective_date"] > merged["date"]).any():
        raise ValueError("Future event joined to a historical quote")
    return merged
