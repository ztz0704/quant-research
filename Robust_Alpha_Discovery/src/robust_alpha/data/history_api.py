"""Isolated, bounded Python API calls. Never persist credentials or SDK console logs."""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import os
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

TOKEN_NAMES = ("TUSHARE_TOKEN", "TUSHARE_API_TOKEN", "TS_TOKEN", "TUSHARE_PRO_TOKEN")
BAO_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST"


def token() -> str | None:
    return next((os.environ[n] for n in TOKEN_NAMES if os.environ.get(n)), None)


def redact(message: str) -> str:
    for name in TOKEN_NAMES:
        if os.environ.get(name):
            message = message.replace(os.environ[name], "[redacted]")
    return message[:600]


def fetch(source: str, endpoint: str, params: dict) -> pd.DataFrame:
    if source == "baostock":
        import baostock as bs
        from baostock.security import sectorinfo

        login = bs.login()  # Anonymous free service; no paid API key is set.
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login {login.error_code}: {login.error_msg}")
        methods = {
            "calendar": bs.query_trade_dates,
            "universe": bs.query_all_stock,
            "basic": bs.query_stock_basic,
            "quotes": bs.query_history_k_data_plus,
            "adjustment": bs.query_adjust_factor,
            "dividends": bs.query_dividend_data,
            "industry": bs.query_stock_industry,
            "hs300": bs.query_hs300_stocks,
            "terminated": sectorinfo.query_terminated_stocks,
            "suspended": sectorinfo.query_suspended_stocks,
            "st": sectorinfo.query_st_stocks,
        }
        rs = methods[endpoint](**params)
        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock query {rs.error_code}: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        # ResultData.next can fail halfway through pagination.
        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock pagination {rs.error_code}: {rs.error_msg}")
        result = pd.DataFrame(rows, columns=rs.fields)
        bs.logout()
        return result
    if source == "akshare":
        import akshare as ak

        methods = {
            "eastmoney": ak.stock_zh_a_hist,
            "tencent": ak.stock_zh_a_hist_tx,
            "calendar": ak.tool_trade_date_hist_sina,
            "delisted_sz": ak.stock_info_sz_delist,
            "delisted_sh": ak.stock_info_sh_delist,
            "share_changes": ak.stock_share_change_cninfo,
        }
        return methods[endpoint](**params)
    if source == "tushare":
        # Official HTTP Python interface avoids installing an unused SDK when no token exists.
        import requests

        credential = token()
        if not credential:
            raise RuntimeError("No configured environment token")
        response = requests.post(
            "https://api.tushare.pro",
            json={"api_name": endpoint, "token": credential, "params": params, "fields": ""},
            timeout=25,
        )
        response.raise_for_status()
        body = response.json()
        if body["code"] != 0:
            raise RuntimeError(f"Tushare code {body['code']}: {redact(body.get('msg', ''))}")
        return pd.DataFrame(body["data"]["items"], columns=body["data"]["fields"])
    raise ValueError(f"Unknown source {source}")


def worker(request_path: Path, response_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["direct"]:
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            os.environ.pop(name, None)
        os.environ["NO_PROXY"] = "*"
    socket.setdefaulttimeout(45)
    result = {"source": request["source"], "endpoint": request["endpoint"]}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            frame = fetch(request["source"], request["endpoint"], request["params"])
        result.update(
            status="ok",
            columns=list(frame),
            data=json.loads(frame.to_json(orient="values", date_format="iso", force_ascii=False)),
            sdk_version=importlib.metadata.version(
                "requests" if request["source"] == "tushare" else request["source"]
            ),
        )
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=redact(str(exc)))
    result["downloaded_at"] = datetime.now(UTC).isoformat()
    response_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    worker(Path(sys.argv[1]), Path(sys.argv[2]))
