from __future__ import annotations

import pandas as pd

from robust_alpha.data.akshare import (
    complete_market_date_grid,
    download_akshare_panel,
    normalize_akshare_history,
)


def _history(symbol: str, dates: list[str]) -> pd.DataFrame:
    size = len(dates)
    return pd.DataFrame(
        {
            "日期": dates,
            "股票代码": [symbol] * size,
            "开盘": [10.0 + index for index in range(size)],
            "收盘": [10.5 + index for index in range(size)],
            "最高": [11.0 + index for index in range(size)],
            "最低": [9.5 + index for index in range(size)],
            "成交量": [100.0] * size,
            "成交额": [1000.0] * size,
        }
    )


def test_normalize_akshare_history_maps_canonical_columns() -> None:
    result = normalize_akshare_history(_history("000001", ["2020-01-02"]), "000001")
    assert result.loc[0, "asset"] == "000001"
    assert result.loc[0, "close"] == 10.5
    assert result.loc[0, "tradable"]


def test_complete_market_grid_marks_missing_quote_non_tradable() -> None:
    first = normalize_akshare_history(
        _history("000001", ["2020-01-02", "2020-01-06"]), "000001"
    )
    second = normalize_akshare_history(
        _history("600000", ["2020-01-02", "2020-01-03", "2020-01-06"]), "600000"
    )
    panel = complete_market_date_grid(pd.concat([first, second], ignore_index=True))
    missing = panel[(panel["asset"] == "000001") & (panel["date"] == "2020-01-03")].iloc[0]
    assert missing["universe_member"]
    assert not missing["tradable"]
    assert missing["close"] == 10.5
    assert missing["volume"] == 0


def test_download_panel_uses_injected_fetcher_without_network() -> None:
    def fetcher(**kwargs: object) -> pd.DataFrame:
        symbol = str(kwargs["symbol"])
        return _history(symbol, ["2020-01-02", "2020-01-03", "2020-01-06"])

    panel = download_akshare_panel(
        ["600000", "000001"],
        start_date="2020-01-02",
        end_date="2020-01-06",
        endpoint="eastmoney",
        pause_seconds=0,
        fetcher=fetcher,
    )
    assert len(panel) == 6
    assert set(panel["asset"]) == {"000001", "600000"}


def test_download_panel_rejects_unsupported_exchange() -> None:
    try:
        download_akshare_panel(
            ["830001"],
            start_date="2020-01-02",
            end_date="2020-01-06",
            pause_seconds=0,
            fetcher=lambda **_: pd.DataFrame(),
        )
    except ValueError as error:
        assert "Shanghai/Shenzhen" in str(error)
    else:  # pragma: no cover - defensive assertion style avoids a pytest dependency here
        raise AssertionError("Expected unsupported exchange to be rejected")
