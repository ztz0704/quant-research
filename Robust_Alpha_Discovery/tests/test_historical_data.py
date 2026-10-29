"""Financial/data integrity boundaries; network-free supplier fixtures are explicitly test-only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_alpha.data.historical import (
    calendar_from_bao,
    construct_panel,
    normalize_bao,
    unique,
)
from robust_alpha.data.history_api import redact
from robust_alpha.data.history_store import Archive, RequestFailed, write_json
from robust_alpha.data.research_contract import ASOF_FIELDS, add_calendar_labels


def fixture_quotes():
    return pd.DataFrame(
        {
            "date": ["2015-01-05", "2015-01-06", "2015-01-07"],
            "code": ["sh.600000"] * 3,
            "open": ["10", "10", "11"],
            "high": ["11", "10", "12"],
            "low": ["9", "10", "10"],
            "close": ["10", "10", "11"],
            "preclose": ["10", "10", "10"],
            "volume": ["1000", "0", "2000"],
            "amount": ["10000", "0", "22000"],
            "turn": ["1.5", "0", "2"],
            "tradestatus": ["1", "0", "1"],
            "isST": ["0", "0", "1"],
        }
    )


def test_halt_marks_are_not_quotes_and_units_are_native():
    result = normalize_bao(fixture_quotes())
    assert result.loc[0, "volume"] == 1000
    assert result.loc[0, "amount"] == 10000
    assert result.loc[0, "turnover_rate"] == 0.015
    assert result.loc[1, "suspended"]
    assert result.loc[1, "vendor_halt_price_present"]
    assert pd.isna(result.loc[1, "raw_open"])
    assert result.loc[1, "buyable"] == False  # noqa: E712
    assert pd.isna(result.loc[0, "buyable"])
    assert result.loc[2, "is_st"]


def test_conflicting_duplicates_are_not_silently_deduped():
    raw = fixture_quotes()
    assert len(unique(pd.concat([raw, raw]), ["date", "code"])) == 3
    conflict = raw.iloc[[0]].assign(close="12")
    with pytest.raises(ValueError, match="Conflicting"):
        unique(pd.concat([raw, conflict]), ["date", "code"])


def test_missing_calendar_days_block_instead_of_skipping():
    raw = pd.DataFrame(
        {"calendar_date": ["2015-01-05", "2015-01-07"], "is_trading_day": ["1", "1"]}
    )
    with pytest.raises(ValueError, match="truncated"):
        calendar_from_bao(raw, "2015-01-05", "2015-01-07")


def test_dynamic_lifetime_and_factor_history_never_backfill():
    calendar = pd.DataFrame({"date": pd.date_range("2015-01-05", periods=4), "exchange": "SSE"})
    lifecycle = pd.DataFrame(
        {
            "asset": ["sh600000"],
            "code": ["sh.600000"],
            "code_name": ["fixture"],
            "list_date": [pd.Timestamp("2015-01-06")],
            "delist_date": [pd.Timestamp("2015-01-08")],
            "listing_status": ["0"],
        }
    )
    factors = pd.DataFrame(
        {"code": ["sh.600000"], "dividOperateDate": ["2015-01-07"], "backAdjustFactor": ["1.2"]}
    )
    panel = construct_panel(fixture_quotes(), calendar, lifecycle, factors)
    assert panel.universe_member.tolist() == [False, True, True, False]
    assert panel.adj_factor.iloc[:2].isna().all()
    assert panel.adj_factor.iloc[2] == 1.2
    assert np.isnan(panel.raw_open.iloc[3])
    assert panel[ASOF_FIELDS].isna().all().all()
    assert not add_calendar_labels(panel, calendar).target_valid.any()


def test_unknown_vendor_state_rejected():
    raw = fixture_quotes()
    raw.loc[0, "isST"] = "maybe"
    with pytest.raises(ValueError, match="Unknown"):
        normalize_bao(raw)


def test_token_redaction(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "fixture_secret_never_real")
    assert "fixture_secret_never_real" not in redact("error fixture_secret_never_real")


def fake_worker(monkeypatch, responses):
    def execute(command, **kwargs):
        result = responses.pop(0)
        write_json(Path(command[-1]), result)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("robust_alpha.data.history_store.subprocess.run", execute)
    monkeypatch.setattr("robust_alpha.data.history_store.time.sleep", lambda _: None)


def test_retry_resume_and_immutable_raw(monkeypatch, tmp_path):
    ok = {
        "status": "ok",
        "columns": ["date", "code"],
        "data": [["2015-01-05", "sh.600000"]],
        "sdk_version": "fixture",
        "downloaded_at": "2026-10-13T00:00:00+00:00",
    }
    failure = {
        "status": "failed",
        "error_type": "TimeoutError",
        "error": "fixture timeout",
        "downloaded_at": "2026-10-13T00:00:00+00:00",
    }
    fake_worker(monkeypatch, [failure, ok])
    archive = Archive(tmp_path, retries=2, pause=0)
    first = archive.get("baostock", "quotes", code="sh.600000")
    pd.testing.assert_frame_equal(first, archive.get("baostock", "quotes", code="sh.600000"))
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 2
    assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 2
    state = next(iter(archive.state.values()))
    (tmp_path / state["raw"]).write_text("tampered")
    with pytest.raises(ValueError, match="tampered"):
        archive.get("baostock", "quotes", code="sh.600000")


def test_final_failures_are_checkpointed(monkeypatch, tmp_path):
    fake_worker(
        monkeypatch,
        [
            {
                "status": "failed",
                "error_type": "ConnectionError",
                "error": "fixture network",
                "downloaded_at": "2026-10-13",
            }
        ],
    )
    archive = Archive(tmp_path, retries=1, pause=0)
    with pytest.raises(RequestFailed):
        archive.get("baostock", "calendar", start_date="2015-01-01")
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert next(iter(state.values()))["status"] == "failed"
