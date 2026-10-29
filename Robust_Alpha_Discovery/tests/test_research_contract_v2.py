import numpy as np
import pandas as pd
import pytest

from robust_alpha.data.research_adapters import normalize_tushare_daily, read_bundle
from robust_alpha.data.research_contract import (
    ASOF_FIELDS,
    add_calendar_labels,
    add_decision_eligibility,
    normalize_dynamic_panel,
    quality_report,
    require_formal_quality,
    require_frozen,
)
from robust_alpha.data.research_fixture import engineering_sample
from robust_alpha.evaluation.conditional_reversal import (
    CONTROLS,
    TERMS,
    build_pressure_features,
    calendar_hac,
    conditional_research,
)


def small_panel():
    dates = pd.bdate_range("2020-01-02", periods=8)
    calendar = pd.DataFrame({"date": dates, "exchange": "SSE"})
    rows = []
    for asset, used in (("OLD", dates), ("IPO", dates[3:]), ("DELIST", dates[:5])):
        for position, date in enumerate(used):
            price = 10 + position
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "exchange": "SSE",
                    "raw_open": price,
                    "raw_high": price + 1,
                    "raw_low": price - 1,
                    "raw_close": price,
                    "volume": 100.0,
                    "amount": 1000.0,
                    "adj_factor": 1.0,
                    "universe_member": True,
                    "is_st": False,
                    "suspended": False,
                    "buyable": True,
                    "sellable": True,
                    "list_date": used[0],
                    "delist_date": pd.NaT,
                    **{f: date + pd.Timedelta(hours=18) for f in ASOF_FIELDS},
                }
            )
    return pd.DataFrame(rows), calendar


def test_ipo_delist_sparse_quote_keep_market_dates_and_never_jump():
    raw, calendar = small_panel()
    missing_date = calendar.date.iloc[2]
    raw = raw.loc[~(raw.asset.eq("OLD") & raw.date.eq(missing_date))]
    panel = add_decision_eligibility(normalize_dynamic_panel(raw, calendar), 2)
    label = add_calendar_labels(panel, calendar).set_index(["date", "asset"])
    row = label.loc[(calendar.date.iloc[0], "OLD")]
    assert row.label_end == missing_date
    assert pd.isna(row.target)  # OLD resumes on the next date; cannot jump to it.
    assert label.loc[(missing_date, "OLD"), "supplied_row"] == False  # noqa: E712
    assert not label.loc[(calendar.date.iloc[2], "IPO"), "eligible"]
    assert not label.loc[(calendar.date.iloc[3], "IPO"), "eligible"]
    assert label.loc[(calendar.date.iloc[4], "IPO"), "eligible"]
    assert pd.isna(label.loc[(calendar.date.iloc[4], "DELIST"), "target"])
    assert label.loc[(calendar.date.iloc[1], "OLD"), "eligible"]


def test_future_quotes_or_trade_states_do_not_change_decision_eligibility():
    raw, calendar = small_panel()
    before = add_decision_eligibility(normalize_dynamic_panel(raw, calendar), 1)
    cutoff = calendar.date.iloc[3]
    raw.loc[raw.date.gt(cutoff), ["raw_open", "raw_high", "raw_low", "raw_close"]] *= 10
    raw.loc[raw.date.gt(cutoff), "buyable"] = False
    after = add_decision_eligibility(normalize_dynamic_panel(raw, calendar), 1)
    pd.testing.assert_series_equal(
        before.loc[before.date.le(cutoff), "eligible"], after.loc[after.date.le(cutoff), "eligible"]
    )


def test_labels_have_raw_price_simple_unit_and_eod_maturity():
    raw, calendar = small_panel()
    result = add_calendar_labels(normalize_dynamic_panel(raw, calendar), calendar)
    row = result.loc[result.asset.eq("OLD")].iloc[0]
    assert row.entry_raw_open == 11
    assert row.exit_raw_open == 12
    assert row.target == pytest.approx(12 / 11 - 1)
    assert row.label_available_at == calendar.date.iloc[2] + pd.Timedelta(hours=18)
    assert row.return_unit == "decimal_simple"


@pytest.mark.parametrize(
    "failure", ["future", "boolean", "duplicate", "exchange", "calendar", "ohlc"]
)
def test_contract_rejects_ambiguous_inputs(failure):
    raw, calendar = small_panel()
    if failure == "future":
        raw.loc[0, "quote_asof"] += pd.Timedelta(days=1)
    elif failure == "boolean":
        raw["is_st"] = "False"
    elif failure == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]])
    elif failure == "exchange":
        raw.loc[0, "exchange"] = "SZSE"
    elif failure == "calendar":
        calendar = calendar.iloc[1:]
    else:
        raw.loc[0, "raw_low"] = 100
    with pytest.raises(ValueError):
        normalize_dynamic_panel(raw, calendar)


def test_missing_formal_fields_and_synthetic_metadata_fail_fast():
    raw, calendar, metadata = engineering_sample(days=130, assets=10)
    panel = normalize_dynamic_panel(raw, calendar)
    report = quality_report(panel, metadata)
    assert not report["formal_ready"]
    with pytest.raises(ValueError, match="Formal PIT"):
        require_formal_quality(report)


def test_frozen_protocol_and_holdout_gates():
    cfg = {
        "phase": "holdout",
        "protocol": {"status": "frozen", "sha256": "x", "holdout_confirmed_untouched": False},
    }
    with pytest.raises(ValueError, match="sealed"):
        require_frozen(cfg, "x", True)
    with pytest.raises(ValueError, match="hash"):
        require_frozen(cfg, "changed", True)
    cfg["protocol"]["status"] = "draft"
    with pytest.raises(ValueError, match="frozen"):
        require_frozen(cfg, "x", False)


def test_pressure_features_and_date_neutralization_ignore_future():
    raw, calendar, _ = engineering_sample(days=180, assets=30)

    def features(frame):
        panel = add_decision_eligibility(normalize_dynamic_panel(frame, calendar), 1)
        return build_pressure_features(panel, 5)[0]

    before = features(raw)
    cutoff = calendar.date.sort_values().iloc[120]
    mutated = raw.copy()
    mutated.loc[mutated.date.gt(cutoff), "volume"] *= 1000
    mutated.loc[mutated.date.gt(cutoff), ["raw_open", "raw_high", "raw_low", "raw_close"]] *= 5
    after = features(mutated)
    pd.testing.assert_frame_equal(
        before.loc[before.date.le(cutoff)], after.loc[after.date.le(cutoff)]
    )


def test_hac_preserves_missing_calendar_sessions():
    sample = pd.Series(np.tile([1.0, -1.0], 20))
    sample.iloc[5:10] = np.nan
    report = calendar_hac(sample)
    assert report["missing_dates"] == 5
    assert report["n_dates"] == 35


def test_tushare_units_and_bundle_hash_rejection(tmp_path):
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20200102"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.0],
            "vol": [2.0],
            "amount": [3.0],
        }
    )
    converted = normalize_tushare_daily(frame)
    assert converted.volume.iloc[0] == 200
    assert converted.amount.iloc[0] == 3000
    assert converted.exchange.iloc[0] == "SZSE"
    converted.to_parquet(tmp_path / "panel.parquet")
    (tmp_path / "metadata.json").write_text('{"files":{"panel.parquet":"wrong"}}')
    with pytest.raises(ValueError, match="identity"):
        read_bundle(tmp_path)


def test_explicit_canonical_filename_keeps_hash_and_preread_boundary(tmp_path):
    import json

    from robust_alpha.data.research_contract import digest_file

    raw = pd.DataFrame(
        {"date": pd.to_datetime(["2015-01-05", "2025-12-31"]), "asset": ["sh600000", "sh600000"]}
    )
    calendar = raw[["date"]].assign(exchange="SSE")
    raw.to_parquet(tmp_path / "canonical_market_panel.parquet", index=False)
    calendar.to_parquet(tmp_path / "calendar.parquet", index=False)
    metadata = {
        "panel_file": "canonical_market_panel.parquet",
        "files": {p.name: digest_file(p) for p in tmp_path.glob("*.parquet")},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    loaded, sessions, _ = read_bundle(tmp_path, "2015-12-31")
    assert len(loaded) == len(sessions) == 1
    assert not (tmp_path / "panel.parquet").exists()
    metadata["panel_file"] = "../canonical_market_panel.parquet"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="inside"):
        read_bundle(tmp_path)


def test_joint_regression_recovers_interaction_and_uses_dates_for_inference():
    rng = np.random.default_rng(12)
    rows, truths = [], []
    dates = pd.bdate_range("2020-01-01", periods=15)
    for date in dates:
        x = rng.normal(size=(40, 7))
        beta = np.array([0.01, 0.002, 0.005]) + rng.normal(0, 0.0001, 3)
        truths.append(beta)
        for i in range(40):
            row = dict(zip(TERMS + CONTROLS, x[i], strict=True))
            row.update(
                date=date,
                asset=f"SYN{i}",
                target=float(
                    x[i, :3] @ beta
                    + x[i, 3:] @ np.array([0.001, 0.003, -0.002, 0.001])
                    + 0.002 * (i % 2)
                ),
                industry=f"G{i % 2}",
                eligible=True,
                target_valid=True,
                liquidity_group="mid",
                volatility_group="mid",
            )
            rows.append(row)
    results = conditional_research(pd.DataFrame(rows), min_assets=20)
    daily = results["cross_sectional_regression_daily"]
    np.testing.assert_allclose(daily[TERMS], truths, atol=1e-14)
    assert results["factor_hac_summary"].n_dates.eq(len(dates)).all()
