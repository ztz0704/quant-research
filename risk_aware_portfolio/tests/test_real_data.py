"""Offline provider gates and causal feature tests; no network required by pytest."""

import numpy as np
import pandas as pd
import pytest

from risk_aware_portfolio.backtest.engine import run_backtest
from risk_aware_portfolio.data.ashare import build_real_panel, coverage_audit, enforce_coverage
from risk_aware_portfolio.data.sina import merge_events
from risk_aware_portfolio.utils.config import variant


@pytest.fixture
def quotes_and_config(config):
    rng = np.random.default_rng(121)
    dates = pd.bdate_range("2020-01-02", periods=75)
    assets = [f"sh{600000 + i}" for i in range(20)]
    rows = []
    for i, asset in enumerate(assets):
        price = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        shares = np.exp(rng.normal(20, 0.1, len(dates)))
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "asset": asset,
                    "close": price,
                    "volume": rng.uniform(1e5, 1e7, len(dates)),
                    "outstanding_share": shares,
                    "hfq_factor": 1.0,
                    "shares_effective_date": dates,
                    "adjustments_effective_date": dates,
                }
            )
        )
    universe = pd.DataFrame({"asset": assets, "industry": [f"industry_{i % 2}" for i in range(20)]})
    config["real_data"] = {}
    settings = variant(
        config,
        real_data={
            "momentum_lookback": 12,
            "momentum_skip": 3,
            "volatility_window": 5,
            "liquidity_window": 3,
            "reversal_window": 2,
            "execution_delay_sessions": 1,
        },
        risk_model={
            "industries": ["industry_0", "industry_1"],
            "styles": ["size", "momentum", "volatility", "liquidity"],
            "min_observations": 10,
            "lookback": 20,
        },
        portfolio={"max_weight": 0.1},
    )
    return pd.concat(rows, ignore_index=True), universe, settings


def test_native_missing_quote_fails_without_dropping_stock(quotes_and_config):
    quotes, universe, _ = quotes_and_config
    calendar = pd.DatetimeIndex(quotes["date"].unique()).sort_values()
    audit = coverage_audit(quotes.drop(index=3), calendar, universe["asset"])
    assert audit["missing_sessions"].sum() == 1
    with pytest.raises(ValueError, match="no stocks were removed"):
        enforce_coverage(audit)


def test_zero_volume_fails_tradability_gate(quotes_and_config):
    quotes, universe, _ = quotes_and_config
    quotes.loc[3, "volume"] = 0
    calendar = pd.DatetimeIndex(quotes["date"].unique()).sort_values()
    with pytest.raises(ValueError, match="tradability gate"):
        enforce_coverage(coverage_audit(quotes, calendar, universe["asset"]))


def test_share_events_cannot_use_future_value():
    quotes = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"])})
    events = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01", "2020-01-03"]), "outstanding_share": [100, 200]}
    )
    merged = merge_events(quotes, events, "shares")
    assert merged["outstanding_share"].tolist() == [100, 200]
    assert (merged["shares_effective_date"] <= merged["date"]).all()
    with pytest.raises(ValueError, match="No earlier"):
        merge_events(quotes, events.iloc[1:], "shares")


def test_real_features_have_explicit_one_session_lag(quotes_and_config):
    quotes, universe, settings = quotes_and_config
    panel = build_real_panel(quotes, universe, settings)
    asset = universe["asset"].iloc[0]
    section = quotes[quotes["asset"].eq(asset)].set_index("date")
    expected = -np.log(section["close"]).diff(2).shift(1)
    actual = panel[panel["asset"].eq(asset)].set_index("date")
    np.testing.assert_allclose(actual["alpha"], expected.loc[actual.index])
    np.testing.assert_allclose(actual["return"], section["close"].pct_change().loc[actual.index])
    assert (actual["feature_asof"] < actual.index).all()


def test_future_native_prices_do_not_change_past_features(quotes_and_config):
    quotes, universe, settings = quotes_and_config
    cutoff = pd.Timestamp(quotes["date"].unique()[45])
    reference = build_real_panel(quotes, universe, settings)
    changed = quotes.copy()
    changed.loc[changed["date"] > cutoff, ["close", "outstanding_share", "hfq_factor"]] *= 1.2
    changed = build_real_panel(changed, universe, settings)
    pd.testing.assert_frame_equal(
        reference[reference["date"] <= cutoff], changed[changed["date"] <= cutoff]
    )


def test_future_native_prices_do_not_change_past_weights(quotes_and_config):
    quotes, universe, settings = quotes_and_config
    cutoff = pd.Timestamp(quotes["date"].unique()[45])
    reference = run_backtest(build_real_panel(quotes, universe, settings), settings)
    quotes.loc[quotes["date"] > cutoff, "close"] *= 1.2
    changed = run_backtest(build_real_panel(quotes, universe, settings), settings)
    np.testing.assert_array_equal(reference.weights.loc[:cutoff], changed.weights.loc[:cutoff])


def test_evaluation_start_preserves_warmup_history(synthetic, config):
    cutoff = pd.Timestamp(synthetic.panel["date"].unique()[30])
    settings = variant(config, backtest={"evaluation_start": str(cutoff.date())})
    result = run_backtest(synthetic.panel, settings)
    assert result.weights.index[0] == cutoff
    assert result.returns["decision_date"].min() == cutoff
    assert (result.returns["return_date"] > result.returns["decision_date"]).all()
