"""Causal expression and frozen-selection invariants, separate from empirical results."""

import numpy as np
import pandas as pd
import pytest

from robust_alpha.features import compute_features
from robust_alpha.features.expressions import evaluate_expression, parse_expression
from robust_alpha.mining import period_panel, score, screen_candidates


def history():
    dates = pd.bdate_range("2020-01-01", periods=40)
    pieces = []
    for asset, offset in (("000001", 0), ("600000", 1)):
        price = pd.Series(np.arange(10 + offset, 50 + offset, dtype=float))
        pieces.append(
            pd.DataFrame(
                {
                    "asset": asset,
                    "date": dates,
                    "open": price,
                    "close": price,
                    "high": price + 1,
                    "low": price - 1,
                    "volume": 1000.0,
                    "amount": 10000.0,
                    "eligible": True,
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


@pytest.mark.parametrize(
    "expression",
    [
        "delay(close, -1)",
        "ts_mean(close, 0)",
        "target_o2o",
        "exit_open",
        "close.shift(-1)",
        "__import__('os')",
        "ts_mean(close, 2 + 3)",
        "open[0]",
    ],
)
def test_expressions_reject_future_and_arbitrary_python(expression):
    with pytest.raises((ValueError, SyntaxError)):
        parse_expression(expression)


def test_custom_factor_matches_hand_computation_and_future_invariance():
    panel = history()
    expression = "-ts_mean(ret_1, 3) / (ts_std(ret_1, 5) + 0.000001)"
    actual = evaluate_expression(panel, expression)
    first = panel.loc[panel.asset.eq("000001"), "close"].pct_change(fill_method=None)
    expected = -first.rolling(3).mean() / (first.rolling(5).std() + 0.000001)
    pd.testing.assert_series_equal(actual.loc[first.index], expected, check_names=False)
    cutoff = panel.date.unique()[20]
    changed = panel.copy()
    changed.loc[changed.date > cutoff, "close"] *= 100
    changed.loc[changed.date > cutoff, "amount"] = 1
    after = evaluate_expression(changed, expression)
    pd.testing.assert_series_equal(
        actual.loc[panel.date <= cutoff], after.loc[panel.date <= cutoff]
    )


def test_custom_factor_dispatch_does_not_expose_label_columns():
    panel = history()
    panel["exit_open"] = 1e20
    definitions = {"custom_rev": {"expression": "-ret_1", "direction": -1}}
    calculated = compute_features(panel, ["custom_rev"], definitions)
    expected = calculated.groupby("asset")["close"].pct_change(fill_method=None)
    pd.testing.assert_series_equal(calculated.custom_rev, expected, check_names=False)


def test_period_reserves_exit_dates_without_reading_future_tradeability():
    panel = history()
    dates = [str(panel.date.min().date()), str(panel.date.unique()[10].date())]
    before = period_panel(panel, dates)
    changed = panel.copy()
    changed["entry_tradable"] = False
    after = period_panel(changed, dates)
    assert before.date.max() == panel.date.unique()[8]
    pd.testing.assert_frame_equal(before, after.drop(columns="entry_tradable"))


def test_candidate_gates_never_force_a_winner_and_deduplicate_on_training_only():
    summary = pd.DataFrame(
        {
            "signal": ["a", "b", "c"],
            "stage": "train",
            "n_dates": 150,
            "mean_ic": [0.04, 0.03, -0.01],
            "holm_p": [0.01, 0.02, 0.9],
            "active_net_mean": [0.001, 0.001, -0.001],
            "active_return_coverage": 1.0,
        }
    )
    validation = summary.copy()
    validation["stage"] = "validation"
    correlations = pd.DataFrame(
        [[1, 0.95, 0], [0.95, 1, 0], [0, 0, 1]], index=["a", "b", "c"], columns=["a", "b", "c"]
    )
    cfg = {
        "mining": {
            "min_train_dates": 100,
            "min_validation_dates": 100,
            "p_threshold": 0.05,
            "min_return_coverage": 1.0,
            "max_correlation": 0.8,
            "max_selected": 5,
        }
    }
    screened = screen_candidates(summary, validation, correlations, cfg)
    assert screened.loc[screened.accepted, "signal"].tolist() == ["a"]
    assert "correlation" in screened.loc[screened.signal.eq("b"), "reason"].iloc[0]
    validation.loc[:, "holm_p"] = 1.0
    assert not screen_candidates(summary, validation, correlations, cfg).accepted.any()


def test_csv_loader_preserves_leading_zero_stock_codes(tmp_path):
    from robust_alpha.data.loader import load_panel

    panel = history()
    panel["universe_member"] = True
    panel["tradable"] = True
    source = tmp_path / "quotes.csv"
    panel.to_csv(source, index=False)
    assert set(load_panel(source).asset.unique()) == {"000001", "600000"}


def test_old_scores_cannot_be_exported_as_current_and_no_selection_cannot_trade(
    monkeypatch, tmp_path
):
    import robust_alpha.mining as mining

    monkeypatch.setattr(mining, "settings", lambda _: ({}, tmp_path))
    monkeypatch.setattr(
        mining, "read_selection", lambda *_: {"history_through": "2019-12-31", "selected": []}
    )
    with pytest.raises(ValueError, match="old data refused"):
        score("unused", "2020-12-31", expected_last_session="2020-12-31")
    with pytest.raises(ValueError, match="No candidate passed"):
        score("unused", "2020-12-31", historical=True)


def test_successful_historical_score_preserves_unavailable_rows_and_uses_no_labels(
    monkeypatch, tmp_path
):
    import robust_alpha.mining as mining

    date = pd.Timestamp("2020-02-20")
    current = pd.DataFrame(
        {
            "date": date,
            "asset": ["000001", "000002", "600000", "600001"],
            "eligible": [True, True, True, False],
            "z__a": [1.0, 0.5, -1.0, 0.0],
        }
    )
    cfg = {"data": {"path": "unused.parquet"}, "evaluation": {"min_assets_per_date": 3}}
    monkeypatch.setattr(mining, "settings", lambda _: (cfg, tmp_path))
    monkeypatch.setattr(
        mining, "read_selection", lambda *_: {"history_through": "2020-01-01", "selected": ["a"]}
    )
    calls = []

    def prepare(_, end, *, labels):
        calls.append((end, labels))
        return current, "history"

    monkeypatch.setattr(mining, "prepare", prepare)
    monkeypatch.setattr(mining, "file_hash", lambda _: "fingerprint")
    filename = score("unused", str(date.date()), historical=True)
    result = pd.read_csv(filename, dtype={"asset": "string"})
    assert calls == [(str(date.date()), False)]
    assert result.signal_available.sum() == 3
    assert result.loc[result.asset.eq("600001"), "alpha"].isna().all()
    assert result.loc[result.asset.eq("000001"), "alpha"].iloc[0] == 1.0


def test_zero_volume_window_is_unavailable_instead_of_infinite():
    from robust_alpha.features.liquidity import volume_shock

    volume = pd.Series([100.0, 100.0, 0.0, 0.0, 0.0])
    result = volume_shock(volume, pd.Series("000001", index=volume.index), short=2, long=4)
    assert result.iloc[-2:].isna().all()
    assert not np.isinf(result).any()
