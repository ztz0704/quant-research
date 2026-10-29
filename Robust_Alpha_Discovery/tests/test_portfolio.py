from __future__ import annotations

import pandas as pd
import pytest

from robust_alpha.pipeline import _build_walk_forward_portfolio_daily
from robust_alpha.portfolio.costs import apply_linear_costs
from robust_alpha.portfolio.turnover import calculate_turnover
from robust_alpha.portfolio.weights import quantile_weights


def test_quantile_long_short_has_unit_long_and_short_legs() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 10),
            "asset": [f"A{i}" for i in range(10)],
            "signal": range(10),
            "eligible": True,
        }
    )
    weights = quantile_weights(frame, "signal", quantile=0.2, kind="long_short")
    assert weights["weight"].clip(lower=0).sum() == pytest.approx(1.0)
    assert weights["weight"].clip(upper=0).sum() == pytest.approx(-1.0)
    assert weights["weight"].sum() == pytest.approx(0.0)


def test_quantile_weights_leave_too_small_cross_section_in_cash() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 3),
            "asset": ["A", "B", "C"],
            "signal": [1.0, 2.0, 3.0],
            "eligible": True,
        }
    )
    weights = quantile_weights(
        frame,
        "signal",
        quantile=0.2,
        kind="long_short",
        min_assets=4,
    )
    assert weights["weight"].eq(0.0).all()


def test_single_name_long_short_does_not_overlap_legs() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "asset": ["A"],
            "signal": [1.0],
            "eligible": True,
        }
    )
    weights = quantile_weights(frame, "signal", quantile=0.5, kind="long_short")
    assert weights["weight"].eq(0.0).all()


def test_turnover_uses_post_return_drift_and_costs() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    weights = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "asset": ["A", "B", "A", "B"],
            "weight": [1.0, -1.0, 1.0, -1.0],
        }
    )
    panel = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "asset": ["A", "B", "A", "B"],
            "target_o2o": 0.0,
        }
    )
    turnover = calculate_turnover(weights, panel)
    assert turnover.loc[0, "turnover"] == pytest.approx(1.0)
    assert turnover.loc[1, "turnover"] == pytest.approx(0.0)

    portfolio = pd.DataFrame({"date": dates, "gross_return": [0.01, 0.02]})
    scenarios = apply_linear_costs(portfolio, turnover, cost_bps=[10])
    assert scenarios.loc[0, "net_return"] == pytest.approx(0.009)
    assert scenarios.loc[1, "net_return"] == pytest.approx(0.02)


def test_future_tradeability_cannot_change_signal_date_weights() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4),
            "asset": ["A", "B", "C", "D"],
            "signal": [1.0, 2.0, 3.0, 4.0],
            "eligible": True,
            "target_valid": [True, False, True, False],
        }
    )
    first = quantile_weights(frame, "signal", quantile=0.25, kind="long_short")
    mutated = frame.copy()
    mutated["target_valid"] = ~mutated["target_valid"]
    second = quantile_weights(mutated, "signal", quantile=0.25, kind="long_short")
    pd.testing.assert_series_equal(first["weight"], second["weight"])


def test_walk_forward_turnover_resets_across_test_years() -> None:
    dates = pd.to_datetime(["2020-01-02", "2022-01-03"])
    panel = pd.DataFrame(
        {
            "date": dates.repeat(4),
            "asset": ["A", "B", "C", "D"] * 2,
            "eligible": True,
            "walk_forward_score": [1.0, 2.0, 3.0, 4.0] * 2,
            "walk_forward_signal_count": 1,
            "target_o2o": 0.0,
        }
    )
    daily = _build_walk_forward_portfolio_daily(
        panel,
        target_column="target_o2o",
        quantile=0.25,
        min_assets=4,
        cost_bps=[0],
    )
    long_only = daily[daily["portfolio"].eq("long_only")].sort_values("date")
    long_short = daily[daily["portfolio"].eq("long_short")].sort_values("date")
    assert long_only["turnover"].tolist() == pytest.approx([0.5, 0.5])
    assert long_short["turnover"].tolist() == pytest.approx([1.0, 1.0])
