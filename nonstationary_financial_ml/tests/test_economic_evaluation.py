import numpy as np
import pandas as pd

from financial_ml.portfolio import half_l1, performance, portfolio_daily, weights


def section():
    return pd.DataFrame({"date": pd.Timestamp("2023-01-03"), "asset": ["A", "B", "C", "D"],
                         "prediction": [4., 3., 2., 1.], "target": [.10, .02, -.02, -.10], "model": "linear"})


def test_selection_never_uses_future_target():
    data = section()
    before = weights(data, "prediction", "long_only", 2)
    data.loc[0, "target"] = np.nan
    assert before.equals(weights(data, "prediction", "long_only", 2))
    assert np.isclose(half_l1(before, pd.Series(dtype=float)), .5)


def test_drifted_turnover_and_cost_hand_calculation():
    first = section()
    second = first.copy()
    second["date"] += pd.Timedelta(days=1)
    cfg = {"labels": {"forward_horizon": 1}, "portfolio": {"quantiles": 2, "cost_bps": [10]}}
    result = portfolio_daily(pd.concat([first, second]), "prediction", "long_only", cfg)
    assert np.isclose(result.gross_return.iloc[0], .06)
    assert np.isclose(result.net_return.iloc[0], .06 - .0005)
    # Previous top-two positions drift from 50/50 to 55/106 and 51/106.
    expected = abs(.5 - .55 / 1.06)
    assert np.isclose(result.turnover.iloc[1], expected)


def test_missing_active_return_invalidates_performance():
    data = section()
    data.loc[0, "target"] = np.nan
    cfg = {"labels": {"forward_horizon": 1}, "portfolio": {"quantiles": 2, "cost_bps": [0]}}
    result = portfolio_daily(data, "prediction", "long_only", cfg)
    assert result.net_return.isna().all()
    assert np.isnan(performance(result)["sharpe"])


def test_constant_predictions_not_arbitrary_stock_picks():
    data = section()
    data["prediction"] = 1.
    assert weights(data, "prediction", "long_only", 2).eq(.25).all()
    assert weights(data, "prediction", "long_short", 2).eq(0).all()
