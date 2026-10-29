from __future__ import annotations

import pandas as pd

from robust_alpha.validation.walk_forward import (
    attach_walk_forward_score,
    walk_forward_ic_selection,
)


def test_walk_forward_selects_using_train_only() -> None:
    rows = []
    for year in (2018, 2019, 2020):
        for date in pd.bdate_range(f"{year}-01-01", periods=120):
            rows.extend(
                [
                    {"date": date, "signal": "stable", "ic": 0.03},
                    {"date": date, "signal": "reverses", "ic": 0.05 if year < 2020 else -0.20},
                    {"date": date, "signal": "weak", "ic": 0.01},
                ]
            )
    result = walk_forward_ic_selection(
        pd.DataFrame(rows), train_years=2, top_k=1, min_train_dates=100
    )
    selection = result["selections"].iloc[0]
    assert selection["test_year"] == 2020
    assert selection["signal"] == "reverses"
    assert selection["train_mean_ic"] > 0
    assert selection["test_mean_ic"] < 0


def test_walk_forward_score_uses_only_that_years_preselected_signals() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-02"]),
            "asset": ["A", "A", "B"],
            "z__first": [100.0, 1.0, 3.0],
            "z__second": [100.0, 3.0, 5.0],
            "z__unused": [-100.0, -100.0, -100.0],
        }
    )
    selections = pd.DataFrame(
        {"test_year": [2020, 2020], "signal": ["first", "second"]}
    )
    result = attach_walk_forward_score(panel, selections)
    assert pd.isna(result.loc[0, "walk_forward_score"])
    assert result.loc[1, "walk_forward_score"] == 2.0
    assert result.loc[2, "walk_forward_score"] == 4.0
