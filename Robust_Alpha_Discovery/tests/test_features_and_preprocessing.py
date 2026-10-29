from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_alpha.features import compute_features
from robust_alpha.preprocessing import cross_sectional_transform


def test_momentum_and_reversal_formula() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=7),
            "asset": "A",
            "close": np.exp(np.arange(7, dtype=float) / 100.0),
            "high": np.exp(np.arange(7, dtype=float) / 100.0) * 1.01,
            "low": np.exp(np.arange(7, dtype=float) / 100.0) * 0.99,
            "volume": 100.0,
            "amount": 1000.0,
        }
    )
    result = compute_features(frame, ["mom_5", "rev_5"])
    assert result.loc[5, "mom_5"] == pytest.approx(0.05)
    assert result.loc[5, "rev_5"] == pytest.approx(-0.05)


def test_cross_sectional_transform_is_date_local() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4),
            "asset": list("ABCD") * 2,
            "signal": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "eligible": True,
        }
    )
    baseline = cross_sectional_transform(frame, ["signal"])
    changed_input = frame.copy()
    changed_input.loc[changed_input["date"] == "2020-01-02", "signal"] *= 1_000_000
    changed = cross_sectional_transform(changed_input, ["signal"])
    first_date = frame["date"] == "2020-01-01"
    pd.testing.assert_series_equal(
        baseline.loc[first_date, "z__signal"], changed.loc[first_date, "z__signal"]
    )


def test_ineligible_observations_do_not_enter_transform() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 4),
            "asset": list("ABCD"),
            "signal": [1.0, 2.0, 3.0, 1_000_000.0],
            "eligible": [True, True, True, False],
        }
    )
    result = cross_sectional_transform(frame, ["signal"])
    assert pd.isna(result.loc[3, "z__signal"])
    assert result.loc[:2, "z__signal"].mean() == pytest.approx(0.0)

