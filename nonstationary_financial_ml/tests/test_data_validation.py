"""Schema failures and controlled nonstationarity checks."""
import numpy as np
import pytest

from financial_ml.data import synthetic, validate


@pytest.fixture
def panel():
    return synthetic(days=30, assets=8)


def test_deterministic_and_nonstationary():
    a = synthetic(days=80, assets=20)
    assert a.equals(synthetic(days=80, assets=20))
    assert a.synthetic_regime.nunique() == 2
    assert synthetic(days=80, assets=20, stationary=True).synthetic_regime.eq(0).all()


@pytest.mark.parametrize("failure", ["duplicate", "missing", "inf", "return", "cap", "ordering", "grid"])
def test_invalid_panel_fails(panel, failure):
    bad = panel.copy()
    if failure == "duplicate":
        bad.loc[1] = bad.loc[0]
    elif failure == "missing":
        bad = bad.drop(columns="industry")
    elif failure == "inf":
        bad.loc[0, "x"] = np.inf
    elif failure == "return":
        bad.loc[0, "return"] = -1.1
    elif failure == "cap":
        bad.loc[0, "market_cap"] = -1
    elif failure == "ordering":
        bad = bad.iloc[::-1]
    elif failure == "grid":
        bad = bad.iloc[1:]
    with pytest.raises(ValueError):
        validate(bad)
