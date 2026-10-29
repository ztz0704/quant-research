import pandas as pd

from financial_ml.data import synthetic
from financial_ml.regimes import regime_dates


def test_regime_thresholds_and_history_ignore_future():
    panel = synthetic(days=100, assets=20)
    dates = sorted(panel.date.unique())
    before, thresholds = regime_dates(panel, dates[40])
    changed = panel.copy()
    changed.loc[changed.date > dates[70], "return"] = .5
    after, changed_thresholds = regime_dates(changed, dates[40])
    assert thresholds == changed_thresholds
    pd.testing.assert_frame_equal(before.loc[before.date <= dates[70]], after.loc[after.date <= dates[70]])
