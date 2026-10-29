import numpy as np

from financial_ml.data import synthetic
from financial_ml.labels import add_labels


def test_horizon_and_tail_and_eligibility():
    frame = synthetic(days=20, assets=2)
    frame["eligible"] = True
    result = add_labels(frame, 5)
    asset = frame.asset.iloc[0]
    values = frame.loc[frame.asset.eq(asset), "return"].to_numpy()
    observed = result.loc[result.asset.eq(asset)]
    assert np.isclose(observed.target.iloc[0], np.prod(1 + values[1:6]) - 1)
    assert observed.target.iloc[-5:].isna().all()
    assert result.eligible.all()
