import numpy as np
import pandas as pd

from financial_ml.preprocessing import residualize, standardize, transform


def test_current_date_only_and_constant():
    frame = pd.DataFrame({"date": [1, 1, 2, 2], "f": [1., 3., 1e9, -1e9], "eligible": True})
    settings = {"winsorize": "percentile", "standardize": True, "neutralize": False}
    result = transform(frame, ["f"], settings)
    assert np.allclose(result.loc[:1, "f"], [-1, 1])
    frame.loc[2:, "f"] = 0
    assert result.loc[:1, "f"].equals(transform(frame, ["f"], settings).loc[:1, "f"])
    assert standardize(pd.Series([2., 2.])).eq(0).all()


def test_exposure_residual_orthogonality():
    section = pd.DataFrame({"industry": ["A", "A", "B", "B", "B"], "size": [1., 2., 3., 4., 5.]})
    residual = residualize(np.array([1., 3., 2., 7., 4.]), section)
    assert abs(residual.sum()) < 1e-10
    assert abs(residual @ section["size"]) < 1e-10
