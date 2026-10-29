import numpy as np
import pandas as pd

from financial_ml.importance import permute_within_dates


def test_grouped_permutation_preserves_relationship_and_date_blocks():
    frame = pd.DataFrame({"date": np.repeat([1, 2], 20), "x": np.arange(40.),
                          "z": 2 * np.arange(40.), "target": np.arange(40.) / 100})
    changed = permute_within_dates(frame, ["x", "z"], 8)
    assert changed.z.eq(2 * changed.x).all()
    assert changed.target.equals(frame.target)
    assert not changed.x.equals(frame.x)
    for date, section in changed.groupby("date"):
        assert set(section.x) == set(frame.loc[frame.date.eq(date), "x"])
