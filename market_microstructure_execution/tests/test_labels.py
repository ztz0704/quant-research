"""Off-by-one, multi-asset separation, irregular clocks and neutral class handling."""

import numpy as np
import pandas as pd
import pytest

from market_microstructure.labels.labeling import build_labels


def test_event_horizon_and_tail(book):
    labels = build_labels(book, horizon_events=10, threshold_bps=0)
    for _, group in book.groupby("asset"):
        mid = (group.bid_price_1 + group.ask_price_1) / 2
        expected = mid.shift(-10) / mid - 1
        np.testing.assert_allclose(labels.loc[group.index, "future_return"], expected, atol=1e-15)
        assert labels.loc[group.index[-10:], "future_return"].isna().all()
        assert labels.loc[group.index[0], "label_end"] == group.timestamp.iloc[10]


def test_time_horizon(book):
    group = book.loc[book.asset == "SYN_A"].iloc[[0, 1, 4, 9]].copy()
    labels = build_labels(group, horizon_events=None, horizon_ms=250)
    assert labels.label_end.iloc[0] == group.timestamp.iloc[2]
    assert labels.label_end.iloc[2] == group.timestamp.iloc[3]
    assert pd.isna(labels.label_end.iloc[-1])


def test_binary_neutral_and_invalid_horizon(book):
    labels = build_labels(book, threshold_bps=1000, classification="binary")
    assert labels.direction.isna().all()
    with pytest.raises(ValueError):
        build_labels(book, horizon_events=10, horizon_ms=100)
