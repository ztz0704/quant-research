"""Exact book formula, bounds, asset isolation and resampling checks."""

import numpy as np
import pandas as pd
import pytest

from market_microstructure.data.resampling import resample_book
from market_microstructure.features.feature_pipeline import build_features, feature_columns
from market_microstructure.features.orderbook import book_features


def test_formulas_and_microprice_direction(book):
    row = book.iloc[:1].copy()
    row["bid_price_1"], row["ask_price_1"] = 99.0, 101.0
    row["bid_size_1"], row["ask_size_1"] = 300.0, 100.0
    features = book_features(row)
    assert features.mid.iloc[0] == 100
    assert features.spread.iloc[0] == 2
    assert features.imbalance_1.iloc[0] == 0.5
    assert features.microprice.iloc[0] == 100.5
    assert features.total_depth.iloc[0] == 400
    row["bid_size_1"], row["ask_size_1"] = 0.0, 0.0
    empty = book_features(row)
    assert empty.imbalance_1.iloc[0] == 0 and empty.microprice.iloc[0] == 100


@pytest.mark.parametrize("weighting", ["equal", "distance_decay"])
def test_bounds_and_depth(book, weighting):
    result = book_features(book, levels=5, weighting=weighting)
    assert result.imbalance_k.between(-1, 1).all()
    assert (result.microprice >= book.bid_price_1).all()
    assert (result.microprice <= book.ask_price_1).all()
    expected = book[[f"{side}_size_{k}" for side in ("bid", "ask") for k in range(1, 6)]].sum(axis=1)
    np.testing.assert_allclose(result.total_depth, expected)


def test_asset_isolation_and_missing_trades(book):
    result = build_features(book, levels=5)
    for asset in book.asset.unique():
        separate = build_features(book.loc[book.asset == asset], levels=5)
        together = result.loc[result.asset == asset].reset_index(drop=True)
        pd.testing.assert_frame_equal(together, separate)
    no_trades = build_features(book.drop(columns=["trade_size", "trade_side", "trade_price"]))
    assert "trade_intensity" not in feature_columns(no_trades)
    assert "synthetic_regime" not in feature_columns(result)
    assert result.realized_vol_bps.iloc[0] != result.realized_vol_bps.iloc[0]


def test_causal_resampling(book):
    result = resample_book(book, "150ms")
    assert (result.observed_timestamp <= result.timestamp).all()
    assert "trade_size" not in result
