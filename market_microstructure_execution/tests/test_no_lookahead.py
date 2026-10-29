"""Future book, price and trade perturbations must not change prefix features."""

import pandas as pd

from market_microstructure.features.feature_pipeline import build_features, feature_columns


def test_future_mutation_does_not_change_features(book):
    cutoff = book.timestamp.iloc[99]
    future = book.timestamp > cutoff
    changed = book.copy()
    for column in book:
        if "price" in column:
            changed.loc[future, column] *= 1.3
        elif "size" in column:
            changed.loc[future, column] *= 3
    changed.loc[future, "trade_side"] *= -1
    first = build_features(book, levels=5)
    second = build_features(changed, levels=5)
    columns = feature_columns(first)
    prefix = first.timestamp <= cutoff
    pd.testing.assert_frame_equal(first.loc[prefix, columns], second.loc[prefix, columns])
    truncated = build_features(book.loc[~future], levels=5)
    pd.testing.assert_frame_equal(first.loc[prefix, columns], truncated[columns])


def test_tick_rule_is_backward_only(book):
    book = book.drop(columns="trade_side")
    cutoff = book.timestamp.iloc[79]
    changed = book.copy()
    changed.loc[book.timestamp > cutoff, "trade_price"] *= 2
    original = build_features(book)
    perturbation = build_features(changed)
    mask = original.timestamp <= cutoff
    pd.testing.assert_series_equal(original.loc[mask, "signed_volume_imbalance"],
                                   perturbation.loc[mask, "signed_volume_imbalance"])
