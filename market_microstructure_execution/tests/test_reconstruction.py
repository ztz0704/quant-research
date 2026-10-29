"""Message replay consumes observed executions; no matching is invented from quote crosses."""

import pandas as pd

from market_microstructure.data.reconstruction import MessageBook
from market_microstructure.labels.labeling import build_labels


def test_original_messages_and_cancel():
    book = MessageBook()
    book.add(1, 1, 10.0, 200, 2)
    book.add(2, 2, 10.01, 300, 2)
    assert book.snapshot(1)["bid_size_1"] == 200
    book.add(3, 1, 0, 100, 1)  # market order is tracked, not placed or simulated
    assert book.snapshot(1)["ask_size_1"] == 300
    book.reduce(3, 100, "trade")
    book.reduce(2, 100, "trade")
    assert book.snapshot(1)["ask_size_1"] == 200
    book.reduce(1, 200, "cancel")
    assert book.best(1) == 0
    book.reduce(999, 10, "trade")
    assert book.audit["missing_trade_order"] == 1


def test_crossed_book_not_fabricated_clear():
    book = MessageBook()
    book.add(1, 1, 10.0, 100, 2)
    book.add(2, 2, 9.99, 100, 2)
    assert book.snapshot(1) is None
    assert len(book.orders) == 2
    assert book.audit["crossed_snapshot"] == 1


def test_labels_do_not_span_lunch_or_overnight(book):
    book["session_id"] = ["AM"] * 120 + ["PM"] * 120
    labels = build_labels(book, horizon_events=5)
    for _, group in book.groupby(["asset", "session_id"]):
        assert labels.loc[group.index[-5:], "future_return"].isna().all()
    assert pd.isna(labels.label_end.iloc[-1])
