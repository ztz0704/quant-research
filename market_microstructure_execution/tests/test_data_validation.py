"""Schema, auditing and deterministic generation invariants."""

import numpy as np
import pandas as pd
import pytest

from market_microstructure.data.loader import load_orderbook, save_orderbook
from market_microstructure.data.synthetic import SyntheticConfig, generate_orderbook
from market_microstructure.data.validation import ValidationError, validate_orderbook
from market_microstructure.utils.config import load_config
from market_microstructure.utils.paths import PROJECT_ROOT, project_path


def test_valid_book_and_seed(book):
    clean, report = validate_orderbook(book)
    assert report.removed == 0 and len(clean) == 240
    pd.testing.assert_frame_equal(book, generate_orderbook(SyntheticConfig(events=120), seed=7))
    assert clean.groupby("asset").timestamp.apply(lambda x: x.is_monotonic_increasing).all()


@pytest.mark.parametrize("column,value", [
    ("bid_price_1", np.inf), ("ask_price_1", 0), ("bid_size_1", -1),
    ("ask_size_1", np.nan), ("trade_size", np.inf), ("asset", ""),
])
def test_invalid_row_audit(book, column, value):
    book.loc[0, column] = value
    with pytest.raises(ValidationError) as error:
        validate_orderbook(book)
    assert error.value.report.removed == 1
    clean, report = validate_orderbook(book, mode="drop", max_removed_fraction=0.01)
    assert len(clean) == len(book) - 1 and report.removed_percentage == pytest.approx(100 / 240)


def test_crossed_duplicate_unsorted_and_partial(book):
    crossed = book.copy()
    crossed.loc[0, "bid_price_1"] = crossed.loc[0, "ask_price_1"]
    with pytest.raises(ValidationError):
        validate_orderbook(crossed)
    duplicated = pd.concat([book.iloc[:1], book], ignore_index=True)
    with pytest.raises(ValidationError, match="duplicate"):
        validate_orderbook(duplicated)
    with pytest.raises(ValidationError, match="sorted"):
        validate_orderbook(book.iloc[::-1], mode="drop")
    with pytest.raises(ValidationError, match="Incomplete"):
        validate_orderbook(book.drop(columns="bid_size_3"))


def test_no_mass_silent_removal(book):
    book.loc[:20, "bid_size_1"] = -1
    with pytest.raises(ValidationError):
        validate_orderbook(book, mode="drop", max_removed_fraction=0.01)


def test_loader_roundtrip(book, tmp_path):
    for suffix in ("parquet", "csv"):
        path = tmp_path / f"book.{suffix}"
        save_orderbook(book, path)
        result, report = load_orderbook(path)
        assert report.removed == 0
        np.testing.assert_allclose(result.bid_price_1, book.bid_price_1)


def test_synthetic_known_beta():
    config = SyntheticConfig(events=30000, assets=("A",), sigma=0.006, beta=0.01)
    frame = generate_orderbook(config, seed=41)
    mid = (frame.bid_price_1 + frame.ask_price_1) / 2
    imbalance = (frame.bid_size_1 - frame.ask_size_1) / (frame.bid_size_1 + frame.ask_size_1)
    beta = np.polyfit(imbalance.iloc[:-1], np.diff(mid), 1)[0]
    assert beta == pytest.approx(config.beta, abs=0.0005)


def test_config_and_paths():
    assert load_config()["seed"] == 20261003
    assert project_path("configs/base.yaml").is_file()
    assert PROJECT_ROOT.name == "market_microstructure_execution"
