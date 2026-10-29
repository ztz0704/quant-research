"""Purging, frozen scaler statistics, OOS stability and statistical baselines."""

import copy

import numpy as np
import pandas as pd
import pytest

from market_microstructure.data.synthetic import SyntheticConfig, generate_orderbook
from market_microstructure.features.feature_pipeline import build_features
from market_microstructure.labels.labeling import build_labels
from market_microstructure.models.estimators import make_model
from market_microstructure.models.research import run_research
from market_microstructure.models.validation import WalkForwardConfig, walk_forward_splits
from market_microstructure.utils.config import load_config


def research_fixture():
    """Short sample with two chronological folds and three classes."""
    book = generate_orderbook(SyntheticConfig(events=500, assets=("A", "B")), seed=8)
    frame = build_features(book, rolling_window=10, min_periods=3)
    frame = frame.join(build_labels(book, horizon_events=5, threshold_bps=0.1))
    config = load_config()
    config["validation"].update(train_events=200, validation_events=50, test_events=100,
                                step_events=100, embargo_events=5)
    config["labels"].update(horizon_events=5, threshold_bps=0.1)
    config["model"].update(forest_trees=5, forest_min_leaf=5)
    return frame, config


def test_purged_shared_clock_and_rolling():
    frame, config = research_fixture()
    for window in ("expanding", "rolling"):
        settings = {**config["validation"], "window": window}
        folds = list(walk_forward_splits(frame, WalkForwardConfig(**settings)))
        assert len(folds) == 2
        for fold in folds:
            assert frame.iloc[fold.train].label_end.max() < fold.validation_start
            assert frame.iloc[fold.validation].label_end.max() < fold.test_start
            assert frame.iloc[fold.test].groupby("timestamp").size().eq(2).all()
        assert not set(folds[0].test) & set(folds[1].test)
    with pytest.raises(ValueError, match="overlap"):
        list(walk_forward_splits(frame, WalkForwardConfig(step_events=10, test_events=100)))


def test_scaler_only_fits_training():
    config = load_config()
    model = make_model("linear", config["model"], 1)
    x = pd.DataFrame({"imbalance": [0.0, 1.0, 2.0], "depth": [1.0, 2.0, np.nan]})
    model.fit(x, [0.0, 1.0, 2.0])
    expected = np.array([1.0, 1.5])
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, expected)
    before = copy.deepcopy(model.named_steps["scaler"].mean_)
    model.predict(pd.DataFrame({"imbalance": [1e9], "depth": [1e9]}))
    np.testing.assert_array_equal(model.named_steps["scaler"].mean_, before)


def test_research_future_changes_do_not_change_prior_signals():
    frame, config = research_fixture()
    original = run_research(frame, config)
    changed = frame.copy()
    last_fold_start = pd.Timestamp(original.audit.test_start.max())
    future = changed.timestamp >= last_fold_start
    for name in ("mid", "imbalance_1", "spread_bps", "future_return_bps", "total_depth"):
        changed.loc[future, name] += 10000
    changed.loc[future, "direction"] *= -1
    perturbed = run_research(changed, config)
    earlier = original.predictions.fold == 0
    pd.testing.assert_frame_equal(original.predictions.loc[earlier].reset_index(drop=True),
                                  perturbed.predictions.loc[earlier].reset_index(drop=True))
    assert original.predictions.signal.between(-1, 1).all()
    assert not original.predictions.duplicated(["model", "timestamp", "asset"]).any()
    assert {"t_stat_hac", "permutation_mean"}.issubset(original.importance)
    assert not {"direction", "future_return_bps", "synthetic_regime"} & set(original.columns)


def test_ablation_cannot_fit_future_targets_as_features():
    frame, config = research_fixture()
    with pytest.raises(ValueError, match="whitelisted"):
        run_research(frame, config, feature_subset=["imbalance_1", "future_return_bps"])
