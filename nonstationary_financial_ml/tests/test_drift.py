import numpy as np

from financial_ml.drift import distribution_shift


def test_distribution_shift_uses_tail_bins_and_training_reference():
    rng = np.random.default_rng(8)
    train = rng.normal(size=5000)
    same = distribution_shift(train, train)
    moved = distribution_shift(train, train + 2)
    assert same["psi"] == 0
    assert same["ks_statistic"] == 0
    assert moved["psi"] > .5
    assert moved["ks_statistic"] > .5
    assert np.isfinite(distribution_shift(np.ones(100), np.full(100, 1e9))["psi"])
