import numpy as np

from financial_ml.robustness import block_mean_ci


def test_block_interval_and_constant_paired_difference():
    assert block_mean_ci(np.zeros(100), 100, 10) == (0., 0.)
    rng = np.random.default_rng(8)
    low, high = block_mean_ci(rng.normal(0, 1, 200), 200, 10)
    assert low < high
    assert np.isnan(block_mean_ci([1, 2, 3])[0])
