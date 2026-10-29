from __future__ import annotations

import numpy as np
import pandas as pd

from robust_alpha.validation.bootstrap import joint_moving_block_max_t
from robust_alpha.validation.multiple_testing import adjust_pvalues


def test_multiple_testing_adjustments_are_monotone_and_conservative() -> None:
    pvalues = pd.Series([0.001, 0.02, 0.03, 0.5], index=list("ABCD"))
    result = adjust_pvalues(pvalues, alpha=0.05).set_index("signal")
    assert result["naive_reject"].sum() == 3
    assert result["bonferroni_reject"].sum() == 1
    # BH step-up rejects the first three values at their rank-dependent thresholds.
    assert result["bh_reject"].sum() == 3
    assert (result["bonferroni_pvalue"] >= result["pvalue"]).all()


def test_joint_block_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(9)
    common = rng.normal(0, 1, 180)
    frame = pd.DataFrame(
        {
            "s1": 0.1 + common + rng.normal(0, 0.2, 180),
            "s2": common + rng.normal(0, 0.2, 180),
            "s3": rng.normal(0, 1, 180),
        }
    )
    first = joint_moving_block_max_t(
        frame, replications=50, block_length=10, seed=123
    )
    second = joint_moving_block_max_t(
        frame, replications=50, block_length=10, seed=123
    )
    np.testing.assert_allclose(first["max_statistics"], second["max_statistics"])
    pd.testing.assert_frame_equal(first["signal_results"], second["signal_results"])
    assert 0 <= first["global_pvalue"] <= 1
