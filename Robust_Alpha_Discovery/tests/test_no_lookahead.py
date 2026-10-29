from __future__ import annotations

import pandas as pd

from robust_alpha.data.synthetic import generate_synthetic_panel
from robust_alpha.features import compute_features


def test_features_never_change_when_future_inputs_are_mutated() -> None:
    frame = generate_synthetic_panel(n_days=170, n_assets=10, seed=11)
    signals = [
        "mom_20",
        "mom_60",
        "rev_5",
        "neg_vol_20",
        "neg_range_20",
        "neg_amihud_20",
        "volshock_5_20",
        "ma_gap_20",
        "mom20_over_vol20",
    ]
    baseline = compute_features(frame, signals)
    cutoff = sorted(frame["date"].unique())[119]

    mutated = frame.copy()
    future = mutated["date"] > cutoff
    mutated.loc[future, ["open", "high", "low", "close"]] *= 100.0
    mutated.loc[future, ["volume", "amount"]] *= 1_000.0
    changed = compute_features(mutated, signals)

    pd.testing.assert_frame_equal(
        baseline.loc[baseline["date"] <= cutoff, ["date", "asset", *signals]].reset_index(
            drop=True
        ),
        changed.loc[changed["date"] <= cutoff, ["date", "asset", *signals]].reset_index(
            drop=True
        ),
    )

