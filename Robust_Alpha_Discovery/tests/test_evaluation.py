from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_alpha.evaluation.factor_return import daily_factor_returns, factor_return_summary
from robust_alpha.evaluation.ic import daily_spearman_ic, summarize_ic


def test_daily_ic_is_one_for_perfect_monotonic_cross_sections() -> None:
    rows = []
    for date in pd.date_range("2020-01-01", periods=8):
        for asset in range(10):
            rows.append(
                {"date": date, "asset": asset, "signal": asset, "target": asset / 100.0}
            )
    frame = pd.DataFrame(rows)
    daily = daily_spearman_ic(frame, ["signal"], target_column="target", min_assets=5)
    summary = summarize_ic(daily)
    assert np.allclose(daily["ic"], 1.0)
    assert summary.loc[0, "mean_ic"] == pytest.approx(1.0)
    assert summary.loc[0, "ic_hit_rate"] == 1.0


def test_factor_return_recovers_known_slope_and_hac_summary() -> None:
    rows = []
    rng = np.random.default_rng(3)
    for date in pd.date_range("2020-01-01", periods=30):
        signal = np.linspace(-1, 1, 20)
        target = 0.002 + 0.01 * signal + rng.normal(0, 0.0001, len(signal))
        rows.extend(
            {"date": date, "asset": index, "signal": x, "target": y}
            for index, (x, y) in enumerate(zip(signal, target, strict=True))
        )
    daily = daily_factor_returns(
        pd.DataFrame(rows), ["signal"], target_column="target", min_assets=10
    )
    summary = factor_return_summary(daily, hac_maxlags=3)
    assert daily["factor_return"].mean() == pytest.approx(0.01, abs=5e-5)
    assert summary.loc[0, "hac_t_stat"] > 10
