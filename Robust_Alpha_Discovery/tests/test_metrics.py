from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_alpha.evaluation.metrics import performance_metrics


def test_max_drawdown_includes_initial_capital() -> None:
    metrics = performance_metrics(pd.Series([-0.10, 0.05]), annualization=2)
    assert metrics["max_drawdown"] == pytest.approx(-0.10)


def test_sortino_uses_zero_mar_downside_deviation() -> None:
    returns = pd.Series([-0.02, 0.01, 0.03])
    metrics = performance_metrics(returns, annualization=3)
    expected_downside = np.sqrt((0.02**2) / 3) * np.sqrt(3)
    expected_return = returns.mean() * 3
    assert metrics["sortino"] == pytest.approx(expected_return / expected_downside)
