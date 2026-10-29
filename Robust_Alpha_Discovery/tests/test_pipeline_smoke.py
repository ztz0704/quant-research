from __future__ import annotations

import pandas as pd

from robust_alpha.pipeline import run_pipeline


def test_full_synthetic_pipeline_smoke() -> None:
    artifacts = run_pipeline("configs/test_smoke.yaml", stop_after="report")
    assert artifacts["research_panel"].exists()
    assert artifacts["ic_summary"].exists()
    assert artifacts["multiple_testing"].exists()
    assert artifacts["report"].exists()
    assert artifacts["manifest"].exists()

    selections = pd.read_csv(artifacts["walk_forward_selections"])
    daily = pd.read_csv(artifacts["walk_forward_portfolio_daily"])
    assert not selections.empty
    assert not daily.empty
    assert set(daily["test_year"].unique()) == set(selections["test_year"].unique())
