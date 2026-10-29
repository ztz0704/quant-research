from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_alpha.config import ConfigError
from robust_alpha.data.align import add_open_to_open_target
from robust_alpha.data.schema import PanelValidationError, normalize_and_validate_panel
from robust_alpha.data.synthetic import generate_synthetic_panel
from robust_alpha.pipeline import _purge_targets_at_boundary, run_pipeline


def test_synthetic_panel_satisfies_schema_and_is_deterministic() -> None:
    first = generate_synthetic_panel(n_days=130, n_assets=10, seed=7)
    second = generate_synthetic_panel(n_days=130, n_assets=10, seed=7)
    validated = normalize_and_validate_panel(first)
    pd.testing.assert_frame_equal(first, second)
    assert len(validated) == 1300
    assert not validated.duplicated(["date", "asset"]).any()


def test_schema_rejects_duplicate_asset_dates() -> None:
    frame = generate_synthetic_panel(n_days=130, n_assets=10, seed=1)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(PanelValidationError, match="Duplicate"):
        normalize_and_validate_panel(duplicated)


def test_schema_strictly_parses_csv_boolean_strings() -> None:
    frame = generate_synthetic_panel(n_days=130, n_assets=10, seed=1).iloc[:2].copy()
    frame["universe_member"] = ["True", "False"]
    frame["tradable"] = ["1", "0"]
    result = normalize_and_validate_panel(frame)
    assert bool(result.loc[0, "universe_member"])
    assert not bool(result.loc[1, "universe_member"])
    assert not bool(result.loc[1, "tradable"])


def test_schema_rejects_missing_boolean_state() -> None:
    frame = generate_synthetic_panel(n_days=130, n_assets=10, seed=1).iloc[:2].copy()
    frame["tradable"] = frame["tradable"].astype(object)
    frame.loc[0, "tradable"] = None
    with pytest.raises(PanelValidationError, match="cannot contain missing"):
        normalize_and_validate_panel(frame)


def test_schema_rejects_sparse_asset_calendar_that_would_shift_targets() -> None:
    dates = pd.date_range("2020-01-01", periods=3)
    rows = []
    for asset, asset_dates in (("A", dates[[0, 2]]), ("B", dates)):
        for date in asset_dates:
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0,
                    "volume": 100.0,
                    "amount": 1000.0,
                    "universe_member": True,
                    "tradable": True,
                }
            )
    with pytest.raises(PanelValidationError, match="every market date"):
        normalize_and_validate_panel(pd.DataFrame(rows))


def test_target_is_next_open_to_open() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5),
            "asset": "A",
            "open": [10.0, 11.0, 12.0, 15.0, 18.0],
            "tradable": [True] * 5,
        }
    )
    aligned = add_open_to_open_target(frame)
    assert aligned.loc[0, "entry_open"] == 11.0
    assert aligned.loc[0, "exit_open"] == 12.0
    assert aligned.loc[0, "entry_date"] == pd.Timestamp("2020-01-02")
    assert aligned.loc[0, "exit_date"] == pd.Timestamp("2020-01-03")
    assert aligned.loc[0, "target_o2o"] == pytest.approx(np.log(12.0 / 11.0))
    assert aligned.loc[2, "target_o2o"] == pytest.approx(np.log(18.0 / 15.0))
    assert pd.isna(aligned.loc[3, "target_o2o"])
    assert not aligned.loc[3, "target_valid"]


def test_target_requires_entry_and_exit_tradeability() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4),
            "asset": "A",
            "open": [10.0, 11.0, 12.0, 13.0],
            "tradable": [True, False, True, True],
        }
    )
    aligned = add_open_to_open_target(frame)
    assert pd.isna(aligned.loc[0, "target_o2o"])
    assert aligned.loc[1, "target_o2o"] == pytest.approx(np.log(13.0 / 12.0))


def test_year_boundary_labels_are_purged_before_walk_forward_selection() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-12-30", "2019-12-31", "2020-01-02"]),
            "asset": "A",
            "open": [10.0, 11.0, 1000.0],
            "tradable": True,
        }
    )
    aligned = add_open_to_open_target(frame)
    purged = _purge_targets_at_boundary(
        aligned,
        target_column="target_o2o",
        require_same_calendar_year=True,
    )
    assert pd.isna(purged.loc[0, "target_o2o"])
    assert not purged.loc[0, "target_valid"]


def test_final_holdout_requires_explicit_frozen_confirmation() -> None:
    with pytest.raises(ConfigError, match="Refusing to load final holdout"):
        run_pipeline(
            "configs/experiment_v1.yaml",
            include_final_holdout=True,
            stop_after="panel",
        )
