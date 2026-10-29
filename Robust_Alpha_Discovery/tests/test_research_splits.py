from __future__ import annotations

import pandas as pd
import pytest

from robust_alpha.config import ConfigError
from robust_alpha.validation.splits import filter_evaluation_dates, resolve_research_window


def _config(*, confirmed: bool = False, status: str = "draft") -> dict[str, object]:
    return {
        "protocol": {"status": status},
        "time_split": {
            "development": ["2018-01-01", "2019-12-31"],
            "validation": ["2020-01-01", "2020-12-31"],
            "final_holdout": ["2021-01-01", "2021-12-31"],
            "final_holdout_confirmed_untouched": confirmed,
        },
    }


def test_validation_phase_filters_evaluation_but_keeps_history_available() -> None:
    window = resolve_research_window(_config(), research_phase="validation")
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2019-12-31", "2020-01-02", "2021-01-04"]), "value": range(3)}
    )
    result = filter_evaluation_dates(frame, window)
    assert result["date"].tolist() == [pd.Timestamp("2020-01-02")]
    assert window.load_end == pd.Timestamp("2020-12-31")


def test_holdout_requires_confirmation_and_frozen_protocol() -> None:
    with pytest.raises(ConfigError, match="both"):
        resolve_research_window(
            _config(confirmed=True, status="draft"),
            research_phase="holdout",
            include_final_holdout=True,
        )
    window = resolve_research_window(
        _config(confirmed=True, status="frozen"),
        research_phase="holdout",
        include_final_holdout=True,
    )
    assert window.phase == "holdout"
