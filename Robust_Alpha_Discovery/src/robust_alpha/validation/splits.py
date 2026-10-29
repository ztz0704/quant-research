"""Research-phase windows and holdout gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from robust_alpha.config import ConfigError

RESEARCH_PHASES = ("auto", "development", "validation", "pre_holdout", "holdout")


@dataclass(frozen=True)
class ResearchWindow:
    """Resolved load boundary and evaluation window for one pipeline run."""

    phase: str
    evaluation_start: pd.Timestamp | None
    evaluation_end: pd.Timestamp | None
    load_end: pd.Timestamp | None
    result_scope: str


def _date_range(split: dict[str, Any], name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        values = split[name]
        if not isinstance(values, list | tuple) or len(values) != 2:
            raise ValueError
        start, end = (pd.Timestamp(value).normalize() for value in values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"time_split.{name} must contain [start_date, end_date].") from exc
    if start > end:
        raise ConfigError(f"time_split.{name} starts after it ends.")
    return start, end


def validate_time_split(config: dict[str, Any]) -> None:
    """Validate ordered, non-overlapping development/validation/holdout ranges."""

    split = config.get("time_split")
    if split is None:
        return
    if not isinstance(split, dict):
        raise ConfigError("time_split must be a mapping.")
    development = _date_range(split, "development")
    validation = _date_range(split, "validation")
    holdout = _date_range(split, "final_holdout")
    if not isinstance(split.get("final_holdout_confirmed_untouched"), bool):
        raise ConfigError("time_split.final_holdout_confirmed_untouched must be a YAML boolean.")
    if not development[1] < validation[0]:
        raise ConfigError("Development must end before validation starts.")
    if not validation[1] < holdout[0]:
        raise ConfigError("Validation must end before final holdout starts.")


def resolve_research_window(
    config: dict[str, Any],
    *,
    research_phase: str = "auto",
    include_final_holdout: bool = False,
) -> ResearchWindow:
    """Resolve a safe evaluation period and enforce the double holdout gate."""

    if research_phase not in RESEARCH_PHASES:
        raise ConfigError(f"research_phase must be one of {list(RESEARCH_PHASES)}.")
    split = config.get("time_split")
    if split is None:
        if research_phase not in {"auto", "pre_holdout"}:
            raise ConfigError("Named research phases require a time_split configuration.")
        return ResearchWindow(
            phase="all",
            evaluation_start=None,
            evaluation_end=None,
            load_end=None,
            result_scope="full configured sample; no final holdout is defined",
        )

    validate_time_split(config)
    development = _date_range(split, "development")
    validation = _date_range(split, "validation")
    holdout = _date_range(split, "final_holdout")
    phase = "holdout" if include_final_holdout and research_phase == "auto" else research_phase
    phase = "pre_holdout" if phase == "auto" else phase

    if phase == "development":
        start, end = development
        scope = "development sample; exploratory model and rule development only"
    elif phase == "validation":
        start, end = validation
        scope = "validation sample; frozen-rule assessment, not final holdout evidence"
    elif phase == "pre_holdout":
        start, end = development[0], validation[1]
        scope = "development plus validation sample; exploratory/pre-holdout evidence"
    else:
        if not include_final_holdout:
            raise ConfigError("Holdout evaluation requires --include-final-holdout.")
        confirmed = bool(split.get("final_holdout_confirmed_untouched", False))
        protocol = config.get("protocol", {})
        frozen = str(protocol.get("status", "draft")).lower() == "frozen"
        if not confirmed or not frozen:
            raise ConfigError(
                "Refusing to load final holdout: both final_holdout_confirmed_untouched=true "
                "and protocol.status=frozen are required."
            )
        start, end = holdout
        scope = "final untouched holdout; one-time confirmatory evaluation"
    return ResearchWindow(
        phase=phase,
        evaluation_start=start,
        evaluation_end=end,
        load_end=end,
        result_scope=scope,
    )


def filter_evaluation_dates(
    frame: pd.DataFrame,
    window: ResearchWindow,
    *,
    date_column: str = "date",
) -> pd.DataFrame:
    """Return only rows whose signal date belongs to the resolved evaluation window."""

    if window.evaluation_start is None or window.evaluation_end is None:
        return frame.copy()
    dates = pd.to_datetime(frame[date_column])
    mask = dates.between(window.evaluation_start, window.evaluation_end)
    return frame.loc[mask].copy()


def filter_walk_forward_years(
    frame: pd.DataFrame,
    window: ResearchWindow,
    *,
    year_column: str = "test_year",
) -> pd.DataFrame:
    """Keep walk-forward test years that belong to the evaluation window."""

    if frame.empty or window.evaluation_start is None or window.evaluation_end is None:
        return frame.copy()
    years = pd.to_numeric(frame[year_column], errors="coerce")
    mask = years.between(window.evaluation_start.year, window.evaluation_end.year)
    return frame.loc[mask].reset_index(drop=True)
