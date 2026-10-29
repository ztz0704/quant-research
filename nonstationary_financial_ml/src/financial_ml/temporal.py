"""Date-block temporal validation with explicit label-end purging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """Non-overlapping date boundaries, inclusive start and exclusive end."""

    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward(frame: pd.DataFrame, settings: dict) -> list[Fold]:
    """Generate strictly past-only rolling or expanding folds on observed dates."""
    mode = settings["mode"]
    if mode not in ("rolling", "expanding"):
        raise ValueError("Expected rolling or expanding")
    train, val, test, step, gap = [
        int(settings[key])
        for key in ("train_days", "validation_days", "test_days", "step_days", "embargo_days")
    ]
    if min(train, val, test) < 1 or gap < 0 or step < test:
        raise ValueError("Invalid windows or overlapping OOS test dates")
    dates = pd.DatetimeIndex(sorted(frame.date.unique()))
    result = []
    first = train + val + 2 * gap
    for start in range(first, len(dates) - test + 1, step):
        val_start = start - gap - val
        train_end = val_start - gap
        train_start = max(0, train_end - train) if mode == "rolling" else 0
        # A synthetic boundary one calendar day beyond the last test date preserves
        # inclusive observed sessions without introducing a fictitious trade date.
        stop = (
            dates[start + test]
            if start + test < len(dates)
            else pd.Timestamp(dates[-1].to_pydatetime() + timedelta(days=1))
        )
        result.append(
            Fold(
                len(result),
                dates[train_start],
                dates[train_end],
                dates[val_start],
                dates[start - gap],
                dates[start],
                stop,
            )
        )
    if not result:
        raise ValueError("Insufficient calendar history for declared temporal windows")
    return result


def rows(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    label_before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Eligibility never depends on future outcomes; labeled fit rows do."""
    mask = frame.date.ge(start) & frame.date.lt(end) & frame.eligible
    if label_before is not None:
        mask &= frame.target_valid & frame.label_end.lt(label_before)
    return frame.loc[mask].copy()


def partitions(frame: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/validation use matured labels; test predictions include unlabeled rows."""
    return (
        rows(frame, fold.train_start, fold.train_end, fold.validation_start),
        rows(frame, fold.validation_start, fold.validation_end, fold.test_start),
        rows(frame, fold.test_start, fold.test_end),
    )
