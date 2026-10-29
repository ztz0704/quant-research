"""Chronological folds on shared timestamp boundaries, purged by label end."""

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    """Counts refer to unique timestamps, preserving simultaneous asset observations."""

    window: Literal["expanding", "rolling"] = "expanding"
    train_events: int = 2500
    validation_events: int = 500
    test_events: int = 750
    step_events: int = 750
    embargo_events: int = 10


@dataclass(frozen=True)
class Fold:
    """Integer positions and immutable boundary metadata."""

    number: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(frame: pd.DataFrame, config: WalkForwardConfig) -> Iterator[Fold]:
    """Do not let training/validation labels touch the following embargo boundary."""
    c = config
    if c.window not in {"expanding", "rolling"} or min(
        c.train_events, c.validation_events, c.test_events, c.step_events,
    ) <= 0 or c.embargo_events < 0:
        raise ValueError("Invalid walk-forward configuration")
    if c.step_events < c.test_events:
        raise ValueError("Test windows must not overlap")
    if c.embargo_events >= min(c.train_events, c.validation_events):
        raise ValueError("Embargo consumes the training or validation window")
    if not frame.timestamp.is_monotonic_increasing:
        raise ValueError("Globally ordered timestamps required")
    clock = pd.DatetimeIndex(frame.timestamp.unique())
    train_stop, number = c.train_events, 0
    while train_stop + c.validation_events + c.test_events <= len(clock):
        test_start = train_stop + c.validation_events
        test_stop = test_start + c.test_events
        train_start = 0 if c.window == "expanding" else train_stop - c.train_events
        train_cutoff = clock[train_stop - c.embargo_events]
        validation_cutoff = clock[test_start - c.embargo_events]
        train_mask = (frame.timestamp >= clock[train_start]) & (frame.timestamp < train_cutoff)
        train_mask &= frame.label_end < train_cutoff
        validation_mask = (frame.timestamp >= clock[train_stop]) & (frame.timestamp < validation_cutoff)
        validation_mask &= frame.label_end < validation_cutoff
        test_mask = (frame.timestamp >= clock[test_start]) & (frame.timestamp <= clock[test_stop - 1])
        yield Fold(number, np.flatnonzero(train_mask), np.flatnonzero(validation_mask),
                   np.flatnonzero(test_mask), clock[train_stop], clock[test_start], clock[test_stop - 1])
        number += 1
        train_stop += c.step_events
    if number == 0:
        raise ValueError("Insufficient unique timestamps for one complete walk-forward fold")
