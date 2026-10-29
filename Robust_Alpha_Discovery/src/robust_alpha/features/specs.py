"""Ex-ante signal registry used by configuration and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SignalSpec:
    """Declarative description of a candidate signal."""

    name: str
    family: str
    expected_direction: int | None
    parameters: dict[str, Any] = field(default_factory=dict)


def _spec(
    name: str,
    family: str,
    expected_direction: int | None = 1,
    **parameters: Any,
) -> SignalSpec:
    return SignalSpec(name, family, expected_direction, parameters)


SIGNAL_SPECS: dict[str, SignalSpec] = {
    "mom_5": _spec("mom_5", "momentum", horizon=5),
    "mom_20": _spec("mom_20", "momentum", horizon=20),
    "mom_60": _spec("mom_60", "momentum", horizon=60),
    "mom_120": _spec("mom_120", "momentum", horizon=120),
    "mom_20_skip5": _spec("mom_20_skip5", "skip_momentum", horizon=20, skip=5),
    "rev_1": _spec("rev_1", "reversal", horizon=1),
    "rev_3": _spec("rev_3", "reversal", horizon=3),
    "rev_5": _spec("rev_5", "reversal", horizon=5),
    "neg_vol_10": _spec("neg_vol_10", "negative_volatility", window=10),
    "neg_vol_20": _spec("neg_vol_20", "negative_volatility", window=20),
    "neg_vol_60": _spec("neg_vol_60", "negative_volatility", window=60),
    "neg_range_10": _spec("neg_range_10", "negative_range", window=10),
    "neg_range_20": _spec("neg_range_20", "negative_range", window=20),
    "neg_amihud_20": _spec("neg_amihud_20", "negative_amihud", window=20),
    "neg_amihud_60": _spec("neg_amihud_60", "negative_amihud", window=60),
    "volshock_5_20": _spec(
        "volshock_5_20", "volume_shock", expected_direction=None, short=5, long=20
    ),
    "ma_gap_20": _spec("ma_gap_20", "moving_average_gap", expected_direction=None, window=20),
    "ma_gap_60": _spec("ma_gap_60", "moving_average_gap", expected_direction=None, window=60),
    "mom20_over_vol20": _spec(
        "mom20_over_vol20", "risk_adjusted_momentum", horizon=20, vol_window=20
    ),
    "mom60_over_vol60": _spec(
        "mom60_over_vol60", "risk_adjusted_momentum", horizon=60, vol_window=60
    ),
}

