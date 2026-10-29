"""Signal registry dispatcher."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from robust_alpha.features.expressions import evaluate_expression, validate_definitions
from robust_alpha.features.liquidity import negative_amihud, volume_shock
from robust_alpha.features.momentum import momentum, risk_adjusted_momentum, skip_momentum
from robust_alpha.features.price import moving_average_gap
from robust_alpha.features.reversal import reversal
from robust_alpha.features.specs import SIGNAL_SPECS
from robust_alpha.features.volatility import negative_range, negative_volatility


def compute_features(
    frame: pd.DataFrame,
    signal_names: Sequence[str],
    custom_signals: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute configured features using observations dated no later than each row."""

    custom_signals = {} if custom_signals is None else custom_signals
    validate_definitions(custom_signals, set(SIGNAL_SPECS))
    unknown = sorted(set(signal_names) - set(SIGNAL_SPECS) - set(custom_signals))
    if unknown:
        raise KeyError(f"Unknown signals: {unknown}")
    panel = frame.sort_values(["asset", "date"], kind="mergesort").copy()
    asset = panel["asset"]

    for name in signal_names:
        if name in custom_signals:
            definition = custom_signals[name]
            panel[name] = definition.get("direction", 1) * evaluate_expression(
                panel, definition["expression"]
            )
            continue
        spec = SIGNAL_SPECS[name]
        parameters = spec.parameters
        if spec.family == "momentum":
            values = momentum(panel["close"], asset, **parameters)
        elif spec.family == "skip_momentum":
            values = skip_momentum(panel["close"], asset, **parameters)
        elif spec.family == "reversal":
            values = reversal(panel["close"], asset, **parameters)
        elif spec.family == "negative_volatility":
            values = negative_volatility(panel["close"], asset, **parameters)
        elif spec.family == "negative_range":
            values = negative_range(
                panel["high"], panel["low"], panel["close"], asset, **parameters
            )
        elif spec.family == "negative_amihud":
            values = negative_amihud(panel["close"], panel["amount"], asset, **parameters)
        elif spec.family == "volume_shock":
            values = volume_shock(panel["volume"], asset, **parameters)
        elif spec.family == "moving_average_gap":
            values = moving_average_gap(panel["close"], asset, **parameters)
        elif spec.family == "risk_adjusted_momentum":
            values = risk_adjusted_momentum(panel["close"], asset, **parameters)
        else:  # pragma: no cover - registry and dispatcher should evolve together
            raise RuntimeError(f"Unsupported signal family: {spec.family}")
        panel[name] = values
    return panel
