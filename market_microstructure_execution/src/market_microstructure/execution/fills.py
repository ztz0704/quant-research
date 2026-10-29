"""Explicitly uncalibrated simplified maker probability, using current observations only."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class FillParameters:
    """Sensitivity parameters, not exchange-estimated queue coefficients."""

    intercept: float = -1.0
    intensity_coefficient: float = 1.0
    imbalance_coefficient: float = 1.5
    queue_coefficient: float = 1.0
    queue_scale: float = 1000.0
    flow_scale: float = 50.0


def fill_probability(side: float, queue: float, trade_size: float,
                     imbalance: float, intensity: float, parameters: FillParameters) -> float:
    """Opposing pressure increases a passive fill's chance and adverse-selection exposure."""
    p = parameters
    if p.queue_scale <= 0 or p.flow_scale <= 0:
        raise ValueError("Positive fill model scales required")
    intensity = intensity if np.isfinite(intensity) else 0.0
    logit = (p.intercept + p.intensity_coefficient * (intensity + np.log1p(trade_size / p.flow_scale))
             - p.imbalance_coefficient * side * imbalance
             - p.queue_coefficient * np.log1p(queue / p.queue_scale))
    return float(expit(logit))
