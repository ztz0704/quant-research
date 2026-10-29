"""Configuration loading and early validation of research choices."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from risk_aware_portfolio.utils.paths import resolve_path

Config = dict[str, Any]


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    """Load a YAML config; reject invalid methods or incompatible scalar settings."""
    with resolve_path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Fail early instead of silently substituting defaults."""
    risk, portfolio = config["risk_model"], config["portfolio"]
    if not 0 < risk["ewma_lambda"] < 1:
        raise ValueError("ewma_lambda must be in (0, 1)")
    if not 2 <= risk["min_observations"] <= risk["lookback"]:
        raise ValueError("Require 2 <= min_observations <= lookback")
    for key in ("covariance_method", "specific_method"):
        if risk[key] not in {"sample", "ewma"}:
            raise ValueError(f"Unknown {key}: {risk[key]}")
    if risk["regression"] not in {"ols", "wls"}:
        raise ValueError("regression must be ols or wls")
    if config["backtest"]["rebalance_frequency"] not in {"daily", "weekly", "monthly"}:
        raise ValueError("Unknown rebalance frequency")
    if portfolio["risk_aversion"] < 0 or portfolio["cost_penalty"] < 0:
        raise ValueError("Objective penalties must be nonnegative")
    if portfolio["max_weight"] <= 0 or portfolio["gross_limit"] <= 0:
        raise ValueError("Exposure limits must be positive")
    if not 0 <= portfolio.get("min_weight", 0.0) <= portfolio["max_weight"]:
        raise ValueError("min_weight must be between zero and max_weight")
    if config["alpha"]["scale"] <= 0:
        raise ValueError("alpha scale must be positive")
    if len(set(risk["industries"])) != len(risk["industries"]):
        raise ValueError("Industry vocabulary must be unique and fixed ex ante")
    if risk["variance_floor"] <= 0 or risk["eigenvalue_floor"] < 0:
        raise ValueError("Invalid covariance floors")


def variant(config: Config, **sections: dict[str, Any]) -> Config:
    """Return an independent config with explicit section overrides."""
    result = deepcopy(config)
    for section, updates in sections.items():
        result[section].update(updates)
    validate_config(result)
    return result
