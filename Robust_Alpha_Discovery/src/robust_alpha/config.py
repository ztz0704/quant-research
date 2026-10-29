"""Configuration loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and validate the top-level contract."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a YAML mapping.")

    required = {
        "project",
        "data",
        "timing",
        "preprocessing",
        "signals",
        "evaluation",
        "costs",
        "multiple_testing",
        "output",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ConfigError(f"Missing top-level configuration sections: {missing}")
    if not config["signals"]:
        raise ConfigError("At least one signal must be configured.")
    if len(config["signals"]) != len(set(config["signals"])):
        raise ConfigError("Configured signal names must be unique.")
    _validate_nested_config(config)
    return config


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer.")
    return value


def _validate_nested_config(config: dict[str, Any]) -> None:
    from robust_alpha.features.expressions import validate_definitions
    from robust_alpha.features.specs import SIGNAL_SPECS
    from robust_alpha.validation.splits import validate_time_split

    data = config["data"]
    if data.get("source") not in {"synthetic", "parquet", "csv"}:
        raise ConfigError("data.source must be synthetic, parquet, or csv.")
    _positive_integer(data.get("min_history"), "data.min_history")

    custom = config.get("custom_signals", {})
    try:
        validate_definitions(custom, set(SIGNAL_SPECS))
    except (ValueError, SyntaxError) as exc:
        raise ConfigError(str(exc)) from exc
    unknown = sorted(set(config["signals"]) - set(SIGNAL_SPECS) - set(custom))
    if unknown:
        raise ConfigError(f"Unknown configured signals: {unknown}")
    method = config["preprocessing"].get("method")
    if method not in {"rank_gaussian", "percentile_rank", "zscore"}:
        raise ConfigError("preprocessing.method is not supported.")

    timing = config["timing"]
    for name in ("require_entry_tradable", "require_exit_tradable"):
        if not isinstance(timing.get(name), bool):
            raise ConfigError(f"timing.{name} must be a YAML boolean.")

    evaluation = config["evaluation"]
    _positive_integer(evaluation.get("annualization"), "evaluation.annualization")
    min_assets = _positive_integer(
        evaluation.get("min_assets_per_date"), "evaluation.min_assets_per_date"
    )
    if min_assets < 3:
        raise ConfigError("evaluation.min_assets_per_date must be at least 3.")
    quantile = evaluation.get("quantile")
    if (
        not isinstance(quantile, int | float)
        or isinstance(quantile, bool)
        or not 0 < quantile <= 0.5
    ):
        raise ConfigError("evaluation.quantile must be in (0, 0.5].")
    hac_lags = evaluation.get("hac_maxlags")
    if isinstance(hac_lags, bool) or not isinstance(hac_lags, int) or hac_lags < 0:
        raise ConfigError("evaluation.hac_maxlags must be a non-negative integer.")

    walk = config.get("walk_forward", {})
    _positive_integer(walk.get("train_years"), "walk_forward.train_years")
    _positive_integer(walk.get("top_k"), "walk_forward.top_k")

    costs = config["costs"].get("bps")
    if not isinstance(costs, list) or not costs:
        raise ConfigError("costs.bps must be a non-empty list.")
    invalid_cost = any(
        isinstance(value, bool) or not isinstance(value, int | float) or value < 0
        for value in costs
    )
    if invalid_cost:
        raise ConfigError("costs.bps values must be non-negative numbers.")

    multiple = config["multiple_testing"]
    alpha = multiple.get("alpha")
    if not isinstance(alpha, int | float) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise ConfigError("multiple_testing.alpha must be in (0, 1).")
    _positive_integer(multiple.get("block_length"), "multiple_testing.block_length")
    _positive_integer(
        multiple.get("bootstrap_replications"), "multiple_testing.bootstrap_replications"
    )

    validate_time_split(config)
    if config.get("time_split") is not None:
        protocol = config.get("protocol")
        if not isinstance(protocol, dict) or str(protocol.get("status", "")).lower() not in {
            "draft",
            "frozen",
        }:
            raise ConfigError("A split experiment requires protocol.status=draft or frozen.")


def project_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a configuration path relative to the project root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    base = Path.cwd() if root is None else Path(root)
    return (base / candidate).resolve()
