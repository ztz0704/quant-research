"""Load explicit YAML configuration; reject ambiguous basic settings."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import yaml

from market_microstructure.utils.paths import project_path


def load_config(path: str | Path = "configs/base.yaml", _seen: tuple[str, ...] = ()) -> dict[str, Any]:
    """Load the full research configuration without changing any seed."""
    with project_path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    resolved = str(project_path(path).resolve())
    if resolved in _seen:
        raise ValueError("Cyclic configuration inheritance")
    if "extends" in config:
        base = load_config(config.pop("extends"), (*_seen, resolved))
        config = _merge(base, config)
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    if config["data"]["source"] not in {"synthetic", "file"}:
        raise ValueError("data.source must be synthetic or file")
    if config["data"]["invalid_policy"] not in {"raise", "drop"}:
        raise ValueError("data.invalid_policy must be raise or drop")
    return config


def _merge(base: dict, override: dict) -> dict:
    """Copy nested mappings; overrides never mutate another run's configuration."""
    result = {**base}
    for key, value in override.items():
        result[key] = _merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return result


def script_config(description: str) -> dict[str, Any]:
    """Parse the common --config command line argument."""
    parser = ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/base.yaml")
    return load_config(parser.parse_args().config)
