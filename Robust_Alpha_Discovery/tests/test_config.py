from __future__ import annotations

from copy import deepcopy

import pytest

from robust_alpha.config import ConfigError, _validate_nested_config, load_config


def test_config_rejects_string_boolean() -> None:
    config = load_config("configs/test_smoke.yaml")
    config["timing"]["require_entry_tradable"] = "false"
    with pytest.raises(ConfigError, match="YAML boolean"):
        _validate_nested_config(config)


def test_config_rejects_overlapping_splits() -> None:
    config = load_config("configs/experiment_v1.yaml")
    bad = deepcopy(config)
    bad["time_split"]["validation"][0] = "2020-01-01"
    from robust_alpha.validation.splits import validate_time_split

    with pytest.raises(ConfigError, match="Development must end"):
        validate_time_split(bad)
