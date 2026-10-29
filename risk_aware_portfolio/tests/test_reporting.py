import json

import pytest

from risk_aware_portfolio.attribution.performance import block_bootstrap_sharpe
from risk_aware_portfolio.utils.manifest import verify_manifest, write_manifest


def test_manifest_rejects_stale_input(config, tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("original", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, config, data)
    verify_manifest(manifest, config, data)
    assert json.loads(manifest.read_text(encoding="utf-8"))["config"] == config
    data.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale"):
        verify_manifest(manifest, config, data)


def test_bootstrap_is_reproducible():
    values = [0.01, -0.01, 0.005, 0.003, -0.004] * 10
    first = block_bootstrap_sharpe(values, repetitions=100)
    assert first == block_bootstrap_sharpe(values, repetitions=100)
    assert first[0] < first[1]
