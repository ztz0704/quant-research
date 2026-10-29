import numpy as np
import pytest

from risk_aware_portfolio.data.loader import load_panel
from risk_aware_portfolio.data.synthetic import generate_synthetic
from risk_aware_portfolio.data.universe import require_balanced_universe
from risk_aware_portfolio.data.validation import validate_panel


def test_valid_reproducible(config, synthetic):
    validate_panel(synthetic.panel)
    assert synthetic.panel.equals(generate_synthetic(config).panel)
    assert len(require_balanced_universe(synthetic.panel)) == 40


@pytest.mark.parametrize("kind", ["duplicate", "order", "column", "missing", "inf", "return"])
def test_invalid_panel(synthetic, kind):
    panel = synthetic.panel.copy()
    if kind == "duplicate":
        panel.iloc[1] = panel.iloc[0]
    elif kind == "order":
        panel = panel.iloc[::-1]
    elif kind == "column":
        panel = panel.drop(columns="alpha")
    elif kind == "missing":
        panel.loc[0, "alpha"] = np.nan
    elif kind == "inf":
        panel.loc[0, "value"] = np.inf
    else:
        panel.loc[0, "return"] = -1.0
    with pytest.raises(ValueError):
        validate_panel(panel)


def test_unbalanced_rejected(synthetic):
    with pytest.raises(ValueError, match="Unbalanced"):
        require_balanced_universe(synthetic.panel.iloc[1:])


def test_external_alpha_strict_merge(synthetic, tmp_path):
    path, signal = tmp_path / "panel.parquet", tmp_path / "alpha.csv"
    synthetic.panel.to_parquet(path)
    alpha = synthetic.panel[["date", "asset", "alpha"]].copy()
    alpha["alpha"] *= 2
    alpha.to_csv(signal, index=False)
    loaded = load_panel(path, signal)
    np.testing.assert_allclose(loaded["alpha"], alpha["alpha"])
    alpha.iloc[1:].to_csv(signal, index=False)
    with pytest.raises(ValueError, match="Missing"):
        load_panel(path, signal)
