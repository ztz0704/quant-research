"""Small deterministic fixtures; tests do not reuse research result artifacts."""

import pytest

from risk_aware_portfolio.data.synthetic import generate_synthetic
from risk_aware_portfolio.utils.config import load_config, variant


@pytest.fixture
def config():
    return variant(
        load_config(),
        synthetic={"n_assets": 40, "n_dates": 55},
        risk_model={"lookback": 25, "min_observations": 15},
        portfolio={"max_weight": 0.08, "style_tolerance": 0.5},
    )


@pytest.fixture
def synthetic(config):
    return generate_synthetic(config)
