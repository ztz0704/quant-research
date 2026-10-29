"""Small deterministic book fixtures."""

import pytest

from market_microstructure.data.synthetic import SyntheticConfig, generate_orderbook


@pytest.fixture
def book():
    """Interleaved assets expose cross-asset rolling/label mistakes."""
    return generate_orderbook(SyntheticConfig(events=120, levels=5), seed=7)
