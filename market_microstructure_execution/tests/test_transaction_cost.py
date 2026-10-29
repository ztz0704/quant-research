"""Cash fees, rebates and proportional slippage costs."""

import pytest

from market_microstructure.costs.fees import transaction_fee
from market_microstructure.execution.slippage import slippage_per_share


def test_fee_signs_and_zero():
    assert transaction_fee(10, 100, 2) == pytest.approx(0.2)
    assert transaction_fee(-10, 100, 2) == pytest.approx(0.2)
    assert transaction_fee(10, 100, -0.5) == pytest.approx(-0.05)
    assert transaction_fee(0, 100, 100) == 0


def test_slippage_monotone_size():
    assert slippage_per_share(100, 10, 100, 0, 2) > slippage_per_share(100, 1, 100, 0, 2)
    assert slippage_per_share(100, 0, 0, 1, 1) == 0
