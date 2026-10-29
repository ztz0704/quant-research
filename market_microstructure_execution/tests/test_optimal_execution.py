"""Conservation, optimality, risk trade-off, discrete timing and lot feasibility."""

from dataclasses import replace

import numpy as np
import pytest

from market_microstructure.optimal_execution.almgren_chriss import (
    ExecutionParameters,
    optimal_inventory,
    schedule_costs,
    simulate_shortfall,
)
from market_microstructure.optimal_execution.experiments import training_inputs
from market_microstructure.optimal_execution.schedules import comparison_schedules, round_lots


@pytest.fixture
def parameters():
    return ExecutionParameters(10000, 20, 20, 0.02, 0.00002, 0.000001, 0.01)


def test_zero_risk_twap_and_conservation(parameters):
    inventory = optimal_inventory(parameters, 0)
    np.testing.assert_allclose(-np.diff(inventory), 500)
    assert inventory[0] == 10000 and inventory[-1] == 0
    single = replace(parameters, intervals=1)
    np.testing.assert_array_equal(optimal_inventory(single, 1), [10000, 0])


def test_optimum_beats_feasible_alternatives_and_satisfies_gradient(parameters):
    risk = 0.001
    inventory = optimal_inventory(parameters, risk)
    others = comparison_schedules(parameters.order_size, parameters.intervals, np.arange(1, 21))
    optimal = schedule_costs(inventory, parameters, risk)["objective_cny"]
    assert all(optimal <= schedule_costs(x, parameters, risk)["objective_cny"] + 1e-8
               for x in others.values())
    gradient = parameters.adjusted_impact * (2 * inventory[1:-1] - inventory[:-2] - inventory[2:])
    gradient += risk * parameters.sigma**2 * parameters.tau * inventory[1:-1]
    np.testing.assert_allclose(gradient, 0, atol=1e-10)


def test_risk_aversion_frontloads_and_reduces_variance(parameters):
    costs = [schedule_costs(optimal_inventory(parameters, risk), parameters, risk)
             for risk in (0, 0.00001, 0.0001, 0.001)]
    assert np.all(np.diff([x["expected_cost_cny"] for x in costs]) >= 0)
    assert np.all(np.diff([x["cost_variance_cny2"] for x in costs]) <= 0)
    path = optimal_inventory(parameters, 0.001)
    assert (-np.diff(path) >= 0).all()
    assert path[1] < 9500


def test_price_path_monte_carlo_matches_analytic_moments(parameters):
    path = optimal_inventory(parameters, 0.0001)
    expected = schedule_costs(path, parameters)
    simulated = simulate_shortfall(path, parameters, paths=30000, seed=11)
    standard_error = expected["cost_std_cny"] / np.sqrt(len(simulated))
    assert abs(simulated.mean() - expected["expected_cost_cny"]) < 4 * standard_error
    assert simulated.var() == pytest.approx(expected["cost_variance_cny2"], rel=0.035)


def test_lot_rounding_conserves_parent_and_is_monotonic(parameters):
    rounded = round_lots(optimal_inventory(parameters, 0.001))
    assert rounded[-1] == 0 and rounded[0] == 10000
    assert (-np.diff(rounded) >= 0).all()
    assert (np.diff(rounded) % 100 == 0).all()
    with pytest.raises(ValueError):
        round_lots(np.array([10001, 0]))


def test_invalid_nonconvex_inputs_rejected(parameters):
    with pytest.raises(ValueError, match="convex"):
        optimal_inventory(replace(parameters, gamma=1), 1)
    with pytest.raises(ValueError):
        optimal_inventory(parameters, -1)
    with pytest.raises(ValueError):
        schedule_costs(np.array([10000, 0]), parameters)


def test_real_risk_inputs_ignore_future_book_and_flow(book):
    from market_microstructure.features.feature_pipeline import build_features

    features = build_features(book.loc[book.asset == book.asset.iloc[0]])
    cutoff = features.timestamp.iloc[49]
    settings = {"intervals": 20, "duration_seconds": 20}
    changed = features.copy()
    changed.loc[features.timestamp > cutoff, ["mid", "spread", "trade_size"]] *= 5
    first_audit, first_volume = training_inputs(features, cutoff, settings)
    second_audit, second_volume = training_inputs(changed, cutoff, settings)
    assert first_audit == second_audit
    np.testing.assert_array_equal(first_volume, second_volume)
