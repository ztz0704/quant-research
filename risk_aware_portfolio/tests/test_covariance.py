import numpy as np
import pandas as pd
import pytest

from risk_aware_portfolio.diagnostics.risk_diagnostics import compare_covariance_estimators
from risk_aware_portfolio.risk.covariance import ensure_psd
from risk_aware_portfolio.risk.factor_covariance import estimate_covariance
from risk_aware_portfolio.risk.model import estimate_risk_history
from risk_aware_portfolio.risk.specific_risk import estimate_specific_variance


@pytest.mark.parametrize("method", ["sample", "ewma"])
def test_covariance_shape_psd(synthetic, config, method):
    factors = synthetic.factor_returns
    cov = estimate_covariance(factors, method)
    np.testing.assert_allclose(cov, cov.T, atol=1e-14)
    assert cov.shape == (9, 9)
    assert np.linalg.eigvalsh(cov).min() >= 0
    values = estimate_specific_variance(factors, method)
    assert (values > 0).all()


def test_full_covariance_identity(synthetic, config):
    snapshot = next(iter(estimate_risk_history(synthetic.panel, config).snapshots.values()))
    x, f, d = snapshot.exposures, snapshot.factor_covariance, snapshot.specific_variance
    np.testing.assert_allclose(
        snapshot.covariance, x.to_numpy() @ f.to_numpy() @ x.to_numpy().T + np.diag(d), atol=1e-14
    )
    assert np.linalg.eigvalsh(snapshot.covariance).min() > 0


def test_repair_logged_and_audited(caplog):
    matrix, audit = ensure_psd(np.array([[1.0, 2.0], [2.0, 1.0]]), 1e-8)
    assert audit.repaired
    assert audit.before.min() < 0
    assert audit.after.min() == 1e-8
    assert "PSD repair" in caplog.text
    assert np.linalg.eigvalsh(matrix).min() > 0


def test_material_asymmetry_rejected():
    with pytest.raises(ValueError, match="asymmetric"):
        ensure_psd(np.array([[1.0, 1.0], [0.0, 1.0]]))


def test_sample_reference():
    values = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 5.0]})
    np.testing.assert_allclose(estimate_covariance(values, "sample"), values.cov())


def test_estimator_comparison_uses_known_covariance(synthetic, config):
    comparison = compare_covariance_estimators(
        synthetic.panel, config, synthetic.factor_covariance
    ).set_index("method")
    assert comparison.loc["sample", "n_forecasts"] == comparison.loc["ewma", "n_forecasts"]
    assert (
        comparison.loc["sample", "mean_effective_observations"]
        > comparison.loc["ewma", "mean_effective_observations"]
    )
    assert (comparison["minimum_asset_eigenvalue"] > 0).all()
    assert (comparison["mean_relative_factor_covariance_error"] > 0).all()
