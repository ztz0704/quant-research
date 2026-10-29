import numpy as np
import pandas as pd
import pytest

from risk_aware_portfolio.backtest.open_events import at_time
from risk_aware_portfolio.portfolio.optimizer import optimize_portfolio
from risk_aware_portfolio.risk.open_model import OPEN_CONVENTION, fit_open_history, open_snapshot
from risk_aware_portfolio.utils.config import variant


def open_fixture(config):
    rng = np.random.default_rng(81)
    dates = pd.bdate_range("2020-01-02", periods=65)
    rows = []
    for t, date in enumerate(dates[:-2]):
        for i in range(40):
            values = rng.normal(size=len(config["risk_model"]["styles"]))
            rows.append(
                {
                    "date": date,
                    "asset": f"S{i:02}",
                    "eligible": True,
                    "industry": config["risk_model"]["industries"][i % 4],
                    "target": rng.normal(0, 0.01),
                    "target_valid": True,
                    "total_market_cap": 1e9 + i * 1e7,
                    "float_market_cap": 1e9,
                    "feature_asof": at_time(date, 18),
                    "decision_time": at_time(date, 18),
                    "label_start": dates[t + 1],
                    "label_end": dates[t + 2],
                    "label_available_at": at_time(dates[t + 2], 18),
                    "return_convention": OPEN_CONVENTION,
                    "return_unit": "decimal_simple",
                    **dict(zip(config["risk_model"]["styles"], values, strict=True)),
                }
            )
    return pd.DataFrame(rows), dates


@pytest.mark.parametrize("method", ["sample", "ewma"])
def test_open_covariance_dynamic_psd_maturity_and_optimizer_constraints(config, method):
    cfg = variant(config, risk_model={"covariance_method": method, "allow_specific_prior": True})
    panel, dates = open_fixture(cfg)
    # New security only becomes observable later; no balanced-universe requirement.
    panel = panel.loc[~(panel.asset.eq("S39") & panel.date.lt(dates[30]))]
    history = fit_open_history(panel, cfg)
    assets = history.exposures[dates[32]].index
    snapshot, diagnostic = open_snapshot(history, dates[32], assets, cfg)
    assert diagnostic["max_label_available"] <= at_time(dates[32], 18)
    assert "S39" in diagnostic["specific_prior_assets"]
    assert snapshot.covariance.index.equals(assets)
    assert snapshot.specific_variance.index.equals(assets)
    assert np.linalg.eigvalsh(snapshot.covariance).min() > 0
    alpha = pd.Series(np.linspace(-0.002, 0.002, len(assets)), index=assets)
    benchmark = pd.Series(1 / len(assets), index=assets)
    solution = optimize_portfolio(alpha, snapshot, benchmark * 0, benchmark, cfg, initial=True)
    assert solution.status.startswith("optimal")
    assert min(solution.slacks.values()) >= -cfg["portfolio"]["constraint_tolerance"]


@pytest.mark.parametrize("position", [25, 35, 45])
def test_multiple_asof_boundaries_ignore_immature_labels_and_future_assets(config, position):
    cfg = variant(config, risk_model={"allow_specific_prior": True})
    panel, dates = open_fixture(cfg)
    before = fit_open_history(panel, cfg)
    snapshot, _ = open_snapshot(
        before, dates[position], before.exposures[dates[position]].index, cfg
    )
    changed = panel.copy()
    future_labels = changed.label_available_at.gt(at_time(dates[position], 18))
    changed.loc[future_labels, "target"] = 0.8
    future_rows = changed.date.gt(dates[position])
    changed.loc[future_rows, cfg["risk_model"]["styles"]] *= 100
    changed.loc[future_rows, "total_market_cap"] *= 100
    after = fit_open_history(changed, cfg)
    revised, _ = open_snapshot(after, dates[position], after.exposures[dates[position]].index, cfg)
    np.testing.assert_array_equal(snapshot.covariance, revised.covariance)


def test_open_risk_rejects_close_convention_and_insufficient_history(config):
    panel, dates = open_fixture(config)
    with pytest.raises(ValueError, match="exact simple open"):
        fit_open_history(panel.assign(return_convention="close_to_next_close_simple"), config)
    history = fit_open_history(panel, config)
    with pytest.raises(ValueError, match="Insufficient matured"):
        open_snapshot(history, dates[3], history.exposures[dates[3]].index, config)
