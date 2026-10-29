from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from risk_aware_portfolio.backtest.engine import run_backtest
from risk_aware_portfolio.backtest.open_diagnostics import rank_open_diagnostic
from risk_aware_portfolio.portfolio.optimizer import optimize_portfolio
from risk_aware_portfolio.preprocessing.neutralization import preprocess_alpha
from risk_aware_portfolio.risk.model import estimate_risk_history
from risk_aware_portfolio.utils.config import variant


def test_close_risk_engine_rejects_open_interval_before_estimating(synthetic, config):
    panel = synthetic.panel.assign(
        return_convention="close_decision_next_open_to_following_open_simple"
    )
    with pytest.raises(ValueError, match="cannot trade open"):
        run_backtest(panel, config)


def test_calibrated_expected_returns_preserve_return_scale(synthetic, config):
    cfg = variant(config, alpha={"input_mode": "expected_return", "winsorize": False})
    section = synthetic.panel.loc[synthetic.panel.date.eq(synthetic.panel.date.min())]
    result = preprocess_alpha(section, cfg)
    np.testing.assert_array_equal(result.selected_alpha, section.alpha)
    with pytest.raises(ValueError, match="clipped"):
        preprocess_alpha(section, variant(cfg, alpha={"winsorize": True}))


def test_optimizer_rejects_permuted_covariance_labels(synthetic, config):
    snapshot = next(iter(estimate_risk_history(synthetic.panel, config).snapshots.values()))
    wrong = replace(snapshot, covariance=snapshot.covariance.iloc[::-1, ::-1])
    alpha = pd.Series(0.0, index=snapshot.exposures.index)
    previous = alpha.copy()
    benchmark = pd.Series(1 / len(alpha), index=alpha.index)
    with pytest.raises(ValueError, match="label alignment"):
        optimize_portfolio(alpha, wrong, previous, benchmark, config, initial=True)


def open_sample():
    dates = pd.bdate_range("2020-01-01", periods=6)
    return pd.DataFrame(
        [
            {
                "date": date,
                "asset": asset,
                "score": i,
                "label_start": dates[t + 1],
                "label_end": dates[t + 2],
                "target": 0.01,
                "target_valid": True,
                "entry_buyable": True,
                "entry_sellable": True,
                "entry_raw_open": 10.0,
                "exit_sellable": True,
                "return_convention": "close_decision_next_open_to_following_open_simple",
            }
            for t, date in enumerate(dates[:4])
            for i, asset in enumerate(["A", "B"])
        ]
    )


def test_open_diagnostic_self_financing_full_l1_costs():
    ledger, weights = rank_open_diagnostic(open_sample(), "score", 100, 0.5)
    assert weights.groupby("date").weight.sum().eq(1).all()
    assert ledger.capital_ratio.iloc[0] == pytest.approx(1 / 1.01)
    assert ledger.cost.iloc[0] == pytest.approx(0.01 / 1.01)
    assert ledger.net_return.iloc[0] == pytest.approx(0.0)
    assert ledger.nav.iloc[0] == pytest.approx(1.0)


def test_unfillable_entry_stops_nav_without_changing_target_or_cash_reset():
    before, original = rank_open_diagnostic(open_sample(), "score", 5, 0.5)
    changed = open_sample()
    first = changed.date.min()
    changed.loc[changed.date.eq(first) & changed.asset.eq("B"), "entry_buyable"] = False
    after, actual = rank_open_diagnostic(changed, "score", 5, 0.5)
    pd.testing.assert_frame_equal(original, actual)
    assert before.nav.notna().all()
    assert after.nav.isna().all()
    assert after.cost.isna().all()


@pytest.mark.parametrize("unavailable_sell_state", [False, None])
def test_eligibility_exit_uses_full_quote_panel_for_real_liquidation(unavailable_sell_state):
    panel = open_sample()
    panel["entry_sellable"] = True
    panel["entry_raw_open"] = 10.0
    predictions = panel.copy()
    dates = sorted(panel.date.unique())
    # B was held on the first date, then leaves the feature universe. Its actual
    # opening quote remains in the source panel, so liquidation is evaluable.
    predictions = predictions.loc[~(predictions.date.ge(dates[1]) & predictions.asset.eq("B"))]
    ledger, targets = rank_open_diagnostic(predictions, "score", 5, 0.5, outcome_panel=panel)
    assert ledger.nav.notna().all()
    assert targets.loc[targets.date.ge(dates[1])].asset.eq("A").all()
    panel["entry_sellable"] = panel.entry_sellable.astype("boolean")
    panel.loc[panel.date.eq(dates[1]) & panel.asset.eq("B"), "entry_sellable"] = (
        unavailable_sell_state
    )
    blocked, _ = rank_open_diagnostic(predictions, "score", 5, 0.5, outcome_panel=panel)
    assert blocked.nav.iloc[1:].isna().all()
