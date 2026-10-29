import json

import numpy as np
import pandas as pd
import pytest
from test_open_risk_v3 import open_fixture

from risk_aware_portfolio.backtest.open_events import INTERFACE_VERSION, at_time
from risk_aware_portfolio.backtest.research_v3 import (
    performance,
    run_bundle,
    run_portfolios,
    validate_signals,
)
from risk_aware_portfolio.utils.config import variant


def integrated_fixture(config):
    cfg = variant(config, risk_model={"allow_specific_prior": True}, portfolio={"min_weight": 1e-6})
    panel, dates = open_fixture(cfg)
    market_dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    calendar = pd.DataFrame({"date": market_dates, "exchange": "SSE"})
    rng = np.random.default_rng(941)
    prices = 10 * np.exp(np.cumsum(rng.normal(0, 0.008, size=(len(market_dates), 40)), axis=0))
    panel["raw_open"] = prices.ravel()
    panel["raw_close"] = panel.raw_open
    panel["research_open"] = panel.raw_open
    panel["exchange"], panel["adj_factor"] = "SSE", 1.0
    panel["buyable"], panel["sellable"], panel["universe_member"] = True, True, True
    panel["buy_fill_fraction"], panel["sell_fill_fraction"] = 1.0, 1.0
    panel["action_complete"], panel["delisted"] = False, False
    for name in ("execution_asof", "open_price_asof", "action_indicator_asof"):
        panel[name] = panel.date.map(lambda d: at_time(d, 9, 30))
    panel["close_price_asof"] = panel.date.map(lambda d: at_time(d, 18))
    indexed = panel.set_index(["date", "asset"]).raw_open
    panel["target"] = [
        indexed.get((r.label_end, r.asset), np.nan) / indexed.get((r.label_start, r.asset), np.nan)
        - 1
        for r in panel.itertuples()
    ]
    panel["target_valid"] = panel.target.notna()
    decision_dates = dates[32:38]
    signals = []
    for date in decision_dates:
        for model in ("ridge", "boosting"):
            for i in range(40):
                score = (i - 20) * 0.00005 * (1 if model == "ridge" else -0.3)
                signals.append(
                    {
                        "date": date,
                        "asset": f"S{i:02}",
                        "model": model,
                        "score": score,
                        "expected_return": score,
                        "rev": -score,
                        "feature_asof": at_time(date, 18),
                        "decision_time": at_time(date, 18),
                        "model_fit_cutoff": dates[32],
                        "calibration_fit_cutoff": dates[32],
                        "estimator_fit_cutoff": dates[29],
                        "max_fit_label_available": dates[28],
                        "max_calibration_label_available": dates[31],
                        "model_version": "known_oracle",
                        "return_convention": panel.return_convention.iloc[0],
                        "return_unit": "decimal_simple",
                    }
                )
    settings = {
        "portfolio": {"quantile": 0.2, "cost_bps_per_full_l1": [10]},
        "event_portfolio": {"initial_cash": 1000.0, "terminal_liquidation": True},
    }
    return cfg, panel, calendar, pd.DataFrame(signals), decision_dates, settings


def test_all_strategies_risk_optimization_actual_fills_and_reconciliation(config):
    cfg, panel, calendar, signals, dates, settings = integrated_fixture(config)
    tables = run_portfolios(panel, calendar, signals, dates, cfg, settings)
    performance_table = tables["portfolio_performance"].set_index("strategy")
    assert set(performance_table.index) == {
        "A_REV",
        "B_Ridge",
        "C_HGB",
        "D_HGB_risk",
        "E_HGB_risk_cost",
        "R_Ridge_risk_cost",
        "Benchmark_equal",
    }
    assert performance_table.valid_return_dates.eq(len(dates)).all()
    assert performance_table.terminal_unliquidated_assets.eq(0).all()
    assert performance_table.loc["D_HGB_risk", "optimized_decisions"] == len(dates)
    assert performance_table.loc["E_HGB_risk_cost", "optimized_decisions"] == len(dates)
    ledger = tables["portfolio_events"]
    np.testing.assert_allclose(ledger.post_nav, ledger.pre_nav - ledger.fees, atol=1e-8)
    assert ledger.cash.min() >= -1e-8
    attr = tables["portfolio_attribution"]
    assert attr.status.eq("reconciled_with_explicit_basis").all()
    np.testing.assert_allclose(
        attr.factor_return + attr.specific_return + attr.basis_return - attr.cost,
        attr.net_return,
        atol=1e-12,
    )
    assert attr.basis_return.abs().max() < 1e-10
    assert tables["actual_fill_constraint_audit"].status.notna().all()


@pytest.mark.parametrize("cut_position", [0, 2, 4])
def test_future_asset_price_state_and_label_changes_preserve_all_past_targets(config, cut_position):
    cfg, panel, calendar, signals, dates, settings = integrated_fixture(config)
    before = run_portfolios(panel, calendar, signals, dates, cfg, settings)
    changed = panel.copy()
    future = changed.date.gt(dates[cut_position]) & changed.asset.isin(["S00", "S39"])
    changed.loc[future, ["raw_open", "raw_close", "research_open"]] *= 8
    changed.loc[future, ["buyable", "sellable", "universe_member", "eligible"]] = False
    # Recompute corrupted future outcomes honestly; never change matured history.
    indexed = changed.set_index(["date", "asset"]).research_open
    changed["target"] = [
        indexed.get((r.label_end, r.asset), np.nan) / indexed.get((r.label_start, r.asset), np.nan)
        - 1
        for r in changed.itertuples()
    ]
    changed_signals = signals.loc[
        ~(signals.date.gt(dates[cut_position]) & signals.asset.isin(["S00", "S39"]))
    ]
    after = run_portfolios(changed, calendar, changed_signals, dates, cfg, settings)
    fields = ["date", "asset", "weight", "strategy", "cost_bps_per_full_l1"]
    a = (
        before["portfolio_targets"]
        .loc[lambda f: f.date.le(dates[cut_position]), fields]
        .reset_index(drop=True)
    )
    b = (
        after["portfolio_targets"]
        .loc[lambda f: f.date.le(dates[cut_position]), fields]
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(a, b)


def test_infeasible_optimizer_and_missing_window_are_explicit(config):
    cfg, panel, calendar, signals, dates, settings = integrated_fixture(config)
    cfg["portfolio"]["max_weight"] = 0.001
    result = run_portfolios(panel, calendar, signals, dates, cfg, settings)
    optimized = result["portfolio_decisions"].strategy.str.contains("risk")
    assert result["portfolio_decisions"].loc[optimized, "status"].eq("BLOCKED").all()
    assert result["portfolio_decisions"].loc[optimized, "constraints_relaxed"].eq(False).all()
    sample = result["portfolio_periods"].loc[lambda f: f.strategy.eq("C_HGB")].copy()
    sample.loc[sample.index[1], ["net_return", "nav"]] = np.nan
    sample.loc[sample.index[1], "nav_valid"] = False
    summary = performance(sample, sample.set_index("date").net_return)
    assert summary["performance_status"] == "BLOCKED_incomplete_window"
    assert np.isnan(summary["sharpe"]) and np.isnan(summary["cagr"])


def test_bridge_schema_version_hash_and_future_signal_timestamp(config, tmp_path):
    _, panel, _, signals, dates, _ = integrated_fixture(config)
    with pytest.raises(ValueError, match="schema"):
        validate_signals(signals.drop(columns="expected_return"), panel, dates)
    changed = signals.copy()
    changed.loc[0, "feature_asof"] = dates[-1]
    with pytest.raises(ValueError, match="provenance"):
        validate_signals(changed, panel, dates)
    (tmp_path / "contract.json").write_text(
        json.dumps({"interface_version": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="version"):
        run_bundle(tmp_path)
    (tmp_path / "contract.json").write_text(
        json.dumps({"interface_version": INTERFACE_VERSION, "inputs": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash contract"):
        run_bundle(tmp_path)
