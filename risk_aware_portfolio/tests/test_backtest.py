import numpy as np
import pandas as pd

from risk_aware_portfolio.backtest.engine import BacktestResult, run_backtest
from risk_aware_portfolio.utils.config import variant


def test_walk_forward_ledgers_and_costs(synthetic, config, tmp_path):
    result = run_backtest(synthetic.panel, config)
    daily = result.returns
    assert (daily["return_date"] > daily["decision_date"]).all()
    assert (daily.loc[~daily["rebalance"], "transaction_cost"] == 0).all()
    np.testing.assert_allclose(daily["nav"], (1 + daily["net_return"]).cumprod())
    attr = result.attribution
    np.testing.assert_allclose(
        attr["factor"] + attr["specific_total"] + attr["transaction_cost"],
        daily["net_return"],
        atol=1e-12,
    )
    result.save(tmp_path)
    reloaded = BacktestResult.load(result.name, tmp_path)
    pd.testing.assert_frame_equal(result.weights, reloaded.weights)


def test_drifted_holdings_are_used(synthetic, config):
    config = variant(config, backtest={"rebalance_frequency": "daily"})
    result = run_backtest(synthetic.panel, config, baseline=True)
    first = result.weights.iloc[0]
    return_date = result.returns.iloc[0]["return_date"]
    realized = synthetic.panel[synthetic.panel["date"].eq(return_date)].set_index("asset")["return"]
    expected = first * (1 + realized) / (1 + first @ realized)
    np.testing.assert_allclose(result.pretrade_weights.iloc[1], expected)
