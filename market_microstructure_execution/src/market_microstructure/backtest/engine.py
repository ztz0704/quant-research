"""Explicit event ordering, per-asset pending orders, latency and inventory limits."""

import logging
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from market_microstructure.backtest.accounting import Portfolio
from market_microstructure.costs.market_impact import temporary_impact
from market_microstructure.data.validation import validate_orderbook
from market_microstructure.execution.ashare_rules import constrained_quantity
from market_microstructure.execution.fills import FillParameters
from market_microstructure.execution.latency import Order
from market_microstructure.execution.maker import LiveLimit, maker_attempt
from market_microstructure.execution.taker import Fill, execute_mid, execute_taker
from market_microstructure.strategies.signal_strategy import order_quantity, target_inventory


@dataclass(frozen=True)
class BacktestConfig:
    """Fixed strategy and execution assumptions, in shares/bps/milliseconds."""

    mode: str = "taker"
    threshold: float = 0.20
    max_inventory: float = 10.0
    trade_size: float = 2.0
    latency_events: int | None = 1
    latency_ms: int | None = None
    taker_fee_bps: float = 2.0
    maker_fee_bps: float = -0.5
    fixed_slippage_bps: float = 0.1
    depth_slippage_bps: float = 1.0
    max_participation: float = 0.1
    initial_cash: float = 1e6
    liquidate_at_end: bool = True
    seed: int = 20261003
    sharpe_interval_ms: int = 1000
    annualization_periods: float | None = None
    impact_coefficient_bps: float = 0.0
    impact_exponent: float = 0.5
    maker_ttl_events: int = 10
    cancel_on_signal_change: bool = True
    fill_parameters: FillParameters = field(default_factory=FillParameters)
    market_type: str = "generic"
    lot_size: int = 1
    sell_tax_bps: float = 0.0
    transfer_bps: float = 0.0
    minimum_commission: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any], mode: str | None = None) -> "BacktestConfig":
        """Map YAML units explicitly rather than accepting ambiguous coefficients."""
        slippage = config["slippage"]
        return cls(mode=mode or config["execution"]["mode"],
                   threshold=config["strategy"]["entry_threshold"],
                   max_inventory=config["strategy"]["max_inventory"],
                   trade_size=config["strategy"]["trade_size"],
                   latency_events=config["execution"]["latency_events"],
                   latency_ms=config["execution"]["latency_ms"],
                   taker_fee_bps=config["fees"]["taker_bps"], maker_fee_bps=config["fees"]["maker_bps"],
                   fixed_slippage_bps=slippage["fixed_bps"] if slippage["enabled"] else 0,
                   depth_slippage_bps=slippage["depth_coefficient_bps"] if slippage["enabled"] else 0,
                   max_participation=config["execution"]["max_participation"],
                   initial_cash=config["backtest"]["initial_cash"],
                   liquidate_at_end=config["backtest"]["liquidate_at_end"], seed=config["seed"],
                   sharpe_interval_ms=config["backtest"]["sharpe_interval_ms"],
                   annualization_periods=config["backtest"]["annualization_periods"],
                   impact_coefficient_bps=config["market_impact"]["coefficient_bps"] if config["market_impact"]["enabled"] else 0,
                   impact_exponent=config["market_impact"]["exponent"],
                   maker_ttl_events=config["execution"]["maker_ttl_events"],
                   cancel_on_signal_change=config["execution"]["cancel_on_signal_change"],
                   fill_parameters=FillParameters(**config["maker"]),
                   market_type=config.get("market", {}).get("type", "generic"),
                   lot_size=config.get("market", {}).get("lot_size", 1),
                   sell_tax_bps=config["fees"].get("sell_tax_bps", 0),
                   transfer_bps=config["fees"].get("transfer_bps", 0),
                   minimum_commission=config["fees"].get("minimum_commission", 0))


@dataclass
class BacktestResult:
    """Every order, fill, inventory event and closed FIFO lot is inspectable."""

    events: pd.DataFrame
    fills: pd.DataFrame
    orders: pd.DataFrame
    closed_lots: pd.DataFrame
    config: BacktestConfig


def _fill(quantity: float, event: dict, config: BacktestConfig) -> Fill:
    """Execute against the quote observable at order arrival."""
    if config.mode == "mid":
        return execute_mid(quantity, (event["bid_price_1"] + event["ask_price_1"]) / 2)
    depth = event["ask_size_1"] if quantity > 0 else event["bid_size_1"]
    capped = min(abs(quantity), depth * config.max_participation)
    quote = event["ask_price_1"] if quantity > 0 else event["bid_price_1"]
    impact = temporary_impact(quote, capped, depth, config.impact_coefficient_bps,
                              config.impact_exponent) if capped else 0.0
    return execute_taker(quantity, event["bid_price_1"], event["ask_price_1"],
                         event["bid_size_1"], event["ask_size_1"], config.taker_fee_bps,
                         config.fixed_slippage_bps, config.depth_slippage_bps, config.max_participation,
                         impact_per_share=impact)


def run_backtest(book: pd.DataFrame, signals: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Sequential state is intentional; features and labels are vectorized upstream.

    Mark prior inventory, execute arrived old decisions, generate the current decision,
    then execute zero-latency decisions. At the known session end cancel pending orders
    and liquidate at the final quote, subject to displayed capacity.
    """
    if config.mode not in {"taker", "maker", "mid"} or config.market_type not in {"generic", "ashare_cash"}:
        raise ValueError("Unknown execution or market type")
    if config.market_type == "ashare_cash" and min(config.taker_fee_bps, config.maker_fee_bps) < 0:
        raise ValueError("A-share cash broker commission cannot be a simulated maker rebate")
    if (config.latency_events is None) == (config.latency_ms is None):
        raise ValueError("Configure exactly one latency clock")
    latency = config.latency_ms if config.latency_ms is not None else config.latency_events
    if latency < 0 or config.sharpe_interval_ms <= 0:
        raise ValueError("Invalid latency or Sharpe sampling interval")
    book, _ = validate_orderbook(book)
    if signals.duplicated(["timestamp", "asset"]).any():
        raise ValueError("Duplicate signals for an event")
    events = book.merge(signals[["timestamp", "asset", "signal"]], on=["timestamp", "asset"],
                        how="left", validate="one_to_one").sort_values(["timestamp", "asset"])
    portfolio = Portfolio(config.initial_cash)
    pending: dict[str, Order] = {}
    live_orders: dict[str, LiveLimit] = {}
    rng = np.random.default_rng(config.seed)
    commission_notional: dict[int, float] = {}
    commission_paid: dict[int, float] = {}
    sequence = {asset: -1 for asset in events.asset.unique()}
    counts = events.groupby("asset").size().to_dict()
    order_id = 0
    event_rows, fill_rows, order_rows = [], [], []
    logger = logging.getLogger("market_microstructure.backtest")

    def constrain(quantity: float, event: dict) -> float:
        if config.market_type == "generic":
            return quantity
        return constrained_quantity(quantity, event["asset"], event["timestamp"], portfolio,
                                    config.lot_size, event["ask_price_1"] * 1.01,
                                    config.taker_fee_bps + config.transfer_bps, config.minimum_commission)

    def apply(order: Order, event: dict, quantity: float, force_exit: bool = False,
              existing_fill: Fill | None = None) -> Fill:
        permitted = constrain(quantity, event)
        if existing_fill is not None:
            # Partial exchange fills may be odd lots; the submitted parent order was a whole lot.
            permitted = min(abs(quantity), abs(permitted)) * np.sign(quantity) if config.market_type == "generic" else quantity
            fill = existing_fill
        else:
            execution_config = replace(config, mode="taker") if force_exit and config.mode == "maker" else config
            fill = _fill(permitted, event, execution_config)
        if config.market_type == "ashare_cash" and fill.quantity:
            # Submitted buys are whole lots; individual partial fills are integer shares.
            shares = int(abs(fill.quantity))
            if fill.quantity > 0:
                affordable = int(max(0, portfolio.cash - config.minimum_commission)
                                 / (fill.price * (1 + (config.taker_fee_bps + config.transfer_bps) / 1e4)))
                shares = min(shares, affordable)
            else:
                today = event["timestamp"].tz_convert("Asia/Shanghai").date()
                settled = sum(lot.quantity for lot in portfolio.lots[order.asset]
                              if lot.quantity > 0 and lot.timestamp.tz_convert("Asia/Shanghai").date() < today)
                shares = min(shares, int(settled))
            ratio = shares / abs(fill.quantity)
            fill = replace(fill, quantity=np.sign(fill.quantity) * shares,
                           spread_cost=fill.spread_cost * ratio, fees=fill.fees * ratio,
                           slippage=fill.slippage * ratio, impact=fill.impact * ratio)
        if config.market_type == "ashare_cash" and fill.quantity and config.mode != "mid":
            notional = abs(fill.quantity) * fill.price
            commission_notional[order.order_id] = commission_notional.get(order.order_id, 0) + notional
            rate = config.maker_fee_bps if fill.liquidity == "maker" else config.taker_fee_bps
            commission = max(config.minimum_commission, commission_notional[order.order_id] * rate / 1e4)
            incremental = commission - commission_paid.get(order.order_id, 0)
            commission_paid[order.order_id] = commission
            other = notional * (config.transfer_bps + (config.sell_tax_bps if fill.quantity < 0 else 0)) / 1e4
            fill = replace(fill, fees=incremental + other)
        portfolio.apply_fill(order.asset, fill, event["timestamp"], sequence[order.asset])
        status = "filled" if fill.quantity else "market_rule_blocked" if permitted != quantity else "no_capacity"
        order_rows.append({"order_id": order.order_id, "asset": order.asset,
                           "timestamp": event["timestamp"], "status": status,
                           "requested_quantity": quantity, "executed_quantity": fill.quantity})
        if fill.quantity:
            logger.debug("fill asset=%s order_id=%s timestamp=%s %s", order.asset, order.order_id,
                         event["timestamp"], asdict(fill))
            fill_rows.append({"order_id": order.order_id, "asset": order.asset,
                              "timestamp": event["timestamp"], "decision_timestamp": order.decision_timestamp,
                              "decision_event": order.decision_event, "fill_event": sequence[order.asset],
                              "forced_exit": force_exit, **asdict(fill)})
        return fill

    def arrive(order: Order, event: dict) -> None:
        quantity = order_quantity(order.target, portfolio.positions[order.asset], config.trade_size,
                                  config.max_inventory)
        if config.mode != "maker":
            apply(order, event, quantity)
            return
        quantity = constrain(quantity, event)
        if not quantity:
            order_rows.append({"order_id": order.order_id, "asset": order.asset,
                               "timestamp": event["timestamp"], "status": "market_rule_blocked"})
            return
        side = "bid" if quantity > 0 else "ask"
        live_orders[order.asset] = LiveLimit(order, event[f"{side}_price_1"], quantity,
                                             event[f"{side}_size_1"], sequence[order.asset])
        order_rows.append({"order_id": order.order_id, "asset": order.asset,
                           "timestamp": event["timestamp"], "status": "maker_arrived"})

    # Array-backed iteration is required only for the path-dependent matching/accounting state.
    names = events.columns.tolist()
    for values in events.itertuples(index=False, name=None):
        event = dict(zip(names, values, strict=True))
        asset, timestamp = event["asset"], event["timestamp"]
        sequence[asset] += 1
        mid = (event["bid_price_1"] + event["ask_price_1"]) / 2
        portfolio.mark(asset, mid)
        current_target = target_inventory(event["signal"], config.threshold, config.max_inventory)
        if config.market_type == "ashare_cash":
            current_target = max(0.0, current_target)
        live = live_orders.get(asset)
        if live is not None:
            expired = sequence[asset] > live.placed_event + config.maker_ttl_events
            if expired:
                order_rows.append({"order_id": live.order.order_id, "asset": asset, "timestamp": timestamp,
                                   "status": "maker_expired"})
                del live_orders[asset]
            else:
                made, diagnostics = maker_attempt(live, event, sequence[asset], config.fill_parameters,
                                                   config.maker_fee_bps, rng, config.max_participation)
                order_rows.append({"order_id": live.order.order_id, "asset": asset, "timestamp": timestamp,
                                   "status": "maker_attempt", **diagnostics})
                if made is not None:
                    executed = apply(live.order, event, made.quantity, existing_fill=made)
                    live.quantity -= executed.quantity
                    if abs(live.quantity) < 1e-10:
                        del live_orders[asset]
                # Interval trades already happened before this snapshot's new signal.
                # A newly observed signal cannot cancel a fill in that past interval.
                if (asset in live_orders and config.cancel_on_signal_change
                        and current_target != live.order.target):
                    order_rows.append({"order_id": live.order.order_id, "asset": asset,
                                       "timestamp": timestamp, "status": "signal_cancel"})
                    del live_orders[asset]
        old = pending.get(asset)
        if old is not None and old.due(timestamp, sequence[asset]):
            arrive(old, event)
            del pending[asset]
        final = sequence[asset] == counts[asset] - 1
        if final:
            if asset in live_orders:
                cancelled_live = live_orders.pop(asset)
                order_rows.append({"order_id": cancelled_live.order.order_id, "asset": asset,
                                   "timestamp": timestamp, "status": "session_end_cancel"})
            if asset in pending:
                cancelled = pending.pop(asset)
                order_rows.append({"order_id": cancelled.order_id, "asset": asset,
                                   "timestamp": timestamp, "status": "session_end_cancel"})
            if config.liquidate_at_end and portfolio.positions[asset] != 0:
                order_id += 1
                liquidation = Order(order_id, asset, 0, timestamp, sequence[asset], latency_events=0)
                apply(liquidation, event, -portfolio.positions[asset], force_exit=True)
        elif asset not in pending and asset not in live_orders:
            target = current_target
            if abs(target - portfolio.positions[asset]) > 1e-10:
                order_id += 1
                order = Order(order_id, asset, target, timestamp, sequence[asset],
                              config.latency_events, config.latency_ms)
                order_rows.append({"order_id": order_id, "asset": asset, "timestamp": timestamp,
                                   "status": "submitted", "target": target})
                if order.due(timestamp, sequence[asset]):
                    arrive(order, event)
                else:
                    pending[asset] = order
        state = {"timestamp": timestamp, "asset": asset, "event": sequence[asset],
                 "signal": event["signal"], "mid": mid, "position": portfolio.positions[asset],
                 "inventory_risk": (portfolio.positions[asset] * mid
                                    * event.get("realized_vol_bps", 0) / 1e4) ** 2,
                 **portfolio.snapshot()}
        event_rows.append(state)
        logger.debug("event %s", state)
    fills = pd.DataFrame(fill_rows)
    if fills.empty:
        fills = pd.DataFrame(columns=["asset", "timestamp", "quantity", "price", "mid", "spread_cost",
                                      "fees", "slippage", "impact", "liquidity", "forced_exit"])
    return BacktestResult(pd.DataFrame(event_rows), fills, pd.DataFrame(order_rows),
                          pd.DataFrame(portfolio.closed_lots), config)
