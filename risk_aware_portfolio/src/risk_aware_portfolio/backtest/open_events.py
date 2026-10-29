"""V3 raw-share/cash event ledger. Unknown marks never erase inventory or cash.

Orders are share quantities frozen after close, not weights recomputed at the
next open. Sell first, then pro-rata cash-constrained buys. A new decision may
explicitly cancel/replace a stale order; absent/unvalued decisions retain it.
Daily corporate actions require actual cash/share events, not adjustment ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from risk_aware_portfolio.backtest.accounting import account_period
from risk_aware_portfolio.costs.transaction_cost import TransactionCost

INTERFACE_VERSION = "ashare-open-events-v3"


def at_time(date, hour: int, minute: int = 0) -> pd.Timestamp:
    """Construct local wall time without NumPy generic-timedelta conversion."""
    return pd.Timestamp(datetime.combine(pd.Timestamp(date).date(), time(hour, minute)))


def known_true(value) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


@dataclass
class OpenOrder:
    asset: str
    remaining: float
    created: pd.Timestamp
    scheduled: pd.Timestamp


@dataclass
class OpenState:
    cash: float
    quantities: dict[str, float] = field(default_factory=dict)
    orders: dict[str, OpenOrder] = field(default_factory=dict)
    last_adjustment: dict[str, float] = field(default_factory=dict)
    uncertain_actions: set[str] = field(default_factory=set)
    applied_action_ids: set[str] = field(default_factory=set)
    unsettled: dict[str, float] = field(default_factory=dict)
    trade_count: int = 0


@dataclass
class OpenEventResult:
    events: pd.DataFrame
    holdings: pd.DataFrame
    orders: pd.DataFrame
    targets: pd.DataFrame
    decisions: pd.DataFrame
    periods: pd.DataFrame
    final_state: OpenState


def validate_event_input(panel: pd.DataFrame, calendar: pd.DataFrame) -> None:
    if {"date", "exchange"} - set(calendar) or calendar.empty:
        raise ValueError("Event calendar schema is empty or incomplete")
    required = {
        "date",
        "asset",
        "exchange",
        "raw_open",
        "raw_close",
        "adj_factor",
        "universe_member",
        "buyable",
        "sellable",
        "execution_asof",
        "open_price_asof",
        "close_price_asof",
        "action_indicator_asof",
    }
    if required - set(panel):
        raise ValueError(f"Event input missing fields: {sorted(required - set(panel))}")
    if panel.duplicated(["date", "asset"]).any() or calendar.duplicated(["date", "exchange"]).any():
        raise ValueError("Duplicate event/calendar keys")
    if not pd.api.types.is_datetime64_any_dtype(panel.date):
        raise ValueError("Event dates must be parsed explicitly")
    for column in (
        "execution_asof",
        "open_price_asof",
        "close_price_asof",
        "action_indicator_asof",
    ):
        if (
            not pd.api.types.is_datetime64_any_dtype(panel[column])
            or panel[column].dt.tz is not None
        ):
            raise ValueError("Event timestamp fields must be parsed local datetime64")
    if (
        not pd.api.types.is_datetime64_any_dtype(calendar.date)
        or calendar.date.dt.tz is not None
        or not calendar.date.eq(calendar.date.dt.normalize()).all()
    ):
        raise ValueError("Event calendar must use normalized local dates")
    if panel.groupby("asset").exchange.nunique().gt(1).any():
        raise ValueError("Event security identifier changes exchange")
    if not panel.date.eq(panel.date.dt.normalize()).all():
        raise ValueError("Event dates must be normalized")
    for column in ("raw_open", "raw_close", "adj_factor"):
        if np.isinf(pd.to_numeric(panel[column], errors="raise")).any():
            raise ValueError("Infinite event price/factor")
    for column in ("buyable", "sellable", "universe_member"):
        if not panel[column].dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError("Event states must be booleans or null")
    for column in ("buy_fill_fraction", "sell_fill_fraction"):
        if column in panel and not panel[column].dropna().between(0, 1).all():
            raise ValueError("Fill fractions must be in [0,1]")
    if (
        not pd.MultiIndex.from_frame(panel[["date", "exchange"]])
        .isin(pd.MultiIndex.from_frame(calendar[["date", "exchange"]]))
        .all()
    ):
        raise ValueError("Event quote outside independent exchange calendar")


def _mark(state: OpenState, rows: pd.DataFrame, field_name: str) -> tuple[float, str]:
    value, reasons = state.cash, []
    for asset, quantity in state.quantities.items():
        if quantity <= 1e-10:
            continue
        if asset in state.uncertain_actions:
            reasons.append(f"{asset}:unresolved_corporate_action")
            continue
        if asset not in rows.index:
            reasons.append(f"{asset}:missing_row")
            continue
        row = rows.loc[asset]
        if known_true(row.get("delisted")):
            reasons.append(f"{asset}:delisted_without_terminal_settlement")
            continue
        price = row[field_name]
        if not pd.notna(price) or price <= 0:
            reasons.append(f"{asset}:missing_{field_name}")
            continue
        if field_name == "raw_open" and (
            pd.isna(row.open_price_asof) or row.open_price_asof > at_time(row.date, 9, 30)
        ):
            reasons.append(f"{asset}:late_or_unknown_open_mark")
            continue
        if field_name == "raw_close" and (
            pd.isna(row.close_price_asof) or row.close_price_asof > at_time(row.date, 18)
        ):
            reasons.append(f"{asset}:late_or_unknown_close_mark")
            continue
        value += quantity * price
    return (np.nan, ";".join(reasons)) if reasons else (float(value), "")


def _actions(state: OpenState, rows: pd.DataFrame, date: pd.Timestamp, log: list) -> float:
    cash_flow = 0.0
    for asset, row in rows.iterrows():
        quantity = state.quantities.get(asset, 0.0)
        notice = row.action_indicator_asof
        if quantity > 1e-10 and (pd.isna(notice) or notice > at_time(date, 9, 30)):
            # Do not read a future adjustment indicator to decide an opening
            # fill. Preserve inventory and quarantine until a complete bridge.
            state.uncertain_actions.add(asset)
            log.append(
                {
                    "date": date,
                    "asset": asset,
                    "event": "action_indicator_unavailable_at_open",
                    "quantity": quantity,
                    "cash": state.cash,
                }
            )
            continue
        factor = row.adj_factor
        previous = state.last_adjustment.get(asset, factor)
        changed = (
            pd.notna(factor) and pd.notna(previous) and not np.isclose(factor, previous, rtol=1e-12)
        )
        complete = known_true(row.get("action_complete"))
        action_id = row.get("action_id")
        identifier = f"{asset}:{action_id}" if pd.notna(action_id) and str(action_id) else None
        reported = complete and (
            identifier is not None
            or row.get("cash_dividend", 0) > 0
            or (pd.notna(row.get("split_ratio")) and row.get("split_ratio") != 1)
        )
        new_action = reported and identifier not in state.applied_action_ids
        action_time = row.get("action_asof", pd.NaT)
        timely = pd.notna(action_time) and action_time <= at_time(date, 9, 30)
        if quantity > 1e-10 and (changed or asset in state.uncertain_actions or new_action):
            from_factor = row.get("reconciliation_from_factor", np.nan)
            reconciles_pending = asset not in state.uncertain_actions or (
                pd.notna(from_factor)
                and pd.notna(previous)
                and np.isclose(from_factor, previous, rtol=1e-12)
            )
            if not (
                complete
                and timely
                and identifier is not None
                and identifier not in state.applied_action_ids
                and reconciles_pending
            ):
                state.uncertain_actions.add(asset)
                log.append(
                    {
                        "date": date,
                        "asset": asset,
                        "event": "corporate_action_unknown",
                        "quantity": quantity,
                        "cash": state.cash,
                    }
                )
                continue
            ratio, dividend = row.get("split_ratio", np.nan), row.get("cash_dividend", np.nan)
            if not np.isfinite([ratio, dividend]).all() or ratio <= 0 or dividend < 0:
                raise ValueError(
                    "Certified action needs explicit split ratio and cash per old share"
                )
            cash_flow += quantity * dividend
            state.cash += quantity * dividend
            state.quantities[asset] = quantity * ratio
            if asset in state.orders:
                state.orders[asset].remaining *= ratio
            state.uncertain_actions.discard(asset)
            state.applied_action_ids.add(identifier)
            log.append(
                {
                    "date": date,
                    "asset": asset,
                    "event": "corporate_action_applied",
                    "quantity": quantity * ratio,
                    "cash": state.cash,
                }
            )
        if pd.notna(factor):
            state.last_adjustment[asset] = factor
        if quantity > 1e-10 and known_true(row.get("delisted")):
            terminal = row.get("delisting_cash_per_share", np.nan)
            if (
                known_true(row.get("delisting_complete"))
                and timely
                and pd.notna(terminal)
                and terminal >= 0
            ):
                amount = state.quantities.get(asset, 0.0) * terminal
                state.cash += amount
                cash_flow += amount
                state.quantities[asset] = 0.0
                state.orders.pop(asset, None)
                state.uncertain_actions.discard(asset)
                log.append(
                    {
                        "date": date,
                        "asset": asset,
                        "event": "delisting_settlement",
                        "quantity": 0.0,
                        "cash": state.cash,
                    }
                )
            else:
                log.append(
                    {
                        "date": date,
                        "asset": asset,
                        "event": "delisted_unknown_terminal",
                        "quantity": quantity,
                        "cash": state.cash,
                    }
                )
    return cash_flow


def _fee(amount: float, reference_nav: float, cost: TransactionCost) -> float:
    if cost.impact_coefficient and not np.isfinite(reference_nav):
        raise ValueError("Impact fee needs a reliable contemporaneous NAV")
    scale = reference_nav if np.isfinite(reference_nav) and reference_nav > 0 else 1.0
    return cost.realized(np.array([amount / scale])) * scale


def _execute(
    state: OpenState,
    rows: pd.DataFrame,
    date: pd.Timestamp,
    cost: TransactionCost,
    reference_nav: float,
    log: list,
) -> tuple[float, float]:
    fees, traded = 0.0, 0.0
    buys = []
    for asset, order in sorted(list(state.orders.items())):
        if date < order.scheduled or abs(order.remaining) < 1e-10:
            continue
        if asset not in rows.index:
            log.append(
                {
                    "date": date,
                    "asset": asset,
                    "event": "rejected_missing_row",
                    "remaining": order.remaining,
                }
            )
            continue
        row = rows.loc[asset]
        now = at_time(date, 9, 30)
        buying = order.remaining > 0
        side = "buy" if buying else "sell"
        reason = ""
        if asset in state.uncertain_actions:
            reason = "corporate_action_unknown"
        elif known_true(row.get("delisted")):
            reason = "delisted"
        elif pd.isna(row.raw_open) or row.raw_open <= 0:
            reason = "missing_price"
        elif pd.isna(row.open_price_asof) or row.open_price_asof > now:
            reason = "late_price"
        elif pd.isna(row.execution_asof) or row.execution_asof > now:
            reason = "late_or_unknown_execution_state"
        elif not known_true(row.buyable if buying else row.sellable):
            reason = f"{side}_not_executable_or_unknown"
        elif buying and not known_true(row.universe_member):
            reason = "not_eligible_for_new_buy"
        fraction = row.get(f"{side}_fill_fraction", np.nan)
        if not reason and (pd.isna(fraction) or fraction <= 0):
            reason = "no_fill_evidence_or_zero_fill"
        if reason:
            log.append(
                {
                    "date": date,
                    "asset": asset,
                    "event": "rejected_" + reason,
                    "side": side,
                    "remaining": order.remaining,
                    "cash": state.cash,
                }
            )
            continue
        quantity = abs(order.remaining) * fraction
        if buying:
            buys.append((asset, quantity, float(row.raw_open)))
            continue
        available = max(0.0, state.quantities.get(asset, 0.0) - state.unsettled.get(asset, 0.0))
        quantity = min(quantity, available)
        amount = quantity * row.raw_open
        charge = _fee(-amount, reference_nav, cost)
        state.cash += amount - charge
        state.quantities[asset] = state.quantities.get(asset, 0.0) - quantity
        order.remaining += quantity
        fees += charge
        traded += amount
        state.trade_count += int(quantity > 0)
        log.append(
            {
                "date": date,
                "asset": asset,
                "event": "sell_filled" if quantity > 0 else "rejected_unsettled_inventory",
                "quantity": quantity,
                "price": row.raw_open,
                "fee": charge,
                "remaining": order.remaining,
                "cash": state.cash,
            }
        )
    if buys:

        def funding(fraction):
            return sum(
                fraction * q * p + _fee(fraction * q * p, reference_nav, cost) for _, q, p in buys
            )

        # Round-off in an exhausted cash balance must not create a root outside
        # [0,1]. Preserve the recorded cash; a nonpositive balance funds no buy.
        budget = max(0.0, state.cash)
        ratio = (
            0.0
            if budget == 0
            else (
                1.0
                if funding(1.0) <= budget
                else brentq(lambda r: funding(r) - budget, 0.0, 1.0, xtol=1e-14)
            )
        )
        for asset, requested, price in buys:
            quantity = requested * ratio
            amount = quantity * price
            charge = _fee(amount, reference_nav, cost)
            before = state.cash
            state.cash -= amount + charge
            state.quantities[asset] = state.quantities.get(asset, 0.0) + quantity
            state.unsettled[asset] = state.unsettled.get(asset, 0.0) + quantity
            state.orders[asset].remaining -= quantity
            fees += charge
            traded += amount
            state.trade_count += int(quantity > 0)
            if not np.isclose(before - amount - charge, state.cash, atol=1e-8):
                raise AssertionError("Trade cash identity failed")
            log.append(
                {
                    "date": date,
                    "asset": asset,
                    "event": "buy_filled" if ratio == 1 else "buy_cash_limited",
                    "quantity": quantity,
                    "requested": requested,
                    "price": price,
                    "fee": charge,
                    "remaining": state.orders[asset].remaining,
                    "cash": state.cash,
                }
            )
    if state.cash < -1e-7 or any(q < -1e-7 for q in state.quantities.values()):
        raise AssertionError("Self financing / long-only inventory violated")
    for asset in list(state.orders):
        if abs(state.orders[asset].remaining) < 1e-8:
            state.orders.pop(asset)
    return float(fees), float(traded)


def _plan(
    state: OpenState,
    weights: pd.Series,
    rows: pd.DataFrame,
    date: pd.Timestamp,
    next_sessions: dict,
    cost: TransactionCost,
    log: list,
) -> str:
    if weights.index.has_duplicates or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Targets must have unique assets and finite long-only weights")
    if not np.isclose(weights.sum(), 1.0, atol=1e-8) and weights.sum() != 0:
        raise ValueError("Target weights must sum to one or explicitly liquidate")
    nav, _ = _mark(state, rows, "raw_close")
    assets = weights.index.union(pd.Index(state.quantities)).union(pd.Index(state.orders))
    priced_assets = {
        a for a in assets if weights.get(a, 0.0) > 0 or state.quantities.get(a, 0.0) > 1e-10
    }
    complete = np.isfinite(nav) and all(
        a in rows.index
        and pd.notna(rows.loc[a, "raw_close"])
        and rows.loc[a, "raw_close"] > 0
        and pd.notna(rows.loc[a, "close_price_asof"])
        and rows.loc[a, "close_price_asof"] <= at_time(date, 18)
        for a in priced_assets
    )
    ratio = 1.0
    if complete and weights.sum() > 0:
        current = weights.reindex(assets, fill_value=0).to_numpy()
        pretrade = np.array(
            [
                state.quantities[a] * rows.loc[a, "raw_close"] / nav
                if state.quantities.get(a, 0.0) > 1e-10
                else 0.0
                for a in assets
            ]
        )
        ratio = account_period(current, pretrade, np.zeros(len(assets)), cost).capital_ratio
    for asset in assets:
        desired_weight = weights.get(asset, 0.0)
        if desired_weight > 0 and not complete:
            continue  # Keep existing share orders; do not invent a NAV-dependent order.
        desired = (
            ratio * nav * desired_weight / rows.loc[asset, "raw_close"]
            if desired_weight > 0
            else 0.0
        )
        remaining = desired - state.quantities.get(asset, 0.0)
        old = state.orders.get(asset)
        if old and np.isclose(old.remaining, remaining, atol=1e-8):
            log.append(
                {
                    "date": date,
                    "asset": asset,
                    "event": "pending_order_retained",
                    "remaining": remaining,
                }
            )
            continue
        if old:
            log.append(
                {
                    "date": date,
                    "asset": asset,
                    "event": "order_cancel_replace_by_new_decision",
                    "remaining": old.remaining,
                }
            )
            state.orders.pop(asset)
        if abs(remaining) < 1e-8:
            continue
        exchange = rows.loc[asset, "exchange"] if asset in rows.index else None
        scheduled = next_sessions.get((exchange, date))
        if scheduled is None:
            log.append(
                {
                    "date": date,
                    "asset": asset,
                    "event": "blocked_no_next_calendar_session",
                    "remaining": remaining,
                }
            )
            continue
        state.orders[asset] = OpenOrder(asset, float(remaining), at_time(date, 18), scheduled)
        log.append(
            {
                "date": date,
                "asset": asset,
                "event": "order_submitted",
                "remaining": remaining,
                "scheduled": scheduled,
            }
        )
    return "planned" if complete else "blocked_unknown_close_valuation_orders_retained"


def run_open_events(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    target_builder: Callable,
    cost: TransactionCost,
    initial_cash: float = 1_000_000.0,
    terminal_liquidation: bool = True,
) -> OpenEventResult:
    """Visit every independent calendar session, including sessions with no quotes.

    target_builder receives ONLY the current close section and state (past fills).
    The final scheduled exit is predeclared; unfilled terminal orders remain.
    """
    validate_event_input(panel, calendar)
    if initial_cash <= 0:
        raise ValueError("Positive initial cash required")
    decisions = pd.DatetimeIndex(decision_dates).sort_values()
    sessions = calendar.sort_values(["exchange", "date"])
    next_sessions = {}
    for exchange, part in sessions.groupby("exchange"):
        days = list(part.date)
        next_sessions.update({(exchange, a): b for a, b in zip(days[:-1], days[1:])})
    all_dates = pd.DatetimeIndex(sorted(sessions.date.unique()))
    if decisions.empty or decisions.has_duplicates or not decisions.isin(all_dates).all():
        raise ValueError("Decision dates must be unique independent calendar sessions")
    last_position = all_dates.get_indexer([decisions[-1]])[0] + 2
    if last_position >= len(all_dates):
        raise ValueError("Independent calendar must include final scheduled entry and exit")
    stop = all_dates[last_position]
    state = OpenState(float(initial_cash))
    sections = {d: s.set_index("asset", drop=False) for d, s in panel.groupby("date")}
    event_rows, positions, order_log, targets, decision_log = [], [], [], [], []
    for date in all_dates[(all_dates >= decisions[0]) & (all_dates <= stop)]:
        rows = sections.get(date, panel.iloc[:0].set_index("asset", drop=False))
        state.unsettled = {}  # Previous session's buys settle before this open.
        cash_start = state.cash
        action_cash = _actions(state, rows, date, order_log)
        pre_nav, pre_reason = _mark(state, rows, "raw_open")
        if date == stop and terminal_liquidation:
            for asset, order in state.orders.items():
                order_log.append(
                    {
                        "date": date,
                        "asset": asset,
                        "event": "terminal_cancel_pending_order",
                        "remaining": order.remaining,
                    }
                )
            state.orders = {}
            for asset, quantity in state.quantities.items():
                if quantity > 1e-10:
                    state.orders[asset] = OpenOrder(asset, -quantity, date, date)
        fees, traded = _execute(state, rows, date, cost.at_date(date), pre_nav, order_log)
        post_nav, reason = _mark(state, rows, "raw_open")
        if (
            np.isfinite(pre_nav)
            and np.isfinite(post_nav)
            and not np.isclose(post_nav, pre_nav - fees, atol=1e-7)
        ):
            raise AssertionError("Open NAV does not reconcile to actual cash fees")
        held = sum(q > 1e-10 for q in state.quantities.values())
        event_rows.append(
            {
                "date": date,
                "pre_nav": pre_nav,
                "post_nav": post_nav,
                "cash_before_actions": cash_start,
                "action_cash": action_cash,
                "cash": state.cash,
                "fees": fees,
                "trade_amount": traded,
                "held_assets": held,
                "pending_orders": len(state.orders),
                "nav_status": "valid" if np.isfinite(post_nav) else "BLOCKED",
                "missing_reason": reason or pre_reason,
                "terminal": date == stop,
            }
        )
        for asset, quantity in state.quantities.items():
            if quantity > 1e-10 or asset in state.orders:
                price = rows.loc[asset, "raw_open"] if asset in rows.index else np.nan
                positions.append(
                    {
                        "date": date,
                        "asset": asset,
                        "quantity": quantity,
                        "raw_open": price,
                        "cash": state.cash,
                        "quantity_certified": asset not in state.uncertain_actions,
                        "pending_quantity": state.orders[asset].remaining
                        if asset in state.orders
                        else 0.0,
                    }
                )
        if date in decisions:
            # Labels are joined onto the research panel for ex-post estimation.
            # They are not decision inputs, even when stored on today's row.
            forward = [
                c
                for c in rows
                if c in {"target", "target_valid"} or c.startswith(("entry_", "exit_", "label_"))
            ]
            visible = rows.drop(columns=forward).copy()
            late_close = visible.close_price_asof.isna() | visible.close_price_asof.gt(
                at_time(date, 18)
            )
            for name in ("raw_close", "research_close", "raw_high", "raw_low", "volume", "amount"):
                if name in visible:
                    visible.loc[late_close, name] = np.nan
            weight, diagnostic = target_builder(date, visible, state)
            diagnostic = {"date": date, **diagnostic}
            if weight is not None:
                targets.extend(
                    {"date": date, "asset": a, "weight": float(w)} for a, w in weight.items()
                )
                diagnostic["order_plan_status"] = _plan(
                    state, weight, rows, date, next_sessions, cost.at_date(date), order_log
                )
            else:
                diagnostic["order_plan_status"] = "no_new_target_pending_orders_retained"
            decision_log.append(diagnostic)
    events = pd.DataFrame(event_rows)
    periods = []
    for date in decisions:
        pos = all_dates.get_indexer([date])[0]
        entry, exit_date = all_dates[pos + 1], all_dates[pos + 2]
        start = events.loc[events.date.eq(entry)].iloc[0]
        end = events.loc[events.date.eq(exit_date)].iloc[0]
        finish = end.post_nav if end.terminal else end.pre_nav
        net = finish / start.pre_nav - 1 if np.isfinite([finish, start.pre_nav]).all() else np.nan
        fee_dollars = start.fees + (end.fees if end.terminal else 0.0)
        fees_fraction = fee_dollars / start.pre_nav if np.isfinite(start.pre_nav) else np.nan
        periods.append(
            {
                "date": date,
                "entry_date": entry,
                "exit_date": exit_date,
                "net_return": net,
                "gross_return": net + fees_fraction,
                "cost": fees_fraction,
                "nav": finish / initial_cash,
                "nav_valid": np.isfinite(finish),
                "return_valid": np.isfinite(net),
                "trade_amount": start.trade_amount,
                "turnover_full_l1": start.trade_amount / start.pre_nav
                if np.isfinite(start.pre_nav)
                else np.nan,
                "missing_reason": start.missing_reason + ";" + end.missing_reason,
            }
        )
    holding_frame = pd.DataFrame(positions).reindex(
        columns=[
            "date",
            "asset",
            "quantity",
            "raw_open",
            "cash",
            "quantity_certified",
            "pending_quantity",
        ]
    )
    order_frame = pd.DataFrame(order_log)
    for column in ("date", "asset", "event", "side", "remaining"):
        if column not in order_frame:
            order_frame[column] = pd.Series(
                dtype=str if column in ("asset", "event", "side") else float
            )
    return OpenEventResult(
        events,
        holding_frame,
        order_frame,
        pd.DataFrame(targets).reindex(columns=["date", "asset", "weight"]),
        pd.DataFrame(decision_log),
        pd.DataFrame(periods),
        state,
    )
