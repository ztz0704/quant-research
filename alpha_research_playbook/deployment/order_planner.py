"""Cash-stock main-board order proposals from real account/quote exports.

This module has no brokerage connection or submit method. It reserves existing
available cash, respects broker-reported sellable shares, and emits proposals
requiring risk review. It never interprets a proposal as an executed fill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

CHINA = ZoneInfo("Asia/Shanghai")
MAIN_BOARD = re.compile(r"(?:000|001|002|003|600|601|603|605)\d{3}")


def aware_time(value: Any) -> datetime:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("Explicit timezone required for account, quote and signal timestamps")
    return result.to_pydatetime().astimezone(CHINA)


def number(value: Any, name: str, *, minimum: float = 0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid numeric value for {name}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"Invalid numeric value for {name}")
    return result


def quantity(value: Any) -> int:
    result = number(value, "shares")
    if result != int(result):
        raise ValueError("Fractional shares are unsupported")
    return int(result)


def strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError("Explicit true/false required; missing state cannot become tradable")


def records(frame: pd.DataFrame, required: set[str]) -> dict[str, dict]:
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Required nonempty table columns: {sorted(required)}")
    if frame["asset"].isna().any() or frame["asset"].duplicated().any():
        raise ValueError("Asset keys must be unique and nonmissing")
    rows = frame.to_dict("records")
    if any(not MAIN_BOARD.fullmatch(str(row["asset"])) for row in rows):
        raise ValueError("Only ordinary Shanghai/Shenzhen main-board A shares are supported")
    return {str(row["asset"]): row for row in rows}


def recent(value: Any, now: datetime, seconds: float, name: str) -> datetime:
    timestamp = aware_time(value)
    age = (now - timestamp).total_seconds()
    if age < 0 or age > seconds:
        raise ValueError(f"Future or stale {name}; do not forward-fill it")
    return timestamp


def plan_orders(
    targets: pd.DataFrame,
    quotes: pd.DataFrame,
    account: dict,
    calendar: pd.DataFrame,
    execution_at: str,
    *,
    max_quote_age_seconds: float = 30,
    max_account_age_seconds: float = 60,
    max_signal_age_hours: float = 96,
) -> dict:
    now = aware_time(execution_at)
    # Continuous sessions only; no auction-specific order types are implemented.
    if not (time(9, 30) <= now.time() < time(11, 30) or time(13) <= now.time() < time(14, 57)):
        raise ValueError("Only continuous-session planning is supported")
    if not {"date", "is_open"}.issubset(calendar.columns):
        raise ValueError("A sourced calendar with date,is_open is required")
    dates = pd.to_datetime(calendar["date"], errors="raise").dt.date
    if dates.duplicated().any() or dates.isna().any():
        raise ValueError("Invalid calendar keys")
    session = calendar.loc[dates == now.date(), "is_open"]
    if len(session) != 1 or not strict_bool(session.iloc[0]):
        raise ValueError("Execution date is not an explicitly confirmed open session")
    for value in [max_quote_age_seconds, max_account_age_seconds, max_signal_age_hours]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Freshness limits must be finite and positive")
    if account.get("source") not in {"broker_export", "paper_ledger"}:
        raise ValueError("Declare account source as broker_export or paper_ledger")
    recent(account["asof"], now, max_account_age_seconds, "account snapshot")
    if account.get("open_orders") != []:
        raise ValueError("Reconcile all outstanding orders before planning a new batch")
    equity = number(account["equity"], "equity", minimum=0.01)
    cash = number(account["cash_available"], "available cash")
    if cash > equity:
        raise ValueError("Available cash cannot exceed account equity")
    policy = account["policy"]
    # User/broker declared policy; no assumed commission or tax schedule.
    reserve_bps = number(policy["fee_reserve_bps"], "fee reserve bps")
    min_fee = number(policy["minimum_commission"], "minimum commission")
    buffer = number(policy["cash_buffer"], "cash buffer")
    cap = number(policy["max_position_weight"], "maximum weight", minimum=0.000001)
    if cap > 1:
        raise ValueError("Long-only cash-stock weight cap must be <= 1")
    target_rows = records(targets, {"asset", "target_weight", "signal_asof"})
    quote_rows = records(
        quotes,
        {
            "asset",
            "quote_at",
            "reference_price",
            "bid",
            "ask",
            "upper_limit",
            "lower_limit",
            "suspended",
            "price_adjustment",
        },
    )
    holdings = {}
    for position in account["positions"]:
        asset = str(position["asset"])
        if asset in holdings or not MAIN_BOARD.fullmatch(asset):
            raise ValueError("Duplicate or unsupported account position")
        total, sellable = quantity(position["quantity"]), quantity(position["sellable_quantity"])
        if sellable > total:
            raise ValueError("Sellable shares cannot exceed total holdings")
        holdings[asset] = (total, sellable)
    if set(holdings) - set(target_rows):
        raise ValueError("Include every held asset explicitly; absence is not a sell instruction")
    if set(target_rows) - set(quote_rows):
        raise ValueError("Missing current quotes or trading state for a target asset")
    weights = {}
    checked_quotes = {}
    for asset, target in target_rows.items():
        weight = number(target["target_weight"], "target weight")
        if weight > cap:
            raise ValueError("Target weight exceeds declared position cap")
        weights[asset] = weight
        signal_time = recent(target["signal_asof"], now, max_signal_age_hours * 3600, "signal")
        if signal_time >= now:
            raise ValueError("Signal must exist strictly before execution planning")
        quote = quote_rows[asset]
        quote_time = recent(quote["quote_at"], now, max_quote_age_seconds, "quote")
        if quote_time.date() != now.date() or quote["price_adjustment"] != "none":
            raise ValueError("Execution requires same-session unadjusted quotes")
        price = number(quote["reference_price"], "reference price", minimum=0.01)
        lower = number(quote["lower_limit"], "lower limit", minimum=0.01)
        upper = number(quote["upper_limit"], "upper limit", minimum=0.01)
        bid, ask = number(quote["bid"], "bid"), number(quote["ask"], "ask")
        if not lower <= price <= upper or lower >= upper or (bid > ask > 0):
            raise ValueError("Inconsistent quote/limit prices")
        if any(x > 0 and not lower <= x <= upper for x in [bid, ask]):
            raise ValueError("Executable quote outside supplied price limits")
        if any(abs(x * 100 - round(x * 100)) > 0.000001 for x in [price, lower, upper, bid, ask]):
            raise ValueError("Main-board prices require a 0.01 CNY tick")
        checked_quotes[asset] = (price, bid, ask, lower, upper, strict_bool(quote["suspended"]))
    if sum(weights.values()) > 1 + 1e-10:
        raise ValueError("Target weights would require leverage")
    remaining_cash = max(0.0, cash - buffer)
    output = []
    # Alphabetical order is a documented cash-allocation convention, not alpha.
    # Unfilled sells never finance buys in this batch.
    for asset in sorted(target_rows):
        price, bid, ask, lower, upper, suspended = checked_quotes[asset]
        current, sellable = holdings.get(asset, (0, 0))
        desired = math.floor(weights[asset] * equity / price / 100) * 100
        delta = desired - current
        row = {
            "asset": asset,
            "target_weight": weights[asset],
            "current_quantity": current,
            "sellable_quantity": sellable,
            "rounded_target_quantity": desired,
            "side": "buy" if delta > 0 else "sell" if delta < 0 else "none",
            "requested_quantity": abs(delta),
            "proposed_quantity": 0,
            "limit_price": None,
            "reserved_cash": 0.0,
            "status": "no_change",
            "reasons": [],
        }
        if delta and suspended:
            row.update(status="blocked", reasons=["suspended"])
        elif delta > 0:
            if ask <= 0 or ask >= upper:
                row.update(status="blocked", reasons=["no_ask_or_at_upper_limit"])
            else:
                qty = (delta // 100) * 100
                affordable = (
                    math.floor(
                        max(0.0, remaining_cash - min_fee) / (ask * (1 + reserve_bps / 10000)) / 100
                    )
                    * 100
                )
                qty = min(qty, affordable)
                reservation = (
                    qty * ask + max(min_fee, qty * ask * reserve_bps / 10000) if qty else 0.0
                )
                if qty:
                    remaining_cash -= reservation
                    row.update(
                        proposed_quantity=qty,
                        limit_price=ask,
                        reserved_cash=reservation,
                        status="proposed",
                    )
                    if qty < delta:
                        row["reasons"].append("reduced_for_lots_or_available_cash")
                else:
                    row.update(status="blocked", reasons=["insufficient_cash_or_below_buy_lot"])
        elif delta < 0:
            if bid <= 0 or bid <= lower:
                row.update(status="blocked", reasons=["no_bid_or_at_lower_limit"])
            else:
                qty = min(-delta, sellable)
                # Odd shares allowed only for a complete, fully sellable exit.
                qty = qty if desired == 0 and qty == current else qty // 100 * 100
                if qty:
                    row.update(proposed_quantity=qty, limit_price=bid, status="proposed")
                    if qty < -delta:
                        row["reasons"].append("reduced_for_settlement_or_odd_lot_restriction")
                else:
                    row.update(status="blocked", reasons=["no_settled_sellable_lot"])
        projected = current + (
            row["proposed_quantity"] if row["side"] == "buy" else -row["proposed_quantity"]
        )
        row["projected_quantity_if_filled"] = projected
        row["projected_weight_if_filled"] = projected * price / equity
        if row["projected_weight_if_filled"] > cap + 1e-10:
            row["reasons"].append("residual_position_exceeds_cap_risk_review_required")
        output.append(row)
    return {
        "execution_at": now.isoformat(),
        "account_source": account["source"],
        "orders": output,
        "unreserved_available_cash": remaining_cash,
        "ready_for_live": False,
        "risk_review_required": True,
        "limits": [
            "Proposals only: no submit, fill or account-state mutation",
            "Main-board continuous sessions only; no STAR/ChiNext/BSE/ETF rules",
            "Price limits and sellability come from caller-supplied real data",
            "No assumed fill at a limit; conservative limit-side rejection",
            "Buy cash excludes proceeds from unfilled sell orders",
            "Recheck industry/style/turnover/risk constraints after lot rounding",
            "Need broker callbacks, reservations, idempotency and reconciliation",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["targets", "quotes", "account", "calendar", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execution-at", required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        parser.error("Use a new output directory; do not overwrite a decision record")
    try:
        targets = pd.read_csv(args.targets, dtype={"asset": str})
        quotes = pd.read_csv(args.quotes, dtype={"asset": str})
        calendar = pd.read_csv(args.calendar)
        account = json.loads(args.account.read_text(encoding="utf-8"))
        result = plan_orders(targets, quotes, account, calendar, args.execution_at)
        result["input_sha256"] = {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ["targets", "quotes", "account", "calendar"]
        }
        result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["status"] = "proposals_require_risk_review"
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = {"status": "blocked", "reason": str(exc), "ready_for_live": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if "orders" in result:
        pd.DataFrame(result["orders"]).to_csv(output / "order_proposals.csv", index=False)
    print(
        json.dumps(
            {"status": result["status"], "ready_for_live": False, "output": str(output)},
            ensure_ascii=True,
        )
    )
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
