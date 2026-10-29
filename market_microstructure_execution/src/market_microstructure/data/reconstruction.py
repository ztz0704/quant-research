"""Conservative SZSE message replay, preserving observed orders/trades and anomalies.

This is a snapshot reconstruction for the published 000400.SZ main-board sample,
not a complete multi-exchange matcher. It never simulates a trade to clear a cross.
Market orders are tracked without posting a made-up price, then reduced by the
actual trade messages. Validation against official exchange snapshots remains required.
"""

import heapq
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RestingOrder:
    """Integer-cent price; unpriced market orders have price zero."""

    side: int
    price: int
    quantity: float


class MessageBook:
    """Price-level aggregates with lazy heaps; actual executions alone consume orders."""

    def __init__(self):
        self.orders: dict[int, RestingOrder] = {}
        self.depth: dict[int, dict[int, float]] = {1: {}, 2: {}}
        self.heaps: dict[int, list[int]] = {1: [], 2: []}
        self.audit: Counter = Counter()

    def _adjust(self, side: int, price: int, quantity: float) -> None:
        if price == 0:
            return
        depth = self.depth[side]
        if price not in depth and quantity > 0:
            heapq.heappush(self.heaps[side], -price if side == 1 else price)
        updated = depth.get(price, 0) + quantity
        if updated < -1e-8:
            raise ValueError("Negative aggregate depth: corrupted message accounting")
        if updated <= 1e-8:
            depth.pop(price, None)
        else:
            depth[price] = updated

    def best(self, side: int) -> int:
        """Return zero only when the actual side has no posted price."""
        heap = self.heaps[side]
        while heap:
            price = -heap[0] if side == 1 else heap[0]
            if price in self.depth[side]:
                return price
            heapq.heappop(heap)
        return 0

    def add(self, identifier: int, side: int, price: float, quantity: float, order_type: int) -> None:
        """Track original limit/market/same-side-best orders without invented fills."""
        if identifier in self.orders or side not in {1, 2} or quantity <= 0:
            raise ValueError("Invalid or duplicate order message")
        integer_price = int(round(price * 100)) if order_type == 2 else 0
        if order_type == 3:
            integer_price = self.best(side)
        if order_type not in {1, 2, 3}:
            raise ValueError(f"Unsupported SZSE order type {order_type}")
        if order_type == 2 and integer_price <= 0:
            raise ValueError("Limit order requires positive price")
        self.orders[identifier] = RestingOrder(side, integer_price, quantity)
        self._adjust(side, integer_price, quantity)
        self.audit[f"order_type_{order_type}"] += 1

    def reduce(self, identifier: int, quantity: float, reason: str) -> None:
        """Missing IDs and over-reductions are audited rather than reconstructed from future data."""
        if identifier == 0:
            return
        if identifier not in self.orders:
            self.audit[f"missing_{reason}_order"] += 1
            return
        order = self.orders[identifier]
        if quantity > order.quantity:
            self.audit[f"over_reduction_{reason}"] += 1
        executed = min(quantity, order.quantity)
        self._adjust(order.side, order.price, -executed)
        order.quantity -= executed
        if order.quantity <= 1e-8:
            del self.orders[identifier]

    def snapshot(self, levels: int) -> dict | None:
        """No synthetic deeper levels: skip incomplete/crossed states with a counter."""
        bids = heapq.nlargest(levels, self.depth[1])
        asks = heapq.nsmallest(levels, self.depth[2])
        if len(bids) < levels or len(asks) < levels:
            self.audit["incomplete_snapshot"] += 1
            return None
        if bids[0] >= asks[0]:
            self.audit["crossed_snapshot"] += 1
            return None
        row = {}
        for side, prices, key in ((1, bids, "bid"), (2, asks, "ask")):
            for level, price in enumerate(prices, start=1):
                row[f"{key}_price_{level}"] = price / 100
                row[f"{key}_size_{level}"] = self.depth[side][price]
        return row


def load_szse_messages(orders_path: str, trades_path: str, symbol: str = "000400.SZ") -> tuple[pd.DataFrame, dict]:
    """Read explicit provider fields; one asset/date/channel scope prevents ID ambiguity."""
    shared = ["MDDate", "MDTime", "SecurityID", "ChannelNo", "ApplSeqNum", "ReceiveDateTime"]
    orders = pd.read_csv(orders_path, usecols=shared + ["OrderType", "OrderPrice", "OrderQty", "OrderBSFlag"],
                         dtype={"SecurityID": str, "ReceiveDateTime": str})
    trades = pd.read_csv(trades_path, usecols=shared + ["TradeBuyNo", "TradeSellNo", "TradeType", "TradeBSFlag", "TradePrice", "TradeQty"],
                         dtype={"SecurityID": str, "ReceiveDateTime": str})
    orders["kind"], trades["kind"] = "order", "trade"
    messages = pd.concat([orders, trades], ignore_index=True)
    scope_counts = messages.SecurityID.value_counts().to_dict()
    messages = messages.loc[messages.SecurityID == symbol].copy()
    if messages.empty:
        raise ValueError(f"No messages for the predeclared asset {symbol}")
    for column in ("SecurityID", "MDDate", "ChannelNo"):
        if messages[column].nunique() != 1:
            raise ValueError(f"Public sample replay requires a single {column}; split files first")
    if messages.ApplSeqNum.duplicated().any():
        raise ValueError("Duplicate shared-channel sequence numbers")
    messages = messages.sort_values("ApplSeqNum").reset_index(drop=True)
    received = pd.to_datetime(messages.ReceiveDateTime, format="%Y%m%d%H%M%S%f", errors="raise")
    received = received.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    # Monotonic replay availability cannot travel backward when vendor reception is noisy.
    clock = received.array.as_unit("ns").asi8
    monotonic = np.maximum.accumulate(clock)
    messages["timestamp"] = pd.to_datetime(monotonic, utc=True)
    audit = {"raw_order_rows": len(orders), "raw_trade_rows": len(trades),
             "scope_rows_by_asset": scope_counts, "selected_message_rows": len(messages),
             "outside_predeclared_asset_rows": len(orders) + len(trades) - len(messages),
             "asset": messages.SecurityID.iloc[0], "date": messages.MDDate.iloc[0],
             "channel": int(messages.ChannelNo.iloc[0]),
             "reception_clock_adjusted_rows": int((clock != monotonic).sum()),
             "reception_clock_max_adjustment_ms": float((monotonic - clock).max() / 1e6),
             "sequence_gaps": int((messages.ApplSeqNum.diff() > 1).sum()),
             "gap_note": "Gaps may contain other securities on the channel, not necessarily missing target messages"}
    return messages, audit


def reconstruct_snapshots(messages: pd.DataFrame, frequency_ms: int = 1000, levels: int = 5) -> tuple[pd.DataFrame, dict]:
    """Emit trailing snapshots at grid endpoints using only messages available by that endpoint.

    Continuous sessions only (09:30-11:30, 13:00-14:57); excluded endpoints are counted.
    Quote state carries through silent intervals, but realized trade flow resets each interval.
    """
    if messages.empty or frequency_ms <= 0:
        raise ValueError("Require messages and a positive snapshot interval")
    state = MessageBook()
    rows = []
    step = pd.Timedelta(milliseconds=frequency_ms)
    endpoint = messages.timestamp.iloc[0].ceil(step)
    volume = signed_volume = traded_value = 0.0
    trade_count = 0
    names = list(messages.columns)
    asset = messages.SecurityID.iloc[0]

    def emit(timestamp: pd.Timestamp) -> None:
        nonlocal volume, signed_volume, traded_value, trade_count
        local = timestamp.tz_convert("Asia/Shanghai")
        minute = local.hour * 60 + local.minute
        continuous = (570 <= minute < 690) or (780 <= minute < 897)
        if continuous:
            quote = state.snapshot(levels)
            if quote is not None:
                rows.append({"timestamp": timestamp, "asset": asset, "source": "public_szse_reconstructed",
                    "timestamp_quality": "vendor_receipt_sequence_replay", "session_id": f"{local.date()}_{'AM' if minute < 690 else 'PM'}",
                    "trade_size": volume, "signed_trade_size": signed_volume,
                    "trade_count": trade_count, "trade_price": traded_value / volume if volume else np.nan,
                    "reconstruction_anomaly_count": sum(v for k, v in state.audit.items()
                        if k.startswith(("missing_", "over_reduction"))), **quote})
        else:
            state.audit["outside_continuous_session_endpoints"] += 1
        volume = signed_volume = traded_value = 0.0
        trade_count = 0

    for values in messages.itertuples(index=False, name=None):
        event = dict(zip(names, values, strict=True))
        timestamp = event["timestamp"]
        # Strict > includes messages exactly at a grid endpoint before emitting it.
        while timestamp > endpoint:
            emit(endpoint)
            endpoint += step
        if event["kind"] == "order":
            state.add(int(event["ApplSeqNum"]), int(event["OrderBSFlag"]), event["OrderPrice"],
                      event["OrderQty"], int(event["OrderType"]))
        else:
            quantity = event["TradeQty"]
            execution_type = int(event["TradeType"])
            # This vendor export uses 0=trade, 1=cancel (verified against raw prices/IDs).
            if execution_type not in {0, 1}:
                raise ValueError("Unknown execution/cancellation type")
            if execution_type == 0 and (event["TradePrice"] <= 0 or not event["TradeBuyNo"] or not event["TradeSellNo"]):
                raise ValueError("Trade message requires positive price and both order IDs")
            reason = "cancel" if execution_type == 1 else "trade"
            state.reduce(int(event["TradeBuyNo"]), quantity, reason)
            state.reduce(int(event["TradeSellNo"]), quantity, reason)
            if execution_type == 0:
                volume += quantity
                signed_volume += quantity * (1 if event["TradeBSFlag"] == 1 else -1 if event["TradeBSFlag"] == 2 else 0)
                traded_value += quantity * event["TradePrice"]
                trade_count += 1
    if endpoint <= messages.timestamp.iloc[-1]:
        emit(endpoint)
    audit = dict(state.audit)
    audit.update(snapshot_rows=len(rows), open_order_count=len(state.orders),
                 external_reference_validation="not_available", frequency_ms=frequency_ms)
    return pd.DataFrame(rows), audit
