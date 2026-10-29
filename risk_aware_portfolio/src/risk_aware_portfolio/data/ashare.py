"""Explicit causal price features and strict gates for a fixed A-share basket."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from risk_aware_portfolio.data.sina import (
    SinaClient,
    decode_events,
    decode_prices,
    decode_share_history,
    merge_events,
)
from risk_aware_portfolio.data.suspensions import insert_suspension_marks
from risk_aware_portfolio.data.validation import validate_panel
from risk_aware_portfolio.utils.config import Config
from risk_aware_portfolio.utils.manifest import file_hash
from risk_aware_portfolio.utils.paths import resolve_path


def read_universe(config: Config) -> pd.DataFrame:
    """Read a predeclared basket; never select survivors using downloaded returns."""
    frame = pd.read_csv(resolve_path(config["real_data"]["universe_path"]), dtype={"asset": str})
    if (
        frame["asset"].duplicated().any()
        or not frame["asset"].map(lambda x: bool(re.fullmatch(r"(sh|sz)\d{6}", x))).all()
    ):
        raise ValueError("Duplicate or invalid A-share symbols")
    frame["classification_asof"] = pd.to_datetime(frame["classification_asof"])
    if frame["classification_asof"].gt(pd.Timestamp(config["real_data"]["fetch_start"])).any():
        raise ValueError("Declared classification dates follow the acquisition start")
    if set(frame["industry"]) != set(config["risk_model"]["industries"]):
        raise ValueError("Declared business groups and model vocabulary differ")
    return frame.sort_values("asset").reset_index(drop=True)


def coverage_audit(
    quotes: pd.DataFrame, calendar: pd.DatetimeIndex, assets: pd.Series
) -> pd.DataFrame:
    """Report every missing quote and zero-volume date without filtering stocks."""
    rows = []
    for asset in assets:
        section = quotes[quotes["asset"].eq(asset)]
        dates = pd.DatetimeIndex(section["date"])
        if dates.duplicated().any():
            raise ValueError(f"Duplicate native quote dates: {asset}")
        rows.append(
            {
                "asset": asset,
                "native_rows": len(section),
                "missing_sessions": len(calendar.difference(dates)),
                "unexpected_sessions": len(dates.difference(calendar)),
                "zero_volume_sessions": int(section["volume"].le(0).sum()),
                "invalid_close_sessions": int(section["close"].le(0).sum()),
                "missing_dates": ";".join(calendar.difference(dates).strftime("%Y-%m-%d")),
            }
        )
    return pd.DataFrame(rows)


def enforce_coverage(audit: pd.DataFrame) -> None:
    """Stop rather than silently handle suspensions, delistings or data gaps."""
    columns = [
        "missing_sessions",
        "unexpected_sessions",
        "zero_volume_sessions",
        "invalid_close_sessions",
    ]
    failures = audit.loc[audit[columns].sum(axis=1).gt(0)]
    if not failures.empty:
        raise ValueError(
            "Native quote coverage/tradability gate failed; no stocks were removed. "
            f"Inspect coverage.csv: {failures['asset'].tolist()}"
        )


def asset_features(section: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Build trailing features, then explicitly delay signal/exposures by one session."""
    settings = config["real_data"]
    section = section.sort_values("date").copy()
    adjusted = section.get("adjusted_close", section["close"] * section["hfq_factor"])
    returns = adjusted.pct_change(fill_method=None)
    log_price = np.log(adjusted)
    cap = section["close"] * section["outstanding_share"]
    turn = section["volume"] / section["outstanding_share"]
    raw = pd.DataFrame(index=section.index)
    raw["market_cap"] = cap
    raw["size"] = np.log(cap)
    raw["momentum"] = log_price.shift(settings["momentum_skip"]) - log_price.shift(
        settings["momentum_lookback"]
    )
    raw["volatility"] = returns.rolling(settings["volatility_window"]).std(ddof=1)
    raw["liquidity"] = np.log(turn.rolling(settings["liquidity_window"]).mean())
    # Fixed five-session reversal is a plumbing signal, not an independently validated alpha.
    raw["alpha"] = -(log_price - log_price.shift(settings["reversal_window"]))
    lagged = raw.shift(settings["execution_delay_sessions"])
    output = section[["date", "asset", "industry"]].copy()
    if "tradable" in section:
        output["tradable"] = section["tradable"].astype(bool)
        output["native_quote"] = section["native_quote"].astype(bool)
    output["return"] = returns
    output = pd.concat([output, lagged], axis=1)
    output["feature_asof"] = section["date"].shift(settings["execution_delay_sessions"])
    for kind in ("shares", "adjustments"):
        output[f"{kind}_effective_date"] = section[f"{kind}_effective_date"].shift(
            settings["execution_delay_sessions"]
        )
    return output


def build_real_panel(quotes: pd.DataFrame, universe: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Only drop the deterministic rolling warmup; missing observations afterward fail."""
    settings = config["real_data"]
    if settings["execution_delay_sessions"] < 1:
        raise ValueError("Real-data signals require at least one complete-session delay")
    joined = quotes.merge(universe[["asset", "industry"]], on="asset", validate="many_to_one")
    warmup = (
        max(
            settings["momentum_lookback"],
            settings["volatility_window"],
            settings["liquidity_window"],
            settings["reversal_window"],
        )
        + settings["execution_delay_sessions"]
    )
    sections = []
    for _, section in joined.groupby("asset", sort=True):
        features = asset_features(section, config)
        sections.append(features.iloc[warmup:])
    panel = pd.concat(sections, ignore_index=True).sort_values(["date", "asset"])
    panel = panel.reset_index(drop=True)
    validate_panel(panel, config["data"]["max_abs_return"])
    if (panel["feature_asof"] >= panel["date"]).any():
        raise ValueError("Feature as-of date must precede the trading date")
    return panel


def acquire_quotes(config: Config, refresh: bool = False, bypass_proxy: bool = False) -> Path:
    """Download raw records, gate native coverage, and persist immutable audit inputs."""
    settings = config["real_data"]
    cache = resolve_path(settings["raw_dir"])
    client = SinaClient(
        cache,
        settings["request_timeout"],
        settings["retries"],
        settings["request_pause"],
        bypass_proxy,
    )
    universe = read_universe(config)
    suspensions = pd.read_csv(resolve_path(settings["suspension_path"]))
    for column in ("date", "announcement_date"):
        suspensions[column] = pd.to_datetime(suspensions[column])
    if suspensions.duplicated(["date", "asset"]).any():
        raise ValueError("Duplicate suspension declarations")
    if settings["suspension_policy"] != "mark_last_and_defer_all_trades":
        raise ValueError("Unknown suspension policy")
    quotes = []
    native = []
    try:
        reference = _window(
            decode_prices(client.payload(settings["calendar_symbol"], "prices", refresh)), settings
        )
        reference.to_parquet(cache / "calendar.parquet", index=False)
        for number, asset in enumerate(universe["asset"], 1):
            print(f"A-share acquisition: {number}/{len(universe)} {asset}", flush=True)
            prices = _window(decode_prices(client.payload(asset, "prices", refresh)), settings)
            native.append(prices.assign(asset=asset))
            prices = insert_suspension_marks(
                prices, pd.DatetimeIndex(reference["date"]), asset, suspensions
            )
            for kind in ("shares", "adjustments"):
                payload = client.payload(asset, kind, refresh)
                if kind == "shares" and "(null)" in payload:
                    if settings.get("share_fallback") != "sina_dated_float_a_history":
                        raise ValueError(f"Missing share count: {asset}")
                    events = decode_share_history(
                        client.payload(asset, "shares_history", refresh),
                        settings["fetch_start"],
                        settings["end_date"],
                    )
                    prices["share_source"] = "historical_float_a_chart"
                else:
                    events = decode_events(
                        payload,
                        kind,
                        settings["fetch_start"],
                        settings["end_date"],
                    )
                    if kind == "shares":
                        prices["share_source"] = "dated_share_count_endpoint"
                prices = merge_events(prices, events, kind)
            prices["adjusted_close"] = prices["close"] * prices["hfq_factor"]
            halted = ~prices["tradable"]
            prices.loc[halted, "adjusted_close"] = (
                prices["adjusted_close"].where(~halted).ffill().loc[halted]
            )
            quotes.append(prices.assign(asset=asset))
    finally:
        client.close()
    native_quotes = pd.concat(native, ignore_index=True).sort_values(["date", "asset"])
    native_quotes.to_parquet(cache / "quotes.parquet", index=False)
    audit = coverage_audit(native_quotes, pd.DatetimeIndex(reference["date"]), universe["asset"])
    audit.to_csv(cache / "coverage.csv", index=False)
    combined = pd.concat(quotes, ignore_index=True).sort_values(["date", "asset"])
    combined.to_parquet(cache / "valuation_quotes.parquet", index=False)
    universe.to_csv(cache / "universe_snapshot.csv", index=False)
    return cache / "valuation_quotes.parquet"


def _window(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    mask = frame["date"].between(settings["fetch_start"], settings["end_date"])
    if not mask.any():
        raise ValueError("No native quotes in the declared period")
    return frame.loc[mask].reset_index(drop=True)


def save_data_manifest(config: Config, panel_path: Path) -> None:
    """Tie raw HTTP payloads, universe declarations and the resulting panel together."""
    cache = resolve_path(config["real_data"]["raw_dir"])
    manifest = {
        "config": config,
        "panel_sha256": file_hash(panel_path),
        "universe_sha256": file_hash(resolve_path(config["real_data"]["universe_path"])),
        "raw_sha256": {p.name: file_hash(p) for p in sorted(cache.glob("*.txt"))},
        "quotes_sha256": file_hash(cache / "quotes.parquet"),
        "valuation_quotes_sha256": file_hash(cache / "valuation_quotes.parquet"),
        "suspensions_sha256": file_hash(resolve_path(config["real_data"]["suspension_path"])),
        "source": "Sina raw quotes decoded using AKShare; effective-date event joins",
        "pit_certified": False,
        "limitations": [
            "Retrospectively declared fixed basket, not survivorship-free historical membership",
            "Coarse researcher business groups, not authenticated historical industry classifications",
            "Effective dates available; historical publication/revision timestamps unavailable",
            "Native adjusted-price proxy, not separately reconciled corporate-action cash accounting",
            "Value and quality omitted: no timestamp-certified financial statement history",
        ],
    }
    (cache / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
