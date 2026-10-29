"""Strict panel loading and explicitly separate synthetic validation data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from financial_ml.utils import output, path, sha256, write_json

REQUIRED = {"date", "asset", "return", "market_cap", "industry"}


def validate(frame: pd.DataFrame) -> None:
    """Reject ambiguous ordering, identity, economics and nonfinite inputs."""
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty or frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Empty panel or missing required values")
    if frame.duplicated(["date", "asset"]).any():
        raise ValueError("Duplicate date-asset keys")
    if not frame[["date", "asset"]].equals(frame.sort_values(["date", "asset"])[["date", "asset"]]):
        raise ValueError("Panel must be sorted by date and asset")
    numeric = frame.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite input")
    if not frame["return"].between(-0.95, 0.95).all() or not frame["market_cap"].gt(0).all():
        raise ValueError("Invalid return or market capitalization")
    if "feature_asof" in frame and (pd.to_datetime(frame["feature_asof"]) >= frame["date"]).any():
        raise ValueError("Cached features must precede their decision date")
    if "tradable" in frame and not frame["tradable"].isin([True, False]).all():
        raise ValueError("Invalid tradability")
    dates = frame["date"].nunique()
    if dates < 10:
        raise ValueError("Insufficient history")
    if not frame.groupby("asset", observed=True).size().eq(dates).all():
        raise ValueError("This version requires a complete market-date grid per asset")


def load(filename: Path) -> pd.DataFrame:
    """Keep asset identifiers as strings and validate; never silently fill returns."""
    frame = pd.read_parquet(filename) if filename.suffix == ".parquet" else pd.read_csv(filename, dtype={"asset": str})
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    validate(frame)
    return frame


def synthetic(days: int = 500, assets: int = 40, seed: int = 8, stationary: bool = False, signal: bool = True) -> pd.DataFrame:
    """Generate known stationary/break/drift regimes for tests, never empirical claims.

    Returns at t use features observed at t-1. The stochastic draws are independent
    of future targets and signal magnitude is modest relative to residual noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=days)
    rows = []
    previous = np.zeros(assets)
    for t, date in enumerate(dates):
        shifted = not stationary and t >= days // 2
        x = rng.normal(float(shifted) * 0.4, 1.0, assets)
        rho = 0.9 if not shifted else 0.5
        copy = rho * x + np.sqrt(1 - rho * rho) * rng.normal(size=assets)
        beta = (0.0015 if not shifted else -0.0015) if signal else 0.0
        ret = beta * previous + rng.normal(0, 0.012 * (1.5 if shifted else 1), assets)
        for i in range(assets):
            rows.append({"date": date, "asset": f"SYN{i:03}", "return": ret[i], "market_cap": np.exp(21 + 0.2 * x[i]),
                         "industry": f"G{i % 4}", "x": x[i], "x_correlated": copy[i], "tradable": True,
                         "synthetic_regime": int(shifted)})
        previous = x
    result = pd.DataFrame(rows).sort_values(["date", "asset"]).reset_index(drop=True)
    validate(result)
    return result


def prepare(cfg: dict[str, Any]) -> pd.DataFrame:
    """Snapshot a real source into this project and record provenance."""
    destination = path(cfg["data"]["input_path"])
    source = cfg["data"]["source"]
    if source == "synthetic":
        if "engineering" not in destination.parts:
            raise ValueError("Synthetic input must live in data/engineering")
        frame = synthetic(**cfg.get("synthetic", {}))
        provenance = {"source": "synthetic", "empirical": False}
    elif source == "file":
        original = path(cfg["data"]["source_path"])
        if not original.is_file():
            raise FileNotFoundError(f"Real source missing; no synthetic fallback: {original}")
        frame = load(original)
        provenance = {"source": str(original.resolve()), "source_sha256": sha256(original), "empirical": True,
                      "provider": cfg["data"]["provider"], "universe_policy": cfg["data"]["universe_policy"],
                      "vendor_publication_history_verified": cfg["data"]["vendor_publication_history_verified"]}
        upstream = original.parent.parent / "raw/ashare/data_manifest.json"
        if upstream.is_file():
            provenance["upstream_manifest_sha256"] = sha256(upstream)
    else:
        raise ValueError(f"Unsupported source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    provenance.update(rows=len(frame), assets=int(frame.asset.nunique()), start=str(frame.date.min()), end=str(frame.date.max()),
                      input_sha256=sha256(destination), columns=list(frame.columns))
    write_json(provenance, output(cfg) / "tables/data_manifest.json")
    return frame
