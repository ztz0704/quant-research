"""V3 orchestration inside the existing ML environment; Risk has its own runtime."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from financial_ml.shared_v3 import (
    INTERFACE_VERSION,
    PROJECTS,
    bind_alpha,
    file_sha,
    implementation_hashes,
)

ALPHA_BINDING = bind_alpha()

from robust_alpha.data.pit_audit_v3 import audit_pit_v3, load_primary_evidence  # noqa: E402
from robust_alpha.data.research_adapters import cached_sina_sample, read_bundle  # noqa: E402
from robust_alpha.data.research_contract import (  # noqa: E402
    add_calendar_labels,
    add_decision_eligibility,
    normalize_dynamic_panel,
    require_frozen,
)
from robust_alpha.data.research_fixture import engineering_sample  # noqa: E402
from robust_alpha.evaluation.conditional_reversal import (  # noqa: E402
    build_pressure_features,
    conditional_research,
)

from financial_ml.diagnostics_v3 import (  # noqa: E402
    model_diagnostics,
    portfolio_inference,
    window_cadence_diagnostics,
)
from financial_ml.research_v2 import (  # noqa: E402
    compare_incremental,
    matched_predictions,
    train_incremental,
)
from financial_ml.utils import path, write_json  # noqa: E402


def clock_at(date, hour: int, minute: int = 0) -> pd.Timestamp:
    from datetime import datetime, time

    return pd.Timestamp(datetime.combine(pd.Timestamp(date).date(), time(hour, minute)))


def load_inputs(cfg: dict) -> tuple:
    source = cfg["data"]["source"]
    if source == "synthetic":
        return engineering_sample(**cfg["synthetic"])
    if source == "cached_sina":
        return cached_sina_sample(PROJECTS / "risk_aware_portfolio")
    if source == "bundle":
        if not cfg["data"].get("load_end"):
            raise ValueError("Bundle needs an explicit pre-read boundary")
        return read_bundle(path(cfg["data"]["bundle_directory"]), cfg["data"]["load_end"])
    raise ValueError("Unsupported V3 source")


def build_panel(raw: pd.DataFrame, calendar: pd.DataFrame, cfg: dict) -> tuple:
    panel = normalize_dynamic_panel(raw, calendar)
    panel = add_decision_eligibility(panel, cfg["data"]["min_history"], allow_unknown_status=True)
    panel, features = build_pressure_features(
        panel, cfg["data"]["min_assets"], cfg["data"]["min_amount_20"]
    )
    return add_calendar_labels(panel, calendar), features


def event_scenario(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    evidence = cfg["execution_evidence"]
    expected = (
        "controlled_synthetic_full_fill_oracle"
        if cfg["data"]["source"] == "synthetic"
        else "assumed_native_bar_full_fill"
    )
    if evidence != expected:
        raise ValueError("Execution evidence/source mismatch; no implicit fill model")
    result = panel.copy()
    # Explicit engineering assumptions, never historical publication testimony.
    opens = result.date.map(lambda date: clock_at(date, 9, 30))
    result["execution_asof"] = opens
    # Separate event scenario timestamp. Original adjustment_asof stays intact
    # and unverified; this is not a supplied historical publication record.
    result["action_indicator_asof"] = opens
    result["open_price_asof"] = opens.where(result.raw_open.notna())
    result["close_price_asof"] = result.date.map(lambda date: clock_at(date, 18)).where(
        result.raw_close.notna()
    )
    result["buy_fill_fraction"] = result.buyable.eq(True).fillna(False).astype(float)
    result["sell_fill_fraction"] = result.sellable.eq(True).fillna(False).astype(float)
    result["action_complete"] = False
    # Null life-cycle status remains unknown on the real sample.
    result["delisted"] = result.delist_date.notna() & result.date.ge(result.delist_date)
    if cfg["data"]["source"] != "synthetic":
        result["delisted"] = result.delisted.astype("boolean").mask(
            result.delist_date.isna(), pd.NA
        )
    result["execution_evidence"] = evidence
    return result


def risk_settings(cfg: dict) -> dict:
    risk_config = yaml.safe_load(path(cfg["risk_config"]).read_text(encoding="utf-8"))
    risk_config["risk_model"].update(cfg["open_risk"])
    risk_config["portfolio"].update(cfg.get("risk_portfolio", {}))
    risk_config["backtest"]["rebalance_frequency"] = "daily"
    # Same reduced-form cost definition for A-E; historical stamp/impact are not
    # silently inherited from an incompatible old experiment.
    risk_config["transaction_cost"] = {"linear_bps": 0.0, "impact_coefficient": 0.0}
    return risk_config


def create_risk_bundle(
    destination: Path, panel: pd.DataFrame, calendar: pd.DataFrame, archive: pd.DataFrame, cfg: dict
) -> Path:
    matched, _ = matched_predictions(archive)
    bridge = destination / "risk_inputs"
    bridge.mkdir(exist_ok=True)
    event_scenario(panel, cfg).to_parquet(bridge / "panel.parquet", index=False)
    calendar.to_parquet(bridge / "calendar.parquet", index=False)
    # The execution consumer cannot inspect future targets or trading states.
    fields = [
        "date",
        "asset",
        "model",
        "score",
        "expected_return",
        "rev",
        "feature_asof",
        "decision_time",
        "model_fit_cutoff",
        "estimator_fit_cutoff",
        "calibration_fit_cutoff",
        "max_fit_label_available",
        "max_calibration_label_available",
        "return_convention",
        "return_unit",
        "model_version",
    ]
    matched[fields].to_parquet(bridge / "signals.parquet", index=False)
    contract = {
        "interface_version": INTERFACE_VERSION,
        "evidence_grade": cfg["phase"],
        "execution_evidence": cfg["execution_evidence"],
        "action_indicator_evidence": "assumed_auction_indicator_not_verified_pit",
        "formal_empirical": False,
        "decision_dates": archive.attrs["oos_calendar"],
        "risk_config": risk_settings(cfg),
        "settings": {"portfolio": cfg["portfolio"], "event_portfolio": cfg["event_portfolio"]},
        "output_directory": str((destination / "portfolio").resolve()),
        "implementation_hashes": implementation_hashes(),
        "inputs": {
            name: file_sha(bridge / name)
            for name in ("panel.parquet", "calendar.parquet", "signals.parquet")
        },
    }
    write_json(contract, bridge / "contract.json")
    return bridge


def execute_risk(bridge: Path, destination: Path) -> None:
    root = PROJECTS / "risk_aware_portfolio"
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        environment[name] = "1"
    process = subprocess.run(
        [
            str(root / ".venv/Scripts/python.exe"),
            str(root / "scripts/run_research_v3.py"),
            "--bundle",
            str(bridge.resolve()),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    (destination / "risk_stdout.log").write_text(process.stdout, encoding="utf-8")
    (destination / "risk_stderr.log").write_text(process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"V3 Risk subprocess failed ({process.returncode}); see risk_stderr.log")


def run_v3(cfg: dict, *, diagnostics: bool = True) -> Path:
    if cfg.get("interface_version") != INTERFACE_VERSION:
        raise ValueError("Unsupported ML/Risk interface")
    protocol = PROJECTS / "Robust_Alpha_Discovery/configs/protocols/research_v2.txt"
    require_frozen(cfg, file_sha(protocol), include_holdout=False)
    if cfg["phase"] not in ("historical_integration", "engineering_validation"):
        raise ValueError("V3 only runs viewed historical / synthetic engineering data")
    if (cfg["data"]["source"] == "synthetic") != (cfg["phase"] == "engineering_validation"):
        raise ValueError("Real/synthetic output separation violated")
    destination = path(cfg["output"]["directory"])
    if "research_v3" not in destination.parts:
        raise ValueError("V3 outputs require a distinct research_v3 directory")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "experiment_manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8"))["config"] != cfg
    ):
        raise ValueError("Existing V3 config differs; choose a separate output directory")
    manifest = {
        "status": "running",
        "interface_version": INTERFACE_VERSION,
        "config": deepcopy(cfg),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "formal_empirical": False,
        "holdout_run": False,
        "evidence_grade": cfg["phase"],
        "alpha_binding": ALPHA_BINDING,
        "implementation_hashes": implementation_hashes(),
        "protocol_sha256": file_sha(protocol),
        "python": platform.python_version(),
        "packages": {
            p: version(p)
            for p in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "joblib")
        },
    }
    write_json(manifest, manifest_path)
    try:
        raw, calendar, metadata = load_inputs(cfg)
        manifest["original_source_files"] = metadata.get("source_files", {})
        primary = {}
        if cfg["data"]["source"] == "bundle":
            root = path(cfg["data"]["bundle_directory"])
            primary = load_primary_evidence(root, metadata, cfg["data"]["load_end"])
            names = ["metadata.json", "panel.parquet", "calendar.parquet"] + [
                info["filename"] for info in metadata.get("audit_files", {}).values()
            ]
            manifest["original_source_files"].update(
                {str((root / name).resolve()): file_sha(root / name) for name in names}
            )
        raw.to_parquet(destination / "source_snapshot.parquet", index=False)
        calendar.to_parquet(destination / "calendar_snapshot.parquet", index=False)
        write_json(metadata, destination / "data_lineage.json")
        normalized = normalize_dynamic_panel(raw, calendar)
        audit = audit_pit_v3(
            normalized, metadata, primary.get("publication_log"), primary.get("lifecycle_registry")
        )
        write_json(audit.pop("report"), destination / "data_quality_report.json")
        for name, frame in audit.items():
            frame.to_csv(destination / f"data_{name}.csv", index=False)
        panel, features = build_panel(raw, calendar, cfg)
        panel.to_parquet(destination / "research_panel.parquet", index=False)
        archive, fits, tuning, models = train_incremental(panel, features, cfg)
        archive.to_parquet(destination / "oos_predictions.parquet", index=False)
        tables = conditional_research(
            panel, cfg["data"]["min_assets"], cfg["statistics"]["hac_lags"]
        )
        factor_summary = tables["factor_ic_summary"]
        factor_summary.loc[factor_summary.missing_dates.gt(0), "icir_annualized"] = np.nan
        factor_summary["annual_icir_status"] = np.where(
            factor_summary.missing_dates.gt(0), "N/A_incomplete_calendar", "descriptive_only"
        )
        tables.update(compare_incremental(archive, cfg))
        tables.update(model_fits=fits, validation_tuning=tuning)
        model_dir = destination / "models"
        model_dir.mkdir(exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, model_dir / f"{name}.joblib")
        bridge = create_risk_bundle(destination, panel, calendar, archive, cfg)
        execute_risk(bridge, destination)
        risk_dir = destination / "portfolio"
        tables.update(
            portfolio_inference(
                pd.read_parquet(risk_dir / "portfolio_periods.parquet"),
                pd.read_parquet(risk_dir / "risk_calibration_daily.parquet"),
                cfg,
            )
        )
        if diagnostics:
            tables.update(model_diagnostics(archive, panel, fits, models, features, cfg))
            tables.update(window_cadence_diagnostics(panel, features, archive, fits, cfg))
        for name, frame in tables.items():
            frame.to_csv(destination / f"{name}.csv", index=False)
        manifest.update(
            status="complete",
            diagnostic_status="complete" if diagnostics else "not_requested",
            oos_dates=len(archive.attrs["oos_calendar"]),
            prediction_rows=len(archive),
            folds=archive.oos_fold.nunique(),
            features=features,
            pit_formal_ready=json.loads(
                (destination / "data_quality_report.json").read_text(encoding="utf-8")
            )["formal_ready"],
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        manifest["artifacts"] = {
            str(p.relative_to(destination)): file_sha(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file() and p != manifest_path and p.name != "verification.json"
        }
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(manifest, manifest_path)
    return destination
