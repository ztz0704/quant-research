"""Recompute the changed Risk stage; reuse only hash-identical Alpha/ML stages.

This is not a stale-hash override: every old artifact/input is checked, only Risk
source changes are allowed, and every portfolio output and inference is rerun.
The original experiment remains untouched in its archival directory.
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import financial_ml

ROOT = Path(__file__).resolve().parents[2]
if (
    Path(financial_ml.__file__).resolve()
    != ROOT / "nonstationary_financial_ml/src/financial_ml/__init__.py"
):
    raise ImportError("Use the existing local ML environment")

from financial_ml.shared_v3 import bind_alpha  # noqa: E402

bind_alpha()

from financial_ml.diagnostics_v3 import portfolio_inference  # noqa: E402
from financial_ml.research_v3_pipeline import create_risk_bundle, execute_risk  # noqa: E402
from financial_ml.shared_v3 import file_sha, implementation_hashes  # noqa: E402
from financial_ml.utils import config, path, write_json  # noqa: E402


def replay(config_file: str, previous: Path):
    cfg = config(config_file)
    destination = path(cfg["output"]["directory"])
    old_path = previous / "experiment_manifest.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    current = implementation_hashes()
    if old["status"] != "complete" or old["config"] != cfg or destination.exists():
        raise ValueError(
            "Replay requires a complete same-config archive and fresh output directory"
        )
    if old["protocol_sha256"] != file_sha(ROOT / "Robust_Alpha_Discovery/configs/protocols/research_v2.txt"):
        raise ValueError("Changed protocol cannot use a cached stage")
    changes = {
        name
        for name in set(current) | set(old["implementation_hashes"])
        if current.get(name) != old["implementation_hashes"].get(name)
    }
    if not changes or any(
        not name.replace("\\", "/").startswith("risk_aware_portfolio/") for name in changes
    ):
        raise ValueError("Alpha/ML/package changes require a full retrain, not Risk-only replay")
    for name, sha in old["artifacts"].items():
        if file_sha(previous / name) != sha:
            raise ValueError(f"Modified old artifact: {name}")
    for name, sha in old["original_source_files"].items():
        if file_sha(Path(name)) != sha:
            raise ValueError(f"Changed original data source: {name}")
    destination.mkdir(parents=True)
    for source in previous.iterdir():
        if source.name in {
            "portfolio",
            "risk_inputs",
            "experiment_manifest.json",
            "verification.json",
            "risk_stdout.log",
            "risk_stderr.log",
        }:
            continue
        if source.is_dir():
            shutil.copytree(source, destination / source.name)
        else:
            shutil.copy2(source, destination / source.name)
    manifest = {
        **old,
        "status": "running",
        "implementation_hashes": current,
        "replay_started_utc": datetime.now(timezone.utc).isoformat(),
        "stage_replay": {
            "previous_manifest_sha256": file_sha(old_path),
            "previous_directory": str(previous),
            "generator_sha256": file_sha(Path(__file__)),
            "changed_sources": sorted(changes),
            "alpha_ml_source_and_artifact_hashes_unchanged": True,
            "risk_and_portfolio_inference_recomputed": False,
        },
        "artifacts": {},
    }
    manifest_file = destination / "experiment_manifest.json"
    write_json(manifest, manifest_file)
    try:
        panel = pd.read_parquet(destination / "research_panel.parquet")
        calendar = pd.read_parquet(destination / "calendar_snapshot.parquet")
        archive = pd.read_parquet(destination / "oos_predictions.parquet")
        bridge = create_risk_bundle(destination, panel, calendar, archive, cfg)
        execute_risk(bridge, destination)
        portfolio = destination / "portfolio"
        tables = portfolio_inference(
            pd.read_parquet(portfolio / "portfolio_periods.parquet"),
            pd.read_parquet(portfolio / "risk_calibration_daily.parquet"),
            cfg,
        )
        for name, frame in tables.items():
            frame.to_csv(destination / f"{name}.csv", index=False)
        manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        manifest["stage_replay"]["risk_and_portfolio_inference_recomputed"] = True
        manifest["artifacts"] = {
            str(p.relative_to(destination)): file_sha(p)
            for p in destination.rglob("*")
            if p.is_file() and p != manifest_file
        }
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(manifest, manifest_file)
    print(destination)


if __name__ == "__main__":
    replay(sys.argv[1], Path(sys.argv[2]).resolve())
