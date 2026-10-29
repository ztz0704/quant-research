"""Verify real source hashes, fold causality, labels, every-event ledgers and final artifacts."""

import argparse
import hashlib
import json
from datetime import datetime, timezone

import nbformat
import numpy as np
import pandas as pd
import yaml

from market_microstructure.data.validation import validate_orderbook
from market_microstructure.features.feature_pipeline import feature_columns
from market_microstructure.utils.config import load_config
from market_microstructure.utils.paths import project_path


def main() -> None:
    """Fail on mixed/stale research, missing figures or an unreconciled event."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-notebooks", action="store_true")
    args = parser.parse_args()
    config = load_config()
    source = yaml.safe_load(project_path("configs/public_sample.yaml").read_text(encoding="utf-8"))
    for file, metadata in source["files"].items():
        with project_path(f"data/raw/szse_public/{file}").open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != metadata["sha256"]:
            raise AssertionError(f"Raw source hash mismatch: {file}")
    book = pd.read_parquet(project_path(config["data"]["input_path"]))
    _, audit = validate_orderbook(book)
    assert audit.removed == 0
    assert set(book.asset) == {"000400.SZ"}
    features = pd.read_parquet(project_path(config["data"]["features_path"]))
    predictions = pd.read_parquet(project_path("data/processed/predictions.parquet"))
    assert not predictions.duplicated(["model", "timestamp", "asset"]).any()
    assert predictions.signal.between(-1, 1).all()
    assert not {"future_return_bps", "direction", "label_end"} & set(feature_columns(features))
    checked = predictions.merge(features[["timestamp", "asset", "future_return_bps"]],
                                 on=["timestamp", "asset"], validate="many_to_one", suffixes=("", "_source"))
    assert len(checked) == len(predictions)
    np.testing.assert_allclose(checked.future_return_bps, checked.future_return_bps_source, equal_nan=True)
    folds = pd.read_csv(project_path("outputs/tables/fold_audit.csv"))
    for row in folds.itertuples():
        assert pd.Timestamp(row.train_max_label_end) < pd.Timestamp(row.validation_start)
        assert pd.Timestamp(row.validation_max_label_end) < pd.Timestamp(row.test_start)
    summary = pd.read_csv(project_path("outputs/tables/execution_summary.csv"))
    max_error, event_count = 0.0, 0
    for row in summary.itertuples():
        events = pd.read_parquet(project_path(f"outputs/backtests/{row.model}_{row.mode}_events.parquet"))
        bridge = events.gross_pnl - events.spread_cost - events.fees - events.slippage - events.impact
        error = float(np.max(np.abs(events.net_pnl - bridge)))
        max_error = max(max_error, error)
        np.testing.assert_allclose(events.net_pnl, bridge, atol=1e-7)
        np.testing.assert_allclose(events.gross_pnl, events.realized_gross_alpha + events.inventory_mark_to_market, atol=1e-7)
        np.testing.assert_allclose(events.equity, events.cash + events.position * events.mid, atol=1e-7)
        assert events.cash.min() >= 0 and events.position.min() >= 0
        assert events.position.max() <= config["strategy"]["max_inventory"]
        assert events.position.iloc[-1] == row.residual_inventory_abs
        event_count += len(events)
    assert summary.closed_lot_count.eq(0).all()
    for name in ("signal_summary", "model_summary", "execution_summary", "pnl_decomposition",
                 "robustness_summary", "optimal_execution_summary"):
        assert not pd.read_csv(project_path(f"outputs/tables/{name}.csv")).empty
    optimal = pd.read_csv(project_path("outputs/tables/optimal_execution_summary.csv"))
    for _, group in optimal.groupby("experiment"):
        ac = group.loc[group.schedule == "Almgren_Chriss", "objective_cny"].iloc[0]
        assert ac <= group.objective_cny.min() + 1e-7
    manifest = json.loads(project_path("outputs/tables/reproducibility_manifest.json").read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    assert config_hash == manifest["config_sha256"], "Report configuration is stale"
    for name, expected in manifest["file_sha256"].items():
        with project_path(name).open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == expected, f"Stale artifact {name}"
    assert manifest["synthetic_primary_results"] is False
    for figure in manifest["figures"]:
        path = project_path("outputs") / figure
        assert path.is_file() and path.stat().st_size > 1000
    notebooks = list(project_path("notebooks").glob("*.ipynb"))
    assert len(notebooks) == 4
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        if args.require_notebooks:
            assert notebook.metadata.get("research_execution", {}).get("status") == "passed"
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    assert cell.execution_count is not None
                    assert all(output.output_type != "error" for output in cell.outputs)
    result = {"verified_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "raw_source_hashes_verified": 2, "snapshot_rows": len(book),
              "oos_rows": len(predictions), "walk_forward_folds": int(folds.fold.nunique()),
              "model_execution_cases": len(summary), "ledger_events_verified": event_count,
              "maximum_pnl_reconciliation_error_cny": max_error,
              "optimal_execution_cases": len(optimal), "figures": len(manifest["figures"]),
              "notebooks": len(notebooks), "notebook_execution_required": args.require_notebooks}
    project_path("outputs/tables/verification_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
