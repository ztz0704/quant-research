"""Verify real ingestion -> shared Alpha -> ML -> separate Risk; never train or infer Alpha."""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from financial_ml.research_v3_pipeline import build_panel, load_inputs
from financial_ml.shared_v3 import PROJECTS, file_sha

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--load-end", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output = args.output.resolve()
bundle = args.bundle.resolve()
raw, calendar, metadata = load_inputs(
    {"data": {"source": "bundle", "load_end": args.load_end, "bundle_directory": str(bundle)}}
)
cfg = {"data": {"min_history": 60, "min_assets": 20, "min_amount_20": 10000000}}
panel, features = build_panel(raw, calendar, cfg)
assert not panel.eligible.any(), "Unknown availability must not produce ML eligibility"
assert not panel.target_valid.any(), "Unknown auction permissions must not create valid labels"
assert panel.date.max() <= pd.Timestamp(args.load_end)
assert metadata["status"] == "EXPLORATORY_ONLY" and not metadata["formal_ready"]
args.output.mkdir(parents=True, exist_ok=True)
event_path = args.output / "event_schema_input.parquet"
for column in ("execution_asof", "open_price_asof", "close_price_asof", "action_indicator_asof"):
    panel[column] = pd.NaT
panel.to_parquet(event_path, index=False)
calendar_path = args.output / "calendar.parquet"
calendar.to_parquet(calendar_path, index=False)
risk_root = PROJECTS / "risk_aware_portfolio"
statement = (
    "import json,sys,pandas as pd;"
    "from risk_aware_portfolio.backtest.open_events import validate_event_input;"
    "p=pd.read_parquet(sys.argv[1]);c=pd.read_parquet(sys.argv[2]);"
    "validate_event_input(p,c);"
    "assert p.action_indicator_asof.isna().all();"
    "print(json.dumps({'risk_schema_valid':True,'rows':len(p),'asof_fabricated':False}))"
)
completed = subprocess.run(
    [
        str(risk_root / ".venv/Scripts/python.exe"),
        "-c",
        statement,
        str(event_path),
        str(calendar_path),
    ],
    cwd=risk_root,
    text=True,
    capture_output=True,
    check=True,
    timeout=60,
)
result = {
    "alpha_ml_schema_valid": True,
    "ml_features": features,
    "risk": json.loads(completed.stdout),
    "status": "EXPLORATORY_ONLY",
    "formal_ready": False,
    "rows": len(panel),
    "assets": int(panel.asset.nunique()),
    "load_end": args.load_end,
    "metadata_sha256": file_sha(bundle / "metadata.json"),
    "source_manifest_sha256": file_sha(bundle / "manifest.json"),
    "verification_code_sha256": file_sha(Path(__file__)),
    "alpha_trained": False,
}
(args.output / "integration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
