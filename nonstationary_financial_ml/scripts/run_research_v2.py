"""V2 engineering/historical integration CLI; use the existing ML environment."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for project in (ROOT, ROOT.parent / "Robust_Alpha_Discovery", ROOT.parent / "risk_aware_portfolio"):
    sys.path.insert(0, str(project / "src"))

from financial_ml.upgrade_pipeline import run_v2  # noqa: E402
from financial_ml.utils import config  # noqa: E402
from financial_ml.verify_v2 import verify_v2  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research_v2_historical.yaml")
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--include-holdout", action="store_true")
    args = parser.parse_args()
    settings = config(args.config)
    if args.verify:
        print(verify_v2(settings))
    else:
        print(run_v2(settings, args.include_holdout, args.quality_only))
