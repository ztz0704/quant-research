"""Windows CMD V3 research runner; environments remain separate and versioned."""

import argparse
from pathlib import Path

import financial_ml

expected = Path(__file__).resolve().parents[1] / "src/financial_ml/__init__.py"
if Path(financial_ml.__file__).resolve() != expected:
    raise ImportError("V3 CLI must use this checkout's existing editable ML environment")

from financial_ml.research_v3_pipeline import run_v3  # noqa: E402
from financial_ml.utils import config  # noqa: E402
from financial_ml.verify_v3 import verify_v3  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research_v3_historical.yaml")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(verify_v3(config(args.config)))
    else:
        print(run_v3(config(args.config), diagnostics=not args.no_diagnostics))
