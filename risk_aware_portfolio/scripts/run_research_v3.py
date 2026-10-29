"""Windows CMD: .venv\\Scripts\\python.exe scripts\\run_research_v3.py --bundle PATH"""

import argparse
from pathlib import Path

import risk_aware_portfolio

expected = Path(__file__).resolve().parents[1] / "src/risk_aware_portfolio/__init__.py"
if Path(risk_aware_portfolio.__file__).resolve() != expected:
    raise ImportError("Risk v3 CLI must use this checkout's existing editable environment")

from risk_aware_portfolio.backtest.research_v3 import run_bundle  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    print(run_bundle(parser.parse_args().bundle.resolve()))
