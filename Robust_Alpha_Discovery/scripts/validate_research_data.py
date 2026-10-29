"""Validate a canonical offline PIT bundle without computing alpha or labels."""

import argparse
import json
from pathlib import Path

from robust_alpha.data.research_adapters import read_bundle
from robust_alpha.data.research_contract import (
    normalize_dynamic_panel,
    quality_report,
    require_formal_quality,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--date-end", required=True, help="Explicit pre-read date boundary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-formal", action="store_true")
    args = parser.parse_args()
    raw, calendar, metadata = read_bundle(args.bundle, args.date_end)
    panel = normalize_dynamic_panel(raw, calendar)
    report = quality_report(panel, metadata)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    if args.require_formal:
        require_formal_quality(report)
    print(destination)
