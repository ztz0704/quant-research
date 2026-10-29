"""Validate external OHLCV CSV without inventing historical membership/trading states."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from robust_alpha.data.loader import load_panel, save_panel
from robust_alpha.mining import file_hash

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".parquet":
        parser.error("Output must be Parquet")
    panel = load_panel(args.input)
    save_panel(panel, args.output)
    manifest = {
        "provider_declared_by_user": args.provider,
        "source": str(args.input.resolve()),
        "source_sha256": file_hash(args.input),
        "panel_sha256": file_hash(args.output),
        "rows": len(panel),
        "assets": int(panel.asset.nunique()),
        "converted_at_utc": datetime.now(UTC).isoformat(),
        "historical_publication_time_certified": False,
        "caveat": "Conversion validates schema, not supplier/history authenticity",
    }
    args.output.with_suffix(".import.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
