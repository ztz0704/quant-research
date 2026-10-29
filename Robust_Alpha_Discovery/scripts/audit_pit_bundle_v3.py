"""Audit a canonical offline bundle; never fetch, purchase or attest vendor PIT."""

import argparse
import json
from pathlib import Path

import pandas as pd

import robust_alpha

expected = Path(__file__).resolve().parents[1] / "src/robust_alpha/__init__.py"
if Path(robust_alpha.__file__).resolve() != expected:
    raise ImportError("Use this checkout's existing Alpha environment")

from robust_alpha.data.pit_audit_v3 import audit_pit_v3  # noqa: E402
from robust_alpha.data.research_adapters import read_bundle  # noqa: E402
from robust_alpha.data.research_contract import digest_file, normalize_dynamic_panel  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--load-end", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw, calendar, metadata = read_bundle(args.bundle, args.load_end)
    evidence = {}
    for kind, info in metadata.get("audit_files", {}).items():
        name = info["filename"]
        if Path(name).name != name or digest_file(args.bundle / name) != info["sha256"]:
            raise ValueError("Stale/unsafe primary audit evidence file")
        frame = pd.read_parquet(args.bundle / name)
        if "date" in frame:
            frame = frame.loc[pd.to_datetime(frame.date).le(pd.Timestamp(args.load_end))]
        evidence[kind] = frame
    audited = audit_pit_v3(
        normalize_dynamic_panel(raw, calendar),
        metadata,
        publication_log=evidence.get("publication_log"),
        lifecycle_registry=evidence.get("lifecycle_registry"),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = audited.pop("report")
    report["input_hashes"] = {p.name: digest_file(p) for p in args.bundle.iterdir() if p.is_file()}
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    for name, frame in audited.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
    print("PASS_INTERNAL_AUDIT" if report["formal_ready"] else "BLOCKED_FORMAL_PIT")
