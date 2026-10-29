"""Generate all scientific figures, measured report and four diagnostic notebooks."""
import argparse
import json
import subprocess
import sys

from financial_ml.notebooks import write_notebooks
from financial_ml.reporting import figures, report
from financial_ml.utils import ROOT, config, output, path, sha256

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = config(args.config)
    destination = output(cfg)
    filename = destination / "tables/diagnostics_manifest.json"
    valid = False
    if filename.is_file():
        cached = json.loads(filename.read_text(encoding="utf-8"))
        valid = cached["config"] == cfg and cached["training_manifest_sha256"] == sha256(destination / "tables/training_manifest.json")
        valid &= all(sha256(destination / f"tables/{name}.csv") == digest for name, digest in cached["table_sha256"].items())
    if not valid:
        subprocess.run([sys.executable, str(path("scripts/run_diagnostics.py")), "--config", args.config], cwd=ROOT, check=True)
    generated = figures(cfg)
    report(cfg, generated)
    write_notebooks(cfg)
    print(f"Report: {destination / 'research_report.md'}; figures={len(generated)}")
