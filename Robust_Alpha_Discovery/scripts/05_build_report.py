"""Run the full pipeline and build deterministic report artifacts."""

from __future__ import annotations

import argparse

from robust_alpha.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    arguments = parser.parse_args()
    artifacts = run_pipeline(arguments.config, stop_after="report")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

