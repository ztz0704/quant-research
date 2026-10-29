"""Build and validate the aligned research panel."""

from __future__ import annotations

import argparse

from robust_alpha.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    arguments = parser.parse_args()
    run_pipeline(arguments.config, stop_after="panel")


if __name__ == "__main__":
    main()

