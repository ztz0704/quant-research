"""Acquire strict, separately stored A-share data using the optional AKShare extra."""

import argparse

import pandas as pd

from risk_aware_portfolio.data.ashare import (
    acquire_quotes,
    build_real_panel,
    read_universe,
    save_data_manifest,
)
from risk_aware_portfolio.utils.config import load_config
from risk_aware_portfolio.utils.paths import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ashare.yaml")
    parser.add_argument(
        "--refresh", action="store_true", help="Replace raw cache with a new download"
    )
    parser.add_argument(
        "--bypass-proxy", action="store_true", help="Ignore proxy environment explicitly"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    quotes_path = acquire_quotes(config, args.refresh, args.bypass_proxy)
    panel = build_real_panel(pd.read_parquet(quotes_path), read_universe(config), config)
    output = resolve_path(config["data"]["input_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    save_data_manifest(config, output)
    print(
        f"Validated real panel: {panel.shape}, {panel['date'].min().date()} .. {panel['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
