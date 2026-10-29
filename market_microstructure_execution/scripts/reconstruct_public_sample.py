"""Reconstruct the downloaded real SZSE sample; audit rather than fabricate missing states."""

import json

from market_microstructure.data.loader import save_orderbook
from market_microstructure.data.reconstruction import load_szse_messages, reconstruct_snapshots
from market_microstructure.utils.logging import configure_logging
from market_microstructure.utils.paths import project_path


def main() -> None:
    """Replay original messages once, preserving raw downloads and their checksums."""
    logger = configure_logging()
    messages, input_audit = load_szse_messages(
        str(project_path("data/raw/szse_public/orders.csv")), str(project_path("data/raw/szse_public/trans.csv")))
    snapshots, replay_audit = reconstruct_snapshots(messages)
    audit = {**input_audit, **replay_audit}
    save_orderbook(snapshots, project_path("data/raw/ashare_orderbook.parquet"))
    project_path("outputs/tables/reconstruction_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    logger.info("Real SZSE reconstruction: %s", audit)


if __name__ == "__main__":
    main()
