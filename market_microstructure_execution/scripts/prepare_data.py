"""Generate a reproducible synthetic book or validate an explicitly configured file."""

import json

from market_microstructure.data.loader import load_orderbook, save_orderbook
from market_microstructure.data.synthetic import SyntheticConfig, generate_orderbook
from market_microstructure.data.validation import validate_orderbook
from market_microstructure.utils.config import script_config
from market_microstructure.utils.logging import configure_logging
from market_microstructure.utils.paths import ensure_directories, research_path


def main() -> None:
    """Persist the book and its mandatory row audit."""
    config = script_config(__doc__)
    ensure_directories()
    logger = configure_logging(config["logging"]["level"])
    if config["data"]["source"] == "synthetic":
        frame = generate_orderbook(SyntheticConfig(**config["synthetic"]), config["seed"])
        save_orderbook(frame, research_path(config, config["data"]["raw_path"]))
        clean, report = validate_orderbook(frame)
    else:
        clean, report = load_orderbook(
            research_path(config, config["data"]["raw_path"]), mode=config["data"]["invalid_policy"],
            max_removed_fraction=config["data"]["max_removed_fraction"],
        )
    save_orderbook(clean, research_path(config, config["data"]["input_path"]))
    research_path(config, "outputs/tables/data_audit.json").write_text(
        json.dumps({"source": config["data"]["source"], "provider": config["data"].get("provider"),
                    "raw_path": config["data"]["raw_path"], "seed": config["seed"], **report.to_dict()},
                   indent=2), encoding="utf-8",
    )
    logger.info("Prepared book: %s", report.to_dict())


if __name__ == "__main__":
    main()
