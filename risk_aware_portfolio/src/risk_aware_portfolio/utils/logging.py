"""Per-run standard Python logging with no silent numerical failures."""

import logging
from pathlib import Path

from risk_aware_portfolio.utils.paths import resolve_path


def configure_logging(name: str, directory: Path | None = None) -> logging.Logger:
    """Replace prior run handlers and log to UTF-8 file and console."""
    logger = logging.getLogger("risk_aware_portfolio")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    path = (directory or resolve_path("outputs/logs")) / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setLevel(
            logging.INFO if isinstance(handler, logging.FileHandler) else logging.WARNING
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger
