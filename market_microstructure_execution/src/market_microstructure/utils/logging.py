"""File logs include DEBUG events; console output stays at INFO or above."""

import logging

from market_microstructure.utils.paths import ensure_directories, project_path


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure the package logger once for a command line process."""
    ensure_directories()
    logger = logging.getLogger("market_microstructure")
    logger.setLevel(level.upper())
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = logging.FileHandler(project_path("outputs/logs/research.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.propagate = False
    return logger
