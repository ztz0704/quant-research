"""Paths resolve against the project, independently of the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_path(value: str | Path) -> Path:
    """Resolve a configured relative path against the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_directories() -> None:
    """Create the documented data and output directories."""
    for name in (
        "data/raw", "data/interim", "data/processed", "outputs/figures", "outputs/tables",
        "outputs/backtests", "outputs/logs", "outputs/models", "notebooks",
    ):
        project_path(name).mkdir(parents=True, exist_ok=True)


def research_path(config: dict, value: str | Path) -> Path:
    """Isolate engineering outputs so a synthetic run cannot overwrite real research."""
    relative = str(value).replace("\\", "/")
    if config["data"]["source"] == "synthetic":
        if relative.startswith("outputs/"):
            relative = relative.replace("outputs/", "outputs/engineering_validation/latest/", 1)
        elif relative == "data/processed/predictions.parquet":
            relative = "data/interim/synthetic_predictions.parquet"
    result = project_path(relative)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result
