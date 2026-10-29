"""Resolve configuration paths relative to the repository, never the working directory."""

from pathlib import Path


def project_root() -> Path:
    """Return the installed editable source repository root."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Respect absolute paths and resolve relative paths against a supplied root."""
    path = Path(path)
    return path if path.is_absolute() else (root or project_root()) / path


def ensure_directories(root: Path | None = None) -> None:
    """Create data and output directories without touching existing artifacts."""
    for name in (
        "data/raw",
        "data/interim",
        "data/processed",
        "outputs/figures",
        "outputs/tables",
        "outputs/logs",
        "outputs/backtests",
    ):
        resolve_path(name, root).mkdir(parents=True, exist_ok=True)
