"""Project-relative paths, deterministic configuration and artifact fingerprints."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def path(value: str | Path) -> Path:
    """Resolve configuration paths relative to this project, never the shell cwd."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def config(filename: str = "configs/base.yaml") -> dict[str, Any]:
    """Load a project configuration."""
    def read(candidate: Path, seen: set[Path]) -> dict:
        candidate = candidate.resolve()
        if candidate in seen:
            raise ValueError("Circular configuration inheritance")
        current = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        parent = current.pop("extends", None)
        if parent is None:
            return current
        inherited = read(candidate.parent / parent, seen | {candidate})
        def merge(base: dict, overrides: dict) -> dict:
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    base[key] = merge(base[key], value)
                else:
                    base[key] = value
            return base
        return merge(inherited, current)
    return read(path(filename), set())


def output(cfg: dict[str, Any]) -> Path:
    """Route synthetic experiments into a visibly separate engineering directory."""
    target = path(cfg["output"]["directory"])
    if cfg["data"]["source"] == "synthetic":
        target = target / "engineering_validation"
    for child in ("tables", "figures", "models", "predictions", "logs"):
        (target / child).mkdir(parents=True, exist_ok=True)
    return target


def sha256(filename: Path) -> str:
    """Hash an immutable input or output using bounded memory."""
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(value: Any, filename: Path) -> None:
    """Write strict JSON; nonfinite research metrics must be represented explicitly."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def logger(cfg: dict[str, Any], stage: str) -> logging.Logger:
    """Use a file and concise console logging, without per-row noise."""
    result = logging.getLogger(stage)
    if not result.handlers:
        result.setLevel(logging.INFO)
        handlers = [logging.StreamHandler(), logging.FileHandler(output(cfg) / f"logs/{stage}.log", encoding="utf-8")]
        for handler in handlers:
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            result.addHandler(handler)
    return result
