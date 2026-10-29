"""Content-addressed run provenance: input, code, config, lockfile, environment."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from risk_aware_portfolio.utils.config import Config
from risk_aware_portfolio.utils.paths import project_root, resolve_path


def file_hash(path: Path) -> str:
    """Stream SHA256; no entire large input file needs to be held in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, config: Config, input_path: Path) -> None:
    """Persist declared parameters and exact source/input hashes for every main run."""
    root = project_root()
    sources = sorted(list((root / "src").rglob("*.py")) + list((root / "scripts").glob("*.py")))
    content = {
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "input_sha256": file_hash(input_path),
        "lock_sha256": file_hash(root / "uv.lock"),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "cvxpy", "matplotlib", "pyarrow")
        },
        "source_sha256": {p.relative_to(root).as_posix(): file_hash(p) for p in sources},
    }
    if config["data"]["alpha_path"]:
        content["alpha_sha256"] = file_hash(resolve_path(config["data"]["alpha_path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def verify_manifest(path: Path, config: Config, input_path: Path) -> None:
    """Reject stale results after data, config, lockfile or source code changes."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["config"] != config or manifest["input_sha256"] != file_hash(input_path):
        raise ValueError("Stale artifacts: config/data changed; rerun the pipeline")
    if manifest["lock_sha256"] != file_hash(project_root() / "uv.lock"):
        raise ValueError("Stale artifacts: lockfile changed; rerun the pipeline")
    if config["data"]["alpha_path"] and manifest.get("alpha_sha256") != file_hash(
        resolve_path(config["data"]["alpha_path"])
    ):
        raise ValueError("Stale external alpha artifacts; rerun the pipeline")
    for relative, expected in manifest["source_sha256"].items():
        if file_hash(project_root() / relative) != expected:
            raise ValueError(f"Stale artifacts: {relative} changed; rerun the pipeline")
