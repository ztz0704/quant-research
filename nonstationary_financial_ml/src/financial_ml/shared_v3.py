"""Explicit local Alpha package binding and versioned subprocess Risk bridge.

No search-path injection: bind one resolved package root, reject an already
loaded different checkout, and hash every module in the run manifest.
Risk executes in its own existing environment via Parquet/JSON v3, not imports.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

INTERFACE_VERSION = "ashare-open-events-v3"
PROJECTS = Path(__file__).resolve().parents[3]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_alpha() -> dict:
    root = (PROJECTS / "Robust_Alpha_Discovery/src/robust_alpha").resolve()
    expected = root / "__init__.py"
    loaded = sys.modules.get("robust_alpha")
    if loaded is not None and Path(loaded.__file__).resolve() != expected:
        raise ImportError("Different Alpha checkout already loaded; refusing ambiguous binding")
    if loaded is None:
        spec = importlib.util.spec_from_file_location(
            "robust_alpha", expected, submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["robust_alpha"] = module
        spec.loader.exec_module(module)
    from robust_alpha.data.research_contract import VERSION

    if VERSION != "ashare-research-v2":
        raise ImportError("Unsupported shared Alpha schema")
    return {"root": str(root), "schema": VERSION, "init_sha256": file_sha(expected)}


def implementation_hashes() -> dict:
    files = []
    for name in ("Robust_Alpha_Discovery", "nonstationary_financial_ml", "risk_aware_portfolio"):
        project = PROJECTS / name
        files.extend((project / "src").rglob("*.py"))
        files.extend((project / "scripts").rglob("*.py"))
        files.extend([project / "pyproject.toml", project / "uv.lock"])
    return {str(p.relative_to(PROJECTS)): file_sha(p) for p in sorted(files)}
