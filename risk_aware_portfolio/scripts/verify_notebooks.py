"""Execute all notebooks with the project interpreter and a temporary kernel spec."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from risk_aware_portfolio.utils.paths import project_root, resolve_path


def main() -> None:
    cache = resolve_path(".cache/notebook-kernels")
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache) as temporary:
        kernel = Path(temporary) / "risk-aware"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Risk-aware project Python",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        specs = KernelSpecManager(kernel_dirs=[temporary])
        for path in sorted(resolve_path("notebooks").glob("*.ipynb")):
            manager = KernelManager(kernel_name="risk-aware", kernel_spec_manager=specs)
            notebook = nbformat.read(path, as_version=4)
            client = NotebookClient(
                notebook,
                km=manager,
                timeout=180,
                resources={"metadata": {"path": str(project_root())}},
            )
            client.execute(cleanup_kc=True)
            nbformat.write(notebook, path)
            print(f"Executed: {path.name}", flush=True)


if __name__ == "__main__":
    main()
