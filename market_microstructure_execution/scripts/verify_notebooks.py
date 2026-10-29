"""Execute and validate every research notebook in the locked project kernel."""

import asyncio
import json
import os
import time
from datetime import datetime, timezone

import nbformat
from nbclient import NotebookClient

from market_microstructure.utils.paths import PROJECT_ROOT, project_path


def main() -> None:
    """Save executed notebooks only after successful completion; record cell counts/timing."""
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    rows = []
    for path in sorted(project_path("notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        start = time.monotonic()
        print(f"Executing {path.name}", flush=True)
        client = NotebookClient(notebook, timeout=180, kernel_name="python3",
                                resources={"metadata": {"path": str(PROJECT_ROOT)}})
        executed = client.execute()
        code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
        if any(cell.execution_count is None for cell in code_cells):
            raise RuntimeError(f"Unexecuted cells in {path.name}")
        if any(output.output_type == "error" for cell in code_cells for output in cell.outputs):
            raise RuntimeError(f"Error output in {path.name}")
        executed.metadata["research_execution"] = {"status": "passed", "data": "real public SZSE reconstruction"}
        nbformat.write(executed, path)
        rows.append({"notebook": path.name, "status": "passed", "code_cells": len(code_cells),
                     "duration_seconds": round(time.monotonic() - start, 2)})
        print(f"Passed {path.name}: {len(code_cells)} code cells", flush=True)
    if len(rows) != 4:
        raise RuntimeError("Expected exactly four research notebooks")
    project_path("outputs/tables/notebook_execution.json").write_text(
        json.dumps({"verified_utc": datetime.now(timezone.utc).isoformat(), "notebooks": rows}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
