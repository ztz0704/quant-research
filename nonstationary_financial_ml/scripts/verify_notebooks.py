"""Execute all four read-only notebooks with the project's installed kernel."""
from datetime import datetime, timezone

import nbformat
from nbclient import NotebookClient

from financial_ml.utils import ROOT, config, output, sha256, write_json

if __name__ == "__main__":
    cfg = config()
    records = []
    filenames = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if len(filenames) != 4:
        raise ValueError("Expected four notebooks")
    for filename in filenames:
        notebook = nbformat.read(filename, as_version=4)
        NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if not all(cell.execution_count is not None for cell in cells):
            raise AssertionError(f"Unexecuted cells: {filename}")
        nbformat.write(notebook, filename)
        records.append({"notebook": filename.name, "executed_code_cells": len(cells), "sha256": sha256(filename), "status": "passed"})
        print(f"Executed {filename.name}: {len(cells)} code cells", flush=True)
    write_json({"checked_at_utc": datetime.now(timezone.utc).isoformat(), "notebooks": records}, output(cfg) / "tables/notebook_execution.json")
