"""Strip execution timing metadata from notebooks, with a CLI taking the files to check.

Executing a notebook records a wall-clock timestamp per cell under ``metadata.execution``, so
re-running the documentation notebooks rewrites every code cell with the time it happened to run at.
Nothing reads it, and the docs are built from the stored outputs, so it is only churn.
"""

import json
import sys
from pathlib import Path


def strip_timing(path: Path) -> bool:
    """Remove per-cell execution timing from the notebook, reporting whether anything changed."""
    notebook = json.loads(path.read_text())

    stripped = [cell for cell in notebook["cells"] if cell.get("metadata", {}).pop("execution", None) is not None]
    if not stripped:
        return False

    # Match the formatting notebooks are already stored in, so the diff is the timing alone.
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return True


def main() -> int:
    """Strip each notebook named on the command line, failing if any of them needed it."""
    rewritten = [path for path in (Path(arg) for arg in sys.argv[1:]) if strip_timing(path)]

    for path in rewritten:
        print(f"Stripped execution timing metadata from {path}")

    return 1 if rewritten else 0


if __name__ == "__main__":
    sys.exit(main())
