#!/usr/bin/env python3
"""Pre-commit hook: notebooks must have cleared outputs.

Committed notebook outputs cause noisy diffs (every re-run shifts execution
counts and timestamps) and risk leaking sensitive data (paths, secrets,
debug prints). Clear outputs before committing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def cells_with_outputs(path: Path) -> list[int]:
    nb = json.loads(path.read_text())
    return [
        i
        for i, cell in enumerate(nb.get("cells", []))
        if cell.get("cell_type") == "code" and cell.get("outputs")
    ]


def main(argv: list[str]) -> int:
    rc = 0
    for arg in argv:
        path = Path(arg)
        if path.suffix != ".ipynb":
            continue
        bad = cells_with_outputs(path)
        if not bad:
            continue
        rc = 1
        preview = ", ".join(str(i) for i in bad[:10]) + (", ..." if len(bad) > 10 else "")
        print(
            f"\n{path}: {len(bad)} code cell(s) have outputs (cells {preview})",
            file=sys.stderr,
        )
        print("\nClear with:", file=sys.stderr)
        print(f"    jupyter-nbconvert --clear-output --inplace {path}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
