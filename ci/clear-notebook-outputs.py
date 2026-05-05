#!/usr/bin/env python3
"""Pre-commit hook: clear outputs from notebook code cells in-place.

Committed notebook outputs cause noisy diffs (every re-run shifts execution
counts and timestamps) and risk leaking sensitive data (paths, secrets,
debug prints). This hook clears outputs and execution_count in any code
cell that has them, then exits non-zero so pre-commit reports "files were
modified by this hook" — re-stage and commit again.

Mutation is delegated to `jupyter-nbconvert --clear-output --inplace` so
the byte-output is canonical and matches what CI would produce.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def needs_clearing(path: Path) -> bool:
    nb = json.loads(path.read_text())
    return any(
        cell.get("cell_type") == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
        for cell in nb.get("cells", [])
    )


def main(argv: list[str]) -> int:
    modified: list[Path] = []
    for arg in argv:
        path = Path(arg)
        if path.suffix != ".ipynb":
            continue
        try:
            if not needs_clearing(path):
                continue
        except (OSError, json.JSONDecodeError) as e:
            print(f"\n{path}: skipped — could not parse notebook ({e})", file=sys.stderr)
            continue

        subprocess.run(
            ["jupyter-nbconvert", "--clear-output", "--inplace", str(path)],
            check=True,
        )
        modified.append(path)

    if modified:
        print(
            f"\nCleared outputs in {len(modified)} notebook(s):",
            file=sys.stderr,
        )
        for p in modified:
            print(f"  {p}", file=sys.stderr)
        print("\nRe-stage and commit again.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
