#!/usr/bin/env python3
"""Pre-commit hook: forbid relative image refs in README.ipynb markdown cells.

A README.md is rendered in places that don't share the .ipynb's working
directory (e.g., Anyscale's template gallery), so relative image paths
break there. Image refs must use absolute URLs. Convention is GitHub raw:

    https://raw.githubusercontent.com/anyscale/templates/main/<path>

Skipped:
- http(s):// URLs (already absolute)
- data: inline images
- README_files/... (nbconvert auto-generated cell-output thumbnails)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

GITHUB_PREFIX = "https://raw.githubusercontent.com/anyscale/templates/main"

ALLOWED_PREFIX = re.compile(r"^(https?://|data:|#|README_files/)")
MARKDOWN_IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMG_SRC = re.compile(r'<img\s[^>]*src="([^"]+)"')


def find_relative_refs(notebook_path: Path) -> list[tuple[int, str]]:
    """Return [(cell_index, relative_url), ...] for relative image refs."""
    nb = json.loads(notebook_path.read_text())
    bad: list[tuple[int, str]] = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", [])
        text = "".join(src) if isinstance(src, list) else src
        for m in MARKDOWN_IMG.finditer(text):
            url = m.group(2).strip()
            if not ALLOWED_PREFIX.match(url):
                bad.append((i, url))
        for m in HTML_IMG_SRC.finditer(text):
            url = m.group(1).strip()
            if not ALLOWED_PREFIX.match(url):
                bad.append((i, url))
    return bad


def main(argv: list[str]) -> int:
    rc = 0
    for arg in argv:
        path = Path(arg)
        if path.name != "README.ipynb":
            continue
        bad = find_relative_refs(path)
        if not bad:
            continue
        rc = 1
        suggestion = f"{GITHUB_PREFIX}/{path.parent}/<rel-path>"
        print(f"\n{path}: {len(bad)} relative image ref(s) in markdown cells:", file=sys.stderr)
        for cell_idx, url in bad:
            print(f"  cell {cell_idx}: {url}", file=sys.stderr)
        print(f"\nUse absolute URLs. Convention: {suggestion}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
