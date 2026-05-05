#!/usr/bin/env python3
"""Pre-commit hook: forbid relative image refs in notebooks and markdown.

A README.md is rendered in places that don't share its source directory
(Anyscale's template gallery, raw-md viewers), so relative image paths
break there. Image refs must use absolute URLs. Convention is GitHub raw:

    https://raw.githubusercontent.com/anyscale/templates/main/<path>

Skipped:
- http(s):// URLs (already absolute)
- data: inline images
- README_files/... (nbconvert auto-generated cell-output thumbnails)

Scope:
- .ipynb files: only markdown cells are inspected
- .md files: the whole file
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

GITHUB_PREFIX = "https://raw.githubusercontent.com/anyscale/templates/main"

ALLOWED_PREFIX = re.compile(r"^(https?://|data:|#|README_files/)")
MARKDOWN_IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
# Match src= with double-quoted, single-quoted, OR unquoted values.
# Without all three a relative ref like <img src='assets/x.png'> would
# bypass the check.
HTML_IMG_SRC = re.compile(
    r'''<img\s[^>]*src=(?:"([^"]+)"|'([^']+)'|([^\s>'"]+))''',
    re.IGNORECASE,
)


def find_relative_in_text(text: str) -> list[str]:
    bad: list[str] = []
    for m in MARKDOWN_IMG.finditer(text):
        url = m.group(2).strip()
        if not ALLOWED_PREFIX.match(url):
            bad.append(url)
    for m in HTML_IMG_SRC.finditer(text):
        url = (m.group(1) or m.group(2) or m.group(3)).strip()
        if not ALLOWED_PREFIX.match(url):
            bad.append(url)
    return bad


def find_relative_refs(path: Path) -> list[tuple[str, str]]:
    """Return [(location, relative_url), ...]."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        out: list[tuple[str, str]] = []
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "markdown":
                continue
            src = cell.get("source", [])
            text = "".join(src) if isinstance(src, list) else src
            for url in find_relative_in_text(text):
                out.append((f"cell {i}", url))
        return out
    if path.suffix == ".md":
        text = path.read_text()
        return [("", url) for url in find_relative_in_text(text)]
    return []


def main(argv: list[str]) -> int:
    rc = 0
    for arg in argv:
        path = Path(arg)
        if path.suffix not in (".ipynb", ".md"):
            continue
        bad = find_relative_refs(path)
        if not bad:
            continue
        rc = 1
        suggestion = f"{GITHUB_PREFIX}/{path.parent}/<rel-path>"
        print(f"\n{path}: {len(bad)} relative image ref(s):", file=sys.stderr)
        for loc, url in bad:
            prefix = f"  {loc}: " if loc else "  "
            print(f"{prefix}{url}", file=sys.stderr)
        print(f"\nUse absolute URLs. Convention: {suggestion}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
