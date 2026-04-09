#!/usr/bin/env python3
"""Convert a README.ipynb notebook to README.md.

Usage:
    python ci/notebook_to_readme.py templates/ray-data-llm/README.ipynb
    python ci/notebook_to_readme.py  # converts all templates/*/README.ipynb files

Markdown cells are emitted as-is.  Code cells are wrapped in ```python fences.
Relative <img src="..."> and ![...](assets/...) paths are rewritten to
absolute GitHub raw URLs, matching the behaviour of ci/auto-generate-readme.sh.
"""

import json
import re
import sys
from pathlib import Path

REPO_PREFIX = "https://raw.githubusercontent.com/anyscale/templates/main"

SKIP_CONVERSION = {
    "e2e-llm-workflows",
    "image-search-and-classification",
    "entity-recognition-with-llms",
}

SKIP_TIME_CHECK = {
    "getting-started",
    "e2e-llm-workflows",
    "ray-summit-multi-modal-search",
    "image-search-and-classification",
    "entity-recognition-with-llms",
}


def notebook_to_markdown(notebook_path: Path) -> str:
    """Return the README.md contents generated from *notebook_path*."""
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])

    parts: list[str] = []
    for cell in cells:
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        if cell_type == "markdown":
            parts.append(source)
        elif cell_type == "code":
            parts.append(f"```python\n{source}\n```")

    return "\n\n".join(parts) + "\n"


def fix_image_paths(content: str, notebook_dir: str) -> str:
    """Rewrite relative image paths to absolute GitHub raw URLs."""
    # <img src="relative/path"> → <img src="REPO_PREFIX/dir/relative/path">
    def replace_img_tag(m: re.Match) -> str:
        src = m.group(1)
        return f'<img src="{REPO_PREFIX}/{notebook_dir}/{src}"'

    content = re.sub(
        r'<img src="(?!https?://)(?!/)([^"]+)"',
        replace_img_tag,
        content,
    )

    # ![alt](assets/foo) → <img src="REPO_PREFIX/dir/assets/foo"/>
    def replace_md_image(m: re.Match) -> str:
        path = m.group(1)
        return f'<img src="{REPO_PREFIX}/{notebook_dir}/{path}"/>'

    content = re.sub(
        r"!\[.*?\]\((assets/[^)]+)\)",
        replace_md_image,
        content,
    )

    return content


def process_notebook(notebook_path: Path, repo_root: Path) -> None:
    notebook_rel = notebook_path.relative_to(repo_root)
    notebook_dir = str(notebook_rel.parent)
    tmpl_name = notebook_rel.parts[1] if len(notebook_rel.parts) > 1 else ""

    if tmpl_name in SKIP_CONVERSION:
        print(f"Skipping README generation for {notebook_rel}")
        return

    notebook_text = notebook_path.read_text(encoding="utf-8")
    if tmpl_name not in SKIP_TIME_CHECK and "Time to complete" not in notebook_text:
        print(
            f"LINT ERROR: {notebook_rel} must include 'Time to complete' statement, failing."
        )
        sys.exit(1)

    print(f"===== Processing {notebook_rel}")
    content = notebook_to_markdown(notebook_path)
    content = fix_image_paths(content, notebook_dir)

    readme_path = notebook_path.parent / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    print(f"      Wrote {readme_path.relative_to(repo_root)}")


def find_repo_root() -> Path:
    import subprocess

    root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    return Path(root)


def main() -> None:
    repo_root = find_repo_root()

    if len(sys.argv) > 1:
        notebooks = [Path(sys.argv[1])]
        if not notebooks[0].is_absolute():
            notebooks = [repo_root / notebooks[0]]
    else:
        notebooks = sorted((repo_root / "templates").rglob("README.ipynb"))

    for nb in notebooks:
        process_notebook(nb, repo_root)

    print("Done.")


if __name__ == "__main__":
    main()
