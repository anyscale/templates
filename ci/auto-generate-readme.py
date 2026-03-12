#!/usr/bin/env python3
"""Pre-commit hook wrapper that filters changed files to affected templates
and concurrently runs ci/auto-generate-readme.sh on each README.ipynb."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    MAX_WORKERS = max(1, int(os.environ.get("MAX_JOBS", 8)))
except ValueError:
    MAX_WORKERS = 8


def find_affected_notebooks(changed_files):
    """Given a list of changed file paths, return README.ipynb files
    in the affected template directories."""
    # Extract unique template directories (templates/<name>)
    tmpl_dirs = set()
    for f in changed_files:
        parts = Path(f).parts
        if len(parts) >= 2 and parts[0] == "templates":
            tmpl_dirs.add(str(Path(parts[0]) / parts[1]))

    if not tmpl_dirs:
        return []

    # Find README.ipynb files within the affected template directories
    notebooks = []
    for tmpl_dir in sorted(tmpl_dirs):
        for notebook in Path(tmpl_dir).rglob("README.ipynb"):
            notebooks.append(str(notebook))

    return notebooks


def run_readme_generation(notebook_path, repo_root):
    """Run the bash script on a single notebook. Returns (path, returncode, output)."""
    script = os.path.join(repo_root, "ci", "auto-generate-readme.sh")
    result = subprocess.run(
        [script, notebook_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=repo_root,
    )
    return notebook_path, result.returncode, result.stdout


def main():
    changed_files = sys.argv[1:]

    if not changed_files:
        print("No files provided; nothing to do.")
        return 0

    notebooks = find_affected_notebooks(changed_files)
    if not notebooks:
        print("No template directories affected; nothing to do.")
        return 0

    repo_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()

    failed = False
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_readme_generation, nb, repo_root): nb
            for nb in notebooks
        }
        for future in as_completed(futures):
            try:
                notebook_path, returncode, output = future.result()
            except Exception as e:
                print(f"FAILED: {futures[future]} ({e})")
                failed = True
                continue
            if output.strip():
                print(output.strip())
            if returncode != 0:
                print(f"FAILED: {notebook_path} (exit code {returncode})")
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
