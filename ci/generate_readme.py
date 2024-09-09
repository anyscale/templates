#!/usr/bin/env python3

import os
import sys
import subprocess
import tempfile


def logln(msg: str):
    print(msg, file=sys.stderr)


_SKIP = {
    "getting-started",
    "e2e-llm-workflows",
}

def generate_readme(notebook: str, readme: str) -> None:
    notebook_dir = os.path.dirname(notebook)
    del notebook_dir

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.check_call([
            "jupyter",
            "nbconvert",
            "--to",
            "markdown",
            notebook,
            "--embed-images",
            "--output-dir",
            tmpdir,
        ])

        with open(os.path.join(tmpdir, "README.md")) as f:
            output = f.read()

    with open(readme, "w") as f:
        f.write(output)


def run():
    templates_dir = "templates"
    tmpls = os.listdir(templates_dir)
    tmpls.sort()

    for tmpl in tmpls:
        if tmpl.startswith("."):
            continue
        if tmpl in _SKIP:
            logln(f"Skipping {tmpl}...")
            continue
        tmpl_dir = os.path.join(templates_dir, tmpl)
        if not os.path.isdir(tmpl_dir):
            continue

        notebook = os.path.join(tmpl_dir, "README.ipynb")
        if not os.path.isfile(notebook):
            continue

        readme = os.path.join(tmpl_dir, "README.md")
        
        logln(f"Generating {readme}...")

        generate_readme(notebook, readme)
        
    

if __name__ == "__main__":
    run()