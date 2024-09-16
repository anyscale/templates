#!/usr/bin/env python3


import os
import re
import sys
import subprocess
import tempfile
from typing import TypedDict, Optional, List, Tuple
from html.parser import HTMLParser



def logln(msg: str):
    print(msg, file=sys.stderr)


_SKIP = {
    "getting-started",
    "e2e-llm-workflows",
}

img_tag_regex = re.compile(r"<img src=.*/>", re.MULTILINE)
img_md_regex = re.compile(r"!\[(.*)\]\((.+)\)", re.MULTILINE)


def _normalize_images(md: str, dir: str, imgs: List[MarkdownImage], inline: bool) -> str:
    cursor = 0

    output = ""
    for img in imgs:
        start = img["start"]
        assert cursor <= start

        if cursor < start:
            output += md[cursor:start]
        output += _md_image_to_html(img, dir)

        cursor = img["end"]
    
    if cursor < len(md):
        output += md[cursor:]

    return output


def generate_readme(notebook: str, github_readme: str, doc_readme: str) -> None:
    notebook_dir = os.path.dirname(notebook)

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.check_call([
            "jupyter",
            "nbconvert",
            "--to",
            "markdown",
            notebook,
            "--output-dir",
            tmpdir,
        ])

        with open(os.path.join(tmpdir, "README.md")) as f:
            md = f.read()

    cursor = 0
    imgs: List[MarkdownImage] = []
    while True:
        match = img_tag_regex.search(md, cursor)
        if match is None:
            break
        cursor = match.end()

        img = parse_html_img(match.group(0), match.start(), match.end())
        if img is not None:
            imgs.append(img)

    cursor = 0
    while True:
        match = img_md_regex.search(md, cursor)
        if match is None:
            break
        cursor = match.end()

        img = MarkdownImage(
            start=match.start(),
            end=match.end(),
            alt=match.group(1),
            src=match.group(2),
            height_px=None,
            width_px=None,
            style="",
            is_md=True,
        )
        imgs.append(img)

    for img in imgs:
        start = img["start"]
        end = img["end"]
        if img["is_md"]:
            logln(f"img md: {start}-{end} " + md[img["start"]:img["end"]])
        else:
            logln(f"img tag: {start}-{end} " + md[img["start"]:img["end"]])

    imgs.sort(key=(lambda img: (img["start"], img["end"])))

    
    with open(readme, "w") as f:
        f.write(final_md)


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

        github_readme = os.path.join(tmpl_dir, "README.md")
        doc_readme = os.path.join(tmpl_dir, "README-doc.md")
        
        logln(f"Generating README...")

        generate_readme(notebook, github_readme, doc_readme)
    

if __name__ == "__main__":
    run()
