#!/usr/bin/env python3

import base64
import os
import re
import sys
import subprocess
import tempfile
from typing import TypedDict, Optional, List, Tuple
from html.parser import HTMLParser
from html import escape as html_escape


def logln(msg: str):
    print(msg, file=sys.stderr)


_SKIP = {
    "getting-started",
    "e2e-llm-workflows",
}

class Image(TypedDict):
    start: int
    end: int
    src: str
    alt: str
    height_px: Optional[int]
    width_px: Optional[int]
    style: str
    is_md: bool


def _inline_image(img_file: str) -> str:
    with open(img_file, "rb") as f:
        img_data = base64.urlsafe_b64encode(f.read()).decode("utf-8")

    assert isinstance(img_data, str)

    if img_file.endswith(".png"):
        img_typ = "data:image/png"
    elif img_file.endswith(".jpg") or img_file.endswith(".jpeg"):
        img_typ = "data:image/jpeg"
    elif img_file.endswith(".gif"):
        img_typ = "data:image/gif"
    else:
        raise ValueError(f"unsupported image type: {img_file}")
    return f"{img_typ};base64,{img_data}"


def _encode_image(img: Image, dir: str) -> str:
    style = ""
    height = img["height_px"]
    if height is not None:
        style += f"height: {height}px;"
    width = img["width_px"]
    if width is not None:
        style += f"width: {width}px;"
    if img["style"]:
        style += img["style"]

    parts = ["<img"]
    src = img["src"]
    if src:
        src_file = os.path.join(dir, src)
        if os.path.isfile(src_file):
            src = _inline_image(src_file)
        parts.append(f'src="{html_escape(src)}"')
    alt = img["alt"]
    if alt:
        parts.append(f'alt="{html_escape(alt)}"')
    if style:
        parts.append(f'style="{html_escape(style)}"')

    parts.append("/>")
    return " ".join(parts)


def _px_value(s: str) -> int:
    if s.endswith("/"):
        s = s[:-1]
    if s.endswith("px"):
        s = s[:-2]
    return int(s)


class ImgParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._height: Optional[int] = None
        self._width: Optional[int] = None
        self._alt = ""
        self._src = ""
        self._count = 0
        self._style = ""

    def handle_starttag(self, tag, attrs):
        if tag != "img":
            return

        self._count += 1
        if self._count > 1:
            return
        
        for attr in attrs:
            k, v = attr
            if k == "src":
                self._src = v
            elif k == "alt":
                self._alt = v
            elif k == "height":
                self._height = _px_value(v)
            elif k == "width":
                self._width = _px_value(v)
            elif k == "style":
                self._style = v
     
    def count(self) -> int:
        return self._count
    
    def img(self, start: int, end: int) -> Image:
        return Image(
            start=start,
            end=end,
            src=self._src,
            alt=self._alt,
            height_px=self._height,
            width_px=self._width,
            style=self._style,
            is_md=False,
        )
    

def parse_html_img(s: str, start: int, end: int) -> Optional[Image]:
    parser = ImgParser()
    parser.feed(s)

    if parser.count() == 0:
        logln(f"want 1 <img> tag, found none in '{s}'")
        return None
    if parser.count() > 1:
        logln(f"want 1 <img> tag, found {parser.count()} in '{s}'")

    return parser.img(start, end)   


img_tag_regex = re.compile(r"<img src=.*/>", re.MULTILINE)
img_md_regex = re.compile(r"!\[(.*)\]\((.+)\)", re.MULTILINE)


def generate_readme(notebook: str, readme: str) -> None:
    notebook_dir = os.path.dirname(notebook)

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
            md = f.read()

    cursor = 0
    imgs: List[Image] = []
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

        img = Image(
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

    cursor = 0

    final_md = ""
    for img in imgs:
        start = img["start"]
        assert cursor <= start

        if cursor < start:
            final_md += md[cursor:start]
        final_md += _encode_image(img, notebook_dir)

        cursor = img["end"]
    
    if cursor < len(md):
        final_md += md[cursor:]

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

        readme = os.path.join(tmpl_dir, "README.md")
        
        logln(f"Generating {readme}...")

        generate_readme(notebook, readme)
    

if __name__ == "__main__":
    run()