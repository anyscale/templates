from typing import TypedDict, Optional
from html import escape as html_escape
import os.path
import base64
from html.parser import HTMLParser

from .logging import logln


class MarkdownImage(TypedDict):
    """Represents a image in a markdown file."""
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
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")

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


def _is_http_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def to_html(img: MarkdownImage, dir: str, inline: bool = False) -> str:
    """Convert a markdown image into an HTML img tag, optionally inlining it."""
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
        if inline and not _is_http_url(src):
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
    # when the image tag ends with '/>', the slash gets mistakenly included
    # in the height/width attribute value. so we remove it if it is here.
    if s.endswith("/"):
        s = s[:-1]
    if s.endswith("px"):
        s = s[:-2]
    return int(s)


class ImgTagParser(HTMLParser):
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
    
    def img(self, start: int, end: int) -> MarkdownImage:
        return MarkdownImage(
            start=start,
            end=end,
            src=self._src,
            alt=self._alt,
            height_px=self._height,
            width_px=self._width,
            style=self._style,
            is_md=False,
        )


def parse_html_img(s: str, start: int, end: int) -> Optional[MarkdownImage]:
    parser = ImgTagParser()
    parser.feed(s)

    if parser.count() == 0:
        logln(f"want 1 <img> tag, found none in '{s}'")
        return None
    if parser.count() > 1:
        logln(f"want 1 <img> tag, found {parser.count()} in '{s}'")

    return parser.img(start, end)   
