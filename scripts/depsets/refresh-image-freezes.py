#!/usr/bin/env python3
"""Refresh the committed image freezes for a Ray version.

    refresh-image-freezes.py <ray-version> [image ...]

A freeze is the pip package list of a published Anyscale base image. Templates seed
their locks from it, so a lock describes the image the template actually runs on
rather than whatever PyPI served the day it was compiled.

With no image arguments this refreshes every entry in `tracked-images.txt`, and an
image that isn't published yet is skipped with a warning rather than failing the run:
base locks land before every variant exists, and the next scheduled run picks it up.
An image named explicitly on the command line is the opposite case -- you asked for
that one, so failing to get it is an error.

Where a freeze lives, and what makes one usable, are defined once in
`scripts/ray-bump/depset_versions.py`. The daily probe and the fanout resolver
already import them; so does this, rather than keeping another copy in shell.

Its output is data, and lives apart from this: `dependencies/` holds the freezes and
the depset config, `scripts/` holds the automation that maintains them.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

def _repo_root() -> Path:
    """Nearest ancestor dir containing BUILD.yaml, so this script can be relocated."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "BUILD.yaml").is_file():
            return parent
    raise RuntimeError("repo root not found: no BUILD.yaml above this script")


REPO = _repo_root()
sys.path.insert(0, str(REPO / "scripts" / "ray-bump"))

from depset_versions import (  # noqa: E402
    MIN_FREEZE_PACKAGES,
    freeze_path,
    tracked_images,
    usable,
)

BASE = "https://docs.anyscale.com/base-images"
TIMEOUT = 60
# The docs CDN 403s urllib's default `Python-urllib/3.x` agent while serving curl
# fine, so the header is load-bearing -- dropping it fails every fetch outright.
USER_AGENT = "anyscale-templates-refresh-image-freezes"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def image_index() -> dict[str, str]:
    """imageName -> filename, fetched once for the whole run.

    Skipping entries missing either key: the index lists thousands of unrelated
    images, and one malformed entry must not break the lookup for every image we
    track.
    """
    return {
        name: filename
        for entry in json.loads(_get(f"{BASE}/index.json"))
        if (name := entry.get("imageName")) and (filename := entry.get("filename"))
    }


def resolve(index: dict[str, str], image: str) -> str | None:
    """The index suffixes CPU variants `-cpu`; BUILD.yaml references the bare alias."""
    return next((name for name in (image, f"{image}-cpu") if name in index), None)


def pins(freeze: Path) -> int:
    return sum(1 for line in freeze.read_text().splitlines() if "==" in line)


def write_freeze(image: str, filename: str, dest: Path) -> int:
    """Write `image`'s package list to `dest`, refusing an implausible one."""
    packages = json.loads(_get(f"{BASE}/{filename}"))["pip"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = Path(f"{dest}.part")
    part.write_text(
        "\n".join(
            [
                f"# pip freeze of {image}",
                "# Regenerate with scripts/depsets/refresh-image-freezes.py",
                *packages,
            ]
        )
        + "\n"
    )
    # A truncated download or an error page leaves a short file that still parses as a
    # freeze, and every lock seeded from it would silently float. Only a plausible file
    # wins the real name, so a failure leaves the previous freeze in place.
    if not usable(part):
        count = pins(part)
        part.unlink()
        raise ValueError(
            f"got {count} packages, expected at least {MIN_FREEZE_PACKAGES} "
            "(truncated or an error page)"
        )
    part.replace(dest)
    return pins(dest)


FETCH_ERRORS = (urllib.error.URLError, OSError, ValueError, KeyError, json.JSONDecodeError)


def main(argv: list[str]) -> int:
    if not argv:
        print(f"Usage: {Path(__file__).name} <ray-version> [image ...]", file=sys.stderr)
        return 2

    version, named = argv[0], argv[1:]
    images = [img.replace("{version}", version) for img in (named or tracked_images())]

    try:
        index = image_index()
    except FETCH_ERRORS as err:
        print(f"::error::cannot read {BASE}/index.json: {err}", file=sys.stderr)
        return 1

    failed, skipped = [], 0
    for image in images:
        dest = freeze_path(image)
        try:
            resolved = resolve(index, image)
            if resolved is None:
                raise ValueError(f"not listed in {BASE}/index.json")
            if resolved != image:
                print(f"note: {image} resolved via {resolved}")
            count = write_freeze(image, index[resolved], dest)
            print(f"Fetched {count} packages for {image} -> {dest.relative_to(REPO)}")
        except FETCH_ERRORS as err:
            if named:
                print(f"::error::{image}: {err}", file=sys.stderr)
                failed.append(image)
            else:
                # Unpublished and fetch-failed look the same from here; either way this
                # freeze is not current, which is what the completeness gate checks.
                print(
                    f"::warning::no usable freeze for {image} ({err}) — its templates "
                    "stay on the previous version's freeze"
                )
                skipped += 1

    if failed:
        return 1
    print(f"Refreshed image freezes for Ray {version} ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
