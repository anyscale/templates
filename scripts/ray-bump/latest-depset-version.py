#!/usr/bin/env python3
"""Resolve the target Ray version for a fanout from dependencies/depsets/.

A version is "complete" once it has both:
  dependencies/depsets/ray_<v>_img_py<PY>.lock   (base image lock)
  a freeze for EVERY image in dependencies/images/tracked-images.txt
With no args this prints
the newest complete version; with --require <v> it validates that <v> is complete
(and echoes it). Exits non-zero with a message on stderr when there's nothing to
resolve — so the caller can fail closed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

def _repo_root() -> Path:
    """Nearest ancestor dir containing BUILD.yaml (robust to where this script lives)."""
    for p in Path(__file__).resolve().parents:
        if (p / "BUILD.yaml").is_file():
            return p
    raise RuntimeError("repo root not found: no BUILD.yaml above this script")


DEPSETS = _repo_root() / "dependencies" / "depsets"
IMAGES = _repo_root() / "dependencies" / "images"


def _versions(*patterns: str) -> set[str]:
    rxs = [re.compile(p) for p in patterns]
    out: set[str] = set()
    for f in DEPSETS.glob("*.lock"):
        for rx in rxs:
            if m := rx.match(f.name):
                out.add(m.group(1))
    return out


def _expected_freezes(version: str) -> list[Path]:
    """Where refresh-image-freezes.sh writes a freeze for each tracked image."""
    out = []
    for line in (IMAGES / "tracked-images.txt").read_text().splitlines():
        if (line := line.strip()) and not line.startswith("#"):
            image = line.replace("{version}", version)
            out.append(IMAGES / f"{image.removeprefix('anyscale/').replace(':', '-')}.freeze.txt")
    return out


def complete_versions() -> set[str]:
    """Versions with a ray_<v>_img_* base lock and a freeze for EVERY tracked image.

    Requiring all of them is what makes the fanout fail closed: images publish over
    hours, and one present freeze is no evidence the rest landed. A template whose
    image is still missing would fail at its seed pre_hook mid-fanout.
    """
    return {
        v
        for v in _versions(r"ray_(\d+\.\d+\.\d+)_img_")
        if all(f.exists() for f in _expected_freezes(v))
    }


def _key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--require", metavar="VERSION",
        help="validate this version has a complete base-lock set (instead of deriving the newest)",
    )
    args = p.parse_args(argv)
    complete = complete_versions()

    if args.require:
        if args.require not in complete:
            print(
                f"error: Ray {args.require} is not complete — needs a "
                f"ray_{args.require}_img_* base lock and a freeze for every tracked "
                f"image. Missing: "
                + (
                    ", ".join(
                        f.name for f in _expected_freezes(args.require) if not f.exists()
                    )
                    or "(none — the base lock is what's absent)"
                ),
                file=sys.stderr,
            )
            return 1
        print(args.require)
        return 0

    if not complete:
        print("error: no complete base-lock set in dependencies/depsets/", file=sys.stderr)
        return 1
    print(max(complete, key=_key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
