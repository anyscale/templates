"""Which Ray versions we hold a complete dependency set for.

A version is complete once `dependencies/images/` holds a usable freeze for every
image in `tracked-images.txt`. That is the whole condition: a freeze exists only once
Anyscale published the image, which is what a per-template bump actually waits on.

Importable by `prepare-ray-version.py` and `scripts/depsets/refresh-image-freezes.py`
— they used to carry copies of this and drifted.

Also the CLI both workflows resolve their target version with. No args prints the
newest complete version; `--require <v>` validates that one. Either way it exits
non-zero with a message on stderr when there is nothing to resolve, so a caller that
propagates the exit code fails closed:

    python3 scripts/ray-bump/depset_versions.py
    python3 scripts/ray-bump/depset_versions.py --require 2.57.0
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


IMAGES = _repo_root() / "dependencies" / "images"

# A freeze this small is a failed fetch, not an image — see refresh-image-freezes.py.
MIN_FREEZE_PACKAGES = 50


def tracked_images() -> list[str]:
    """The image names in tracked-images.txt, `{version}` left unexpanded."""
    return [
        line
        for raw in (IMAGES / "tracked-images.txt").read_text().splitlines()
        if (line := raw.strip()) and not line.startswith("#")
    ]


def freeze_path(image: str) -> Path:
    """Where refresh-image-freezes.py writes this image's freeze."""
    return IMAGES / f"{image.removeprefix('anyscale/').replace(':', '-')}.freeze.txt"


def expected_freezes(version: str) -> list[Path]:
    return [freeze_path(img.replace("{version}", version)) for img in tracked_images()]


def usable(freeze: Path) -> bool:
    """A freeze that exists but holds no packages seeds nothing, so treat it as absent."""
    if not freeze.is_file():
        return False
    return sum(1 for line in freeze.read_text().splitlines() if "==" in line) >= MIN_FREEZE_PACKAGES


def missing_freezes(version: str) -> list[Path]:
    return [f for f in expected_freezes(version) if not usable(f)]


def known_versions() -> set[str]:
    """Every Ray version any committed freeze mentions."""
    rx = re.compile(r"-(\d+\.\d+\.\d+)-")
    return {m.group(1) for f in IMAGES.glob("*.freeze.txt") if (m := rx.search(f.name))}


def complete_versions() -> set[str]:
    """Versions with a usable freeze for EVERY tracked image.

    Requiring all of them is what makes the fanout fail closed: images publish over
    hours, and one present freeze is no evidence the rest landed. A template whose
    image is still missing would fail at its seed pre_hook mid-fanout.
    """
    return {v for v in known_versions() if not missing_freezes(v)}


def version_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def newest_complete() -> str | None:
    c = complete_versions()
    return max(c, key=version_key) if c else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--require", metavar="VERSION",
        help="validate this version has a freeze for every tracked image "
             "(instead of deriving the newest)",
    )
    args = p.parse_args(argv)

    if args.require:
        if missing := missing_freezes(args.require):
            print(
                f"error: Ray {args.require} is not complete — needs a freeze for every "
                f"tracked image. Missing: " + ", ".join(f.name for f in missing),
                file=sys.stderr,
            )
            return 1
        print(args.require)
        return 0

    if not (newest := newest_complete()):
        print("error: no Ray version has a freeze for every tracked image", file=sys.stderr)
        return 1
    print(newest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
