"""Which Ray versions we hold a complete dependency set for.

Both the daily probe (`prepare-base-locks.py`) and the fanout's resolver
(`latest-depset-version.py`) gate on this. They used to each carry their own copy
and drifted: the probe accepted a version on one freeze while the resolver required
all of them, so after a partial image publish the probe called the version done and
stopped refreshing the freezes the resolver was still waiting for. One definition,
imported by both.
"""

from __future__ import annotations

import re
from pathlib import Path


def _repo_root() -> Path:
    """Nearest ancestor dir containing BUILD.yaml (robust to where this script lives)."""
    for p in Path(__file__).resolve().parents:
        if (p / "BUILD.yaml").is_file():
            return p
    raise RuntimeError("repo root not found: no BUILD.yaml above this script")


DEPSETS = _repo_root() / "dependencies" / "depsets"
IMAGES = _repo_root() / "dependencies" / "images"

# A freeze this small is a failed fetch, not an image — see fetch-image-freeze.sh.
MIN_FREEZE_PACKAGES = 50


def lock_versions(*patterns: str) -> set[str]:
    rxs = [re.compile(p) for p in patterns]
    out: set[str] = set()
    for f in DEPSETS.glob("*.lock"):
        for rx in rxs:
            if m := rx.match(f.name):
                out.add(m.group(1))
    return out


def expected_freezes(version: str) -> list[Path]:
    """Where refresh-image-freezes.sh writes a freeze for each tracked image."""
    out = []
    for line in (IMAGES / "tracked-images.txt").read_text().splitlines():
        if (line := line.strip()) and not line.startswith("#"):
            image = line.replace("{version}", version)
            out.append(IMAGES / f"{image.removeprefix('anyscale/').replace(':', '-')}.freeze.txt")
    return out


def usable(freeze: Path) -> bool:
    """A freeze that exists but holds no packages seeds nothing, so treat it as absent."""
    if not freeze.is_file():
        return False
    return sum(1 for line in freeze.read_text().splitlines() if "==" in line) >= MIN_FREEZE_PACKAGES


def missing_freezes(version: str) -> list[Path]:
    return [f for f in expected_freezes(version) if not usable(f)]


def complete_versions() -> set[str]:
    """Versions with a ray_<v>_img_* base lock and a usable freeze for EVERY tracked image.

    Requiring all of them is what makes the fanout fail closed: images publish over
    hours, and one present freeze is no evidence the rest landed. A template whose
    image is still missing would fail at its seed pre_hook mid-fanout.
    """
    return {
        v
        for v in lock_versions(r"ray_(\d+\.\d+\.\d+)_img_")
        if not missing_freezes(v)
    }


def version_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def newest_complete() -> str | None:
    c = complete_versions()
    return max(c, key=version_key) if c else None
