#!/usr/bin/env python3
"""Prepare a new Ray version so a fanout can fire.

The fanout only fires a version we hold a freeze for on every tracked image. This
resolves the newest stable Ray and, if we don't have it, adds its `build_arg_sets` to
`dependencies/template.depsets.yaml` and leaves the change staged for a PR. The caller
then refreshes the freezes (`scripts/depsets/refresh-image-freezes.py`).

Minor-only: on the auto path it acts only when the newest stable Ray advances the minor
or major, since templates track minor releases — 2.56.0 → 2.56.1 is a no-op, and a new
minor adopts its newest patch. `--version` / `--force` bypass the gate.

Copy-forward: it clones the newest-complete version's bundle matrix, substituting the
new version, and checks Anyscale published every tracked image for it. It never invents
a matrix — if a tracked image has no `<v>` tag (a py/cuda moved, as at 2.55→2.56) it
stops and a human runs `upgrade-dependencies.md`.

Exit codes: 0 = nothing to do (already current / waiting on upstream / dry-run);
            10 = changes prepared (caller should open a PR);
            2  = needs human (matrix changed or bad input).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from depset_versions import complete_versions, newest_complete, tracked_images

def _repo_root() -> Path:
    """Nearest ancestor dir containing BUILD.yaml (robust to where this script lives)."""
    for p in Path(__file__).resolve().parents:
        if (p / "BUILD.yaml").is_file():
            return p
    raise RuntimeError("repo root not found: no BUILD.yaml above this script")


REPO_ROOT = _repo_root()
CONFIG = REPO_ROOT / "dependencies" / "template.depsets.yaml"

IMAGE_INDEX = "https://docs.anyscale.com/base-images/index.json"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def set_output(**kv: str) -> None:
    """Expose results to the GitHub Actions step (no-op locally)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        for k, v in kv.items():
            f.write(f"{k}={v}\n")


def compact(v: str) -> str:
    return v.replace(".", "")


def _ver(s: str) -> tuple[int, ...]:
    """Version string → int tuple for ordering (e.g. '2.56.1' → (2, 56, 1))."""
    return tuple(int(x) for x in s.split("."))


# ── upstream / repo state ──────────────────────────────────────────────────


def newest_stable_ray() -> str:
    """Newest non-yanked X.Y.Z release of `ray` on PyPI."""
    with urllib.request.urlopen("https://pypi.org/pypi/ray/json", timeout=30) as r:
        data = json.load(r)
    stable = []
    for v, files in data["releases"].items():
        if re.fullmatch(r"\d+\.\d+\.\d+", v) and files and not all(f.get("yanked") for f in files):
            stable.append(v)
    if not stable:
        raise RuntimeError("no stable ray release found on PyPI")
    return max(stable, key=_ver)


def is_minor_upgrade(target: str, current: str | None) -> bool:
    """Whether `target` advances (major, minor) beyond `current` — the only case the
    scheduled path auto-prepares. Patch-only bumps (same major.minor) and versions
    behind `current` return False; a missing `current` (bootstrap) returns True."""
    if current is None:
        return True
    return _ver(target)[:2] > _ver(current)[:2]


# ── upstream image discovery (did Anyscale publish what we track?) ──────────


def unpublished_images(v: str) -> list[str]:
    """Tracked images with no published `<v>` tag.

    The index lists CPU images under an explicit -cpu suffix while BUILD.yaml uses the
    bare alias, so accept either — same fallback as refresh-image-freezes.py.
    """
    # the CDN 403s urllib's default User-Agent
    req = urllib.request.Request(IMAGE_INDEX, headers={"User-Agent": "prepare-ray-version"})
    with urllib.request.urlopen(req, timeout=30) as r:
        names = {e.get("imageName") for e in json.load(r)}
    missing = []
    for image in (img.replace("{version}", v) for img in tracked_images()):
        if image not in names and f"{image}-cpu" not in names:
            missing.append(image)
    return missing


# ── config edit (ruamel round-trip: preserve comments + layout) ─────────────


def _yaml():
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    y.width = 4096  # don't wrap long sequences
    y.indent(mapping=2, sequence=4, offset=2)  # match the file's block style (no reformat churn)
    return y


def bundles_for(cfg, version) -> list[dict]:
    """The build_arg_set dicts declared for `version`, in order."""
    prefix = f"ray{compact(version)}_"
    return [dict(b) for name, b in cfg["build_arg_sets"].items() if name.startswith(prefix)]


def bundle_name(v: str, b: dict) -> str:
    # CUDA_VARIANT is only carried where something interpolates it (the ray-llm
    # freeze names); elsewhere the bundle is just (ray version, python).
    suffix = f"_{b['CUDA_VARIANT']}" if "CUDA_VARIANT" in b else ""
    return f"ray{compact(v)}_py{b['PYTHON_SHORT']}{suffix}"


def apply_edit(cfg, target: str, prev_bundles: list[dict]) -> list[str]:
    """Declare the target's build_arg_sets, mirroring the previous version's matrix.

    Nothing references them yet — each template picks its bundle up when the fanout
    repoints it (workflows/bump-ray-version.md).
    """
    from ruamel.yaml.comments import CommentedMap
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq

    bas = cfg["build_arg_sets"]
    added: list[str] = []
    for pb in prev_bundles:
        name = bundle_name(target, pb)
        if name in bas:
            continue
        m = CommentedMap()
        m["RAY_VERSION"] = dq(target)
        m["PYTHON_VERSION"] = dq(pb["PYTHON_VERSION"])
        m["PYTHON_SHORT"] = dq(pb["PYTHON_SHORT"])
        if "CUDA_VARIANT" in pb:  # only where an entry interpolates it
            m["CUDA_VARIANT"] = dq(pb["CUDA_VARIANT"])
        bas[name] = m
        added.append(name)
    return added


# ── orchestration ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", help="target Ray version (default: newest stable on PyPI)")
    p.add_argument("--force", action="store_true", help="prepare even if the version already looks complete")
    p.add_argument("--dry-run", action="store_true", help="plan + edit in memory, print the diff, don't recompile or write")
    args = p.parse_args(argv)

    target = args.version or newest_stable_ray()
    if not re.fullmatch(r"\d+\.\d+\.\d+", target):
        log(f"error: bad version {target!r}")
        return 2
    log(f"Target Ray version: {target}")

    # Minor-only gate (scheduled/auto path): templates track minor Ray releases, so a
    # patch over the current minor (e.g. 2.56.0 → 2.56.1) shouldn't open a base-locks PR.
    # An explicit --version (or --force) bypasses this — a human targeting a specific release.
    if not args.version and not args.force:
        current = newest_complete()
        if not is_minor_upgrade(target, current):
            log(f"Ray {target} is not a new minor over {current} "
                "(templates track minor Ray releases; --version overrides) — nothing to do.")
            set_output(status="skipped-patch", version=target)
            return 0

    if target in complete_versions() and not args.force:
        log("Already have a freeze for every tracked image — nothing to do.")
        set_output(status="current", version=target)
        return 0

    prev = newest_complete()
    if prev is None:
        log("error: no existing complete version to copy the matrix forward from — needs human.")
        set_output(status="needs-human", version=target)
        return 2
    log(f"Copying the bundle matrix forward from {prev}.")

    cfg = _yaml().load(CONFIG.read_text())
    prev_bundles = bundles_for(cfg, prev)
    if not prev_bundles:
        log(f"error: couldn't read {prev}'s bundle matrix from the config — needs human.")
        set_output(status="needs-human", version=target)
        return 2

    # An image we track but that <target> never shipped means the matrix moved; the
    # freezes could not be fetched anyway, so stop rather than prepare half a version.
    if missing := unpublished_images(target):
        if len(missing) == len(tracked_images()):
            log(f"No {target} images published yet. Waiting.")
            set_output(status="waiting", version=target)
            return 0
        log(f"Image matrix changed for {target}: no published tag for {', '.join(missing)} — needs human.")
        set_output(status="needs-human", version=target)
        return 2

    added = apply_edit(cfg, target, prev_bundles)
    log(f"build_arg_sets added: {', '.join(added) or '(none — already declared)'}")

    if args.dry_run:
        buf = io.StringIO()
        _yaml().dump(cfg, buf)
        (CONFIG.parent / "template.depsets.yaml.planned").write_text(buf.getvalue())
        subprocess.run(["git", "--no-pager", "diff", "--no-index", "--", str(CONFIG),
                        str(CONFIG.parent / "template.depsets.yaml.planned")], cwd=REPO_ROOT)
        (CONFIG.parent / "template.depsets.yaml.planned").unlink()
        set_output(status="dry-run", version=target)
        return 0

    with open(CONFIG, "w") as f:
        _yaml().dump(cfg, f)

    log(f"Prepared Ray {target}. The caller refreshes the image freezes.")
    set_output(status="prepared", version=target)
    return 10


if __name__ == "__main__":
    sys.exit(main())
