#!/usr/bin/env python3
"""Canonical effective hash per template, used by drift-scan and by product's
build.sh (`python ci/compute_effective_hash.py <tmpl-name>`)."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validate_build_yaml import REPO_ROOT, Entry  # noqa: E402


def load_entries() -> list[Entry]:
    raw = yaml.safe_load((REPO_ROOT / "BUILD.yaml").read_text())
    return TypeAdapter(list[Entry]).validate_python(raw)


def find_entry(entries: list[Entry], name: str) -> Entry:
    for e in entries:
        if e.name == name:
            return e
    raise KeyError(name)


def _canonical_yaml(entry: Entry) -> bytes:
    return yaml.safe_dump(
        entry.model_dump(),
        sort_keys=True,
        default_flow_style=False,
    ).encode()


def _git_object_hash(path: str, *, ref: str = "HEAD") -> str:
    # Works for both directories (tree hash) and files (blob hash).
    out = subprocess.check_output(
        ["git", "rev-parse", f"{ref}:{path}"],
        cwd=REPO_ROOT,
    )
    return out.strip().decode()


def effective_hash(entry: Entry, *, ref: str = "HEAD") -> str:
    h = hashlib.sha256()
    h.update(_canonical_yaml(entry))
    h.update(b"\n")
    h.update(_git_object_hash(entry.dir, ref=ref).encode())
    h.update(b"\n")
    h.update(_git_object_hash(entry.compute_config.GCP, ref=ref).encode())
    h.update(b"\n")
    h.update(_git_object_hash(entry.compute_config.AWS, ref=ref).encode())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Template name (BUILD.yaml entry name)")
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Git ref to hash against (default: HEAD)",
    )
    args = parser.parse_args()

    entries = load_entries()
    try:
        entry = find_entry(entries, args.name)
    except KeyError:
        print(f"error: no BUILD.yaml entry named {args.name!r}", file=sys.stderr)
        return 1

    print(effective_hash(entry, ref=args.ref))
    return 0


if __name__ == "__main__":
    sys.exit(main())
