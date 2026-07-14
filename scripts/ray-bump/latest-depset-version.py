#!/usr/bin/env python3
"""Resolve the target Ray version for a fanout from dependencies/depsets/.

The base locks there register the blessed image variants per Ray version:
  ray_<v>_img_py<PY>.lock         (base image)
  rayllm_<v>_py<PY>_cu<CU>.lock   (LLM image)
A version is "complete" once both families are present. With no args this prints
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


def _versions(*patterns: str) -> set[str]:
    rxs = [re.compile(p) for p in patterns]
    out: set[str] = set()
    for f in DEPSETS.glob("*.lock"):
        for rx in rxs:
            if m := rx.match(f.name):
                out.add(m.group(1))
    return out


def complete_versions() -> set[str]:
    """Versions present as BOTH a ray_<v>_img_* and a rayllm_<v>_* base lock
    (the old ray_<v>_llm_* naming is also accepted for the LLM family)."""
    img = _versions(r"ray_(\d+\.\d+\.\d+)_img_")
    llm = _versions(r"rayllm_(\d+\.\d+\.\d+)_", r"ray_(\d+\.\d+\.\d+)_llm_")
    return img & llm


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
                f"error: no complete base-lock set (ray_{args.require}_img_* AND "
                f"rayllm_{args.require}_*) in dependencies/depsets/",
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
