#!/usr/bin/env python3
"""Resolve the target Ray version for a fanout from the committed image freezes.

Completeness is defined in depset_versions.py, shared with prepare-ray-version.py.
With no args this prints the newest complete version; with --require <v> it
validates that <v> is complete (and echoes it). Exits non-zero with a message on
stderr when there's nothing to resolve — so the caller can fail closed.
"""

from __future__ import annotations

import argparse
import sys

from depset_versions import complete_versions, missing_freezes, version_key as _key


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--require", metavar="VERSION",
        help="validate this version has a freeze for every tracked image (instead of deriving the newest)",
    )
    args = p.parse_args(argv)
    complete = complete_versions()

    if args.require:
        if args.require not in complete:
            print(
                f"error: Ray {args.require} is not complete — needs a freeze for "
                f"every tracked image. Missing: "
                + ", ".join(f.name for f in missing_freezes(args.require)),
                file=sys.stderr,
            )
            return 1
        print(args.require)
        return 0

    if not complete:
        print("error: no Ray version has a freeze for every tracked image", file=sys.stderr)
        return 1
    print(max(complete, key=_key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
