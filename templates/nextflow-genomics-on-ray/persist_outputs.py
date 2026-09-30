"""Copy a finished run's published results somewhere that outlives the cluster."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DURABLE_MOUNTS = ("/mnt/user_storage", "/mnt/shared_storage")

SUBDIR = "nextflow-genomics-on-ray/results"


def durable_mount() -> str | None:
    for mount in DURABLE_MOUNTS:
        path = Path(mount)
        if path.is_dir() and os.access(path, os.W_OK):
            return str(path / SUBDIR)
    return None


def persist(results: Path, dest: str) -> int:
    """Copy ``results`` to ``dest``; return the number of files copied."""
    files = [p for p in results.rglob("*") if p.is_file()]
    if dest.startswith("s3://"):
        # sync, so a re-run after a partial copy moves only what is missing.
        subprocess.run(
            ["aws", "s3", "sync", "--only-show-errors", str(results), dest.rstrip("/")],
            check=True,
        )
    else:
        shutil.copytree(results, dest, dirs_exist_ok=True)
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", required=True, type=Path, help="the run's --outdir")
    parser.add_argument("--dest", help="s3:// URI or path; else NF_RAY_RESULTS, else a mount")
    args = parser.parse_args(argv)

    if not args.results.is_dir():
        # The pipeline's own exit status already reports the failure.
        print(f"no results directory at {args.results}; nothing to persist", file=sys.stderr)
        return 0

    dest = args.dest or os.environ.get("NF_RAY_RESULTS") or durable_mount()
    if not dest:
        print(
            "WARNING: no durable destination for results.\n"
            "  This run's outputs are under a mount that is deleted when the cluster\n"
            "  terminates, which for a job happens on success. Set NF_RAY_RESULTS to an\n"
            f"  s3:// URI or a path on one of {', '.join(DURABLE_MOUNTS)}.",
            file=sys.stderr,
        )
        return 0

    count = persist(args.results, dest)
    print(f"persisted {count} file(s) from {args.results} to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
