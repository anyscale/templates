"""Copy a finished run's published results somewhere that outlives the cluster.

Anyscale terminates a job's cluster when the job ends, and `/mnt/cluster_storage` goes with
it. So a run that publishes there, the default because it is shared across nodes and fast,
loses its results even when it succeeds.

Nextflow makes the copy small. Every process that produces a result declares a
`publishDir`, so `--outdir` already holds the declared outputs and nothing else -- the
work directory, with every intermediate BAM, stays behind. This copies that tree, including
`pipeline_info/` (the trace, and the executor's placement record the Gantt chart reads).

Destination, in order:

  1. ``--dest``, or ``NF_RAY_RESULTS``. An ``s3://`` URI or a path.
  2. the first writable durable mount: ``/mnt/user_storage``, then ``/mnt/shared_storage``.
     Both persist across clusters where the cloud provides them, which is exactly the
     property ``/mnt/cluster_storage`` lacks.
  3. nothing, with a loud warning. Not an error: a workspace has no cluster to outlive, and
     failing a finished run at the last step because a bucket was unset would be worse than
     the problem being solved.

    python persist_outputs.py --results /mnt/cluster_storage/nf-genomics/results
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Mounts that survive cluster termination. Not present on every Anyscale cloud, hence
#: probed rather than assumed.
DURABLE_MOUNTS = ("/mnt/user_storage", "/mnt/shared_storage")

#: Under the chosen mount, so two templates' results do not land on top of each other.
SUBDIR = "nextflow-genomics-on-ray/results"


def durable_mount() -> str | None:
    """The first durable mount this cluster actually has and can write to."""
    for mount in DURABLE_MOUNTS:
        path = Path(mount)
        if path.is_dir() and os.access(path, os.W_OK):
            return str(path / SUBDIR)
    return None


def persist(results: Path, dest: str) -> int:
    """Copy ``results`` to ``dest``; return the number of files copied."""
    files = [p for p in results.rglob("*") if p.is_file()]
    if dest.startswith("s3://"):
        # `sync` rather than `cp --recursive`: re-running after a partial copy moves only
        # what is missing. Signed, unlike the demo-data download: this is a write, and it
        # goes to a bucket the node's role is meant to reach.
        subprocess.run(
            ["aws", "s3", "sync", "--only-show-errors", str(results), dest.rstrip("/")],
            check=True,
        )
    else:
        # dirs_exist_ok, so a second persist of the same run refreshes it in place.
        shutil.copytree(results, dest, dirs_exist_ok=True)
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", required=True, type=Path, help="the run's --outdir")
    parser.add_argument("--dest", help="s3:// URI or path; else NF_RAY_RESULTS, else a mount")
    args = parser.parse_args(argv)

    if not args.results.is_dir():
        # The pipeline failed before publishing anything, which its own exit status already
        # says. Copying nothing is correct; masking that with an error here is not.
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
