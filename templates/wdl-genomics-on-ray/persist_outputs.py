"""Copy a completed run's declared outputs somewhere that outlives the cluster.

Anyscale terminates a job's cluster when the job *succeeds*, and `/mnt/cluster_storage` is
deleted with it. So a run pointed at that mount, which is the default because it is shared
across nodes and fast, destroys its own results by finishing correctly. A 14h44m
HG002 chr20 assembly completed, reported a 33.3 Mbp N50, and left
nothing behind but the driver log.

This copies only the *declared workflow outputs*, not the run tree. The distinction matters at
this scale: the same run's `40-polishing/bubbles_1.fasta` intermediate was 3.85 GB on its own,
and none of it is a result.

Destination, in order:

  1. ``--dest``, or ``WDL_ON_RAY_RESULTS``. An ``s3://`` URI or a path.
  2. the first writable durable mount: ``/mnt/user_storage``, then ``/mnt/shared_storage``.
     Both persist across clusters where the cloud provides them, which is exactly the property
     ``/mnt/cluster_storage`` lacks.
  3. nothing, with a loud warning. Not an error: the local and demo paths have no cluster to
     outlive, and failing a 14h run at the last step because a bucket was unset would be worse
     than the problem being solved.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

#: Mounts that survive cluster termination; probed, since not every cloud has them.
DURABLE_MOUNTS = ("/mnt/user_storage", "/mnt/shared_storage")

Json = str | int | float | bool | None | list["Json"] | dict[str, "Json"]


def durable_mount() -> str | None:
    for mount in DURABLE_MOUNTS:
        path = Path(mount)
        if path.is_dir() and os.access(path, os.W_OK):
            return mount
    return None


def local_files(value: Json) -> list[str]:
    """Every existing local file path reachable in ``value``."""
    if isinstance(value, str):
        return [value] if value.startswith("/") and Path(value).is_file() else []
    if isinstance(value, dict):
        return [found for item in value.values() for found in local_files(item)]
    if isinstance(value, list):
        return [found for item in value for found in local_files(item)]
    return []


def _destination_names(sources: list[str]) -> list[str]:
    # Scatter shards share basenames: colliding names get an index directory, unique ones stay flat.
    totals = Counter(Path(source).name for source in sources)
    seen: dict[str, int] = {}
    names = []
    for source in sources:
        base = Path(source).name
        if totals[base] == 1:
            names.append(base)
        else:
            index = seen.get(base, 0)
            seen[base] = index + 1
            names.append(f"{index}/{base}")
    return names


def copy_out(sources: list[str], dest: str, prefix: str) -> list[str]:
    """Copy ``sources`` under ``dest/prefix/``, returning the new locations, in order."""
    written: list[str] = []
    names = _destination_names(sources)
    if dest.startswith("s3://"):
        target = f"{dest.rstrip('/')}/{prefix}"
        for source, name in zip(sources, names):
            # One cp per file: the sources are scattered over per-task directories, not a tree.
            destination = f"{target}/{name}"
            subprocess.run(["aws", "s3", "cp", source, destination], check=True)
            written.append(destination)
        return written

    target_dir = Path(dest) / prefix
    for source, name in zip(sources, names):
        local_destination = target_dir / name
        local_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, local_destination)
        written.append(str(local_destination))
    return written


def rewrite(value: Json, moved: dict[str, str]) -> Json:
    """``value`` with every copied path replaced and its shape (maps, nested arrays) preserved."""
    if isinstance(value, str):
        return moved.get(value, value)
    if isinstance(value, dict):
        return {key: rewrite(item, moved) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite(item, moved) for item in value]
    return value


def persist(outputs_json: Path, dest: str) -> dict[str, Json]:
    """Copy every File output in ``outputs_json`` to ``dest``, rewriting the report's paths."""
    report = json.loads(outputs_json.read_text())
    # miniwdl's run-root outputs.json is the bare name -> value map; the envelope is stdout-only.
    outputs: dict[str, Json] = report.get("outputs", report)

    persisted: dict[str, Json] = {}
    total = 0
    for name in sorted(outputs):
        sources = local_files(outputs[name])
        if not sources:
            persisted[name] = outputs[name]
            continue
        prefix = name.split(".")[-1]
        copied = copy_out(sources, dest, prefix)
        persisted[name] = rewrite(outputs[name], dict(zip(sources, copied)))
        total += len(copied)
        print(f"  {name} -> {len(copied)} file(s)")

    result: dict[str, Json] = {
        "source_dir": report.get("dir"),
        "persisted_to": dest,
        "outputs": persisted,
    }
    summary = json.dumps(result, indent=2, sort_keys=True)

    if dest.startswith("s3://"):
        local_copy = outputs_json.with_name("outputs.persisted.json")
        local_copy.write_text(summary)
        subprocess.run(
            ["aws", "s3", "cp", str(local_copy), f"{dest.rstrip('/')}/outputs.json"], check=True
        )
    else:
        # Also when nothing was copied: the report is still written.
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "outputs.json").write_text(summary)

    print(f"persisted {total} file(s) to {dest}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True, type=Path, help="the run's outputs JSON")
    parser.add_argument("--dest", help="s3:// URI or path; else WDL_ON_RAY_RESULTS, else a mount")
    args = parser.parse_args(argv)

    if not args.outputs.is_file():
        # The program's own exit status already reports the failure; copying nothing is correct.
        print(f"no outputs JSON at {args.outputs}; nothing to persist", file=sys.stderr)
        return 0

    dest = args.dest or os.environ.get("WDL_ON_RAY_RESULTS") or durable_mount()
    if not dest:
        print(
            "WARNING: no durable destination for results.\n"
            "  This run's outputs are under a mount that is deleted when the cluster\n"
            "  terminates, which for a job happens on success. Set WDL_ON_RAY_RESULTS to an\n"
            f"  s3:// URI or a path on one of {', '.join(DURABLE_MOUNTS)}.",
            file=sys.stderr,
        )
        return 0

    persist(args.outputs, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
