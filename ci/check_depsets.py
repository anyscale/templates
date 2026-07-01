#!/usr/bin/env python3
"""Scoped, resilient check-depsets (see .claude/skills/template/references/dependencies.md).

`./update_deps.sh --check` re-resolves *every* lock against live PyPI/PyTorch, so
running it on every PR makes unrelated PRs fail on another template's ambient
upstream drift or a transient index 503. This wrapper runs the check only when a
PR actually changes something that feeds a lock, and retries transient failures.

Coarse by design: when it does run, it runs the FULL check (per-entry scoping
would need `raydepsets --check --name`, whose behaviour isn't verified here — a
noted follow-up). Fail-safe: any uncertainty -> full check; it never skips when a
lock input might have changed. The scheduled refresh workflow handles ambient
drift out-of-band so these full checks stay green.

Usage: check_depsets.py <base-sha> <head-sha>   (empty base-sha -> full check)
"""
import re
import subprocess
import sys
import time

CONFIG = "dependencies/template.depsets.yaml"


def full_check(attempts: int = 3) -> int:
    """./update_deps.sh --check, retried to absorb transient upstream index errors."""
    for i in range(1, attempts + 1):
        if subprocess.run(["./update_deps.sh", "--check"]).returncode == 0:
            return 0
        if i < attempts:
            print(f"::warning::check-depsets attempt {i} failed; retrying in {i * 20}s", file=sys.stderr)
            time.sleep(i * 20)
    print("::error::check-depsets failed after retries", file=sys.stderr)
    return 1


def lock_bearing_dirs(config_text: str) -> set:
    """templates/<name> dirs that have an active (non-commented) depset entry."""
    return set(re.findall(r"(?m)^\s*output:\s*(templates/[^/\s]+)/python_depset\.lock\s*$", config_text))


def decide(files: list, lock_dirs: set) -> str:
    """Return 'full' (run the whole check) or 'skip' (nothing lock-relevant changed)."""
    def shared(f: str) -> bool:
        # Inputs that feed *every* lock -> a change means check everything.
        return (
            f == CONFIG
            or f == "update_deps.sh"
            or f.startswith("dependencies/scripts/")
            or f.startswith("dependencies/depsets/")
        )

    if any(shared(f) for f in files):
        return "full"

    touched = {
        "/".join(f.split("/")[:2])
        for f in files
        if f.startswith("templates/") and (f.endswith("/requirements.txt") or f.endswith("/python_depset.lock"))
    }
    return "full" if (touched & lock_dirs) else "skip"


def changed_files(base: str, head: str):
    p = subprocess.run(["git", "diff", "--name-only", base, head], capture_output=True, text=True)
    return None if p.returncode != 0 else [f for f in p.stdout.splitlines() if f.strip()]


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else ""
    head = sys.argv[2] if len(sys.argv) > 2 else "HEAD"

    if not base:
        print("No base ref (push / non-PR) -> full check", file=sys.stderr)
        return full_check()

    files = changed_files(base, head)
    if files is None:
        print("Could not diff against base -> full check", file=sys.stderr)
        return full_check()

    if decide(files, lock_bearing_dirs(open(CONFIG).read())) == "skip":
        print("No lock-bearing depset input changed -> skipping check-depsets", file=sys.stderr)
        return 0
    print("Depset input changed -> full check", file=sys.stderr)
    return full_check()


if __name__ == "__main__":
    sys.exit(main())
