#!/usr/bin/env python3
"""Scoped, resilient check-depsets (see .claude/skills/template/references/dependencies.md).

`./update_deps.sh --check` re-resolves *every* lock against live PyPI/PyTorch, so on
every PR it fails unrelated PRs on another template's ambient drift or a transient
index 503. This picks the narrowest still-correct check for a PR's changed files:
  * skip   — no lock input changed.
  * scoped — regenerate + diff only the affected templates' locks. A template is
             affected if its requirements/lock changed, or if only its own entry in
             the depset config changed (the base entries + build_arg_sets map identical).
  * full   — a global input changed (update_deps.sh, dependencies/{scripts,depsets}/*,
             or a shared part of the config), or any uncertainty. Never skip when unsure.

Usage: check-depsets.py <base-sha> <head-sha>   (empty base-sha -> full check)
"""
import re
import subprocess
import sys
import time

import yaml

CONFIG = "dependencies/template.depsets.yaml"
UPDATE_DEPS = "./update_deps.sh"

# Global inputs we can't attribute to specific templates -> a change means check all.
# (CONFIG isn't here: its diff is analyzed, so a single-template edit can still scope.)
GLOBAL_FILES = {"update_deps.sh"}
GLOBAL_PREFIXES = ("dependencies/scripts/", "dependencies/depsets/")

# A depset entry's `output:` when it's a template lock; group(1) = the templates/<dir>.
TEMPLATE_LOCK_RE = re.compile(r"(templates/[^/]+)/python_depset\.lock$")


def run_retry(cmd: list, attempts: int = 3) -> bool:
    """True on a clean exit, retrying transient index errors (503s). NB: a lock *drift*
    isn't a failure here — update_deps.sh exits 0 regardless; the caller's diff catches it."""
    for i in range(1, attempts + 1):
        if subprocess.run(cmd).returncode == 0:
            return True
        if i < attempts:
            print(f"::warning::`{' '.join(cmd)}` attempt {i} failed; retrying in {i * 20}s", file=sys.stderr)
            time.sleep(i * 20)  # linear backoff: 20s, 40s
    return False


def full_check() -> int:
    """All-sets `./update_deps.sh --check` — for global changes and fail-safe."""
    if run_retry([UPDATE_DEPS, "--check"]):
        return 0
    print("::error::check-depsets (full) failed after retries", file=sys.stderr)
    return 1


def _template_entries(cfg: dict) -> dict:
    """templates/<dir> -> its depset entry, for entries whose output is a template lock."""
    out = {}
    for e in cfg.get("depsets") or []:
        m = TEMPLATE_LOCK_RE.match(e.get("output") or "")
        if m:
            out[m.group(1)] = e
    return out


def template_instances(cfg: dict) -> dict:
    """templates/<dir> -> [raydepsets instance name(s)] for each template lock entry."""
    bas = cfg.get("build_arg_sets") or {}  # token -> {RAY_VERSION, PYTHON_VERSION, CUDA_VARIANT, ...}
    out = {}
    for d, entry in _template_entries(cfg).items():
        instances = []
        for ba in entry.get("build_arg_sets") or []:
            # interpolate the entry's ${VAR} name template with this build-arg-set's values
            name = entry["name"]
            for var, val in (bas.get(ba) or {}).items():
                name = name.replace(f"${{{var}}}", str(val))
            instances.append(name)
        out[d] = instances
    return out


def config_scope(base_text: str, head_cfg: dict):
    """Template dirs whose config entry changed — or None if any *shared* part of the
    config (build_arg_sets map, base entries, structure) changed -> caller full-checks."""
    base_cfg = yaml.safe_load(base_text)

    def shared(cfg):  # the config minus its template entries; a change here can affect any lock
        rest = [e for e in (cfg.get("depsets") or []) if not TEMPLATE_LOCK_RE.match(e.get("output") or "")]
        return {**cfg, "depsets": rest}

    if shared(base_cfg) != shared(head_cfg):
        return None
    base_t, head_t = _template_entries(base_cfg), _template_entries(head_cfg)
    return {d for d in head_t if base_t.get(d) != head_t[d]}  # added or modified template entries


def changed_files(base: str, head: str):
    """Paths changed between base..head, or None if git can't diff (-> caller full-checks)."""
    p = subprocess.run(["git", "diff", "--name-only", base, head], capture_output=True, text=True)
    return None if p.returncode != 0 else [f for f in p.stdout.splitlines() if f.strip()]


def git_show(ref: str, path: str):
    """Contents of `path` at `ref`, or None if it can't be read."""
    p = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def classify(files: list, head_cfg: dict, base_config_text) -> tuple:
    """(action, affected, reason): action is 'full' | 'skip' | 'scoped'. base_config_text
    is CONFIG at the base commit (None if CONFIG is unchanged, or unreadable)."""
    if any(f in GLOBAL_FILES or f.startswith(GLOBAL_PREFIXES) for f in files):
        return "full", [], "a global input changed"

    lock_dirs = set(template_instances(head_cfg))
    # templates whose own requirements/lock changed
    affected = {
        "/".join(f.split("/")[:2])
        for f in files
        if f.startswith("templates/") and (f.endswith("/requirements.txt") or f.endswith("/python_depset.lock"))
    } & lock_dirs

    if CONFIG in files:  # analyze the config diff rather than always full-checking
        if base_config_text is None:
            return "full", [], "config changed but its base version is unavailable"
        changed = config_scope(base_config_text, head_cfg)
        if changed is None:
            return "full", [], "a shared part of the depset config changed"
        affected |= changed

    return ("scoped", sorted(affected), "") if affected else ("skip", [], "")


def scoped_check(affected: list, instances_by_dir: dict) -> int:
    """Regenerate only the affected templates' locks, then diff them."""
    instances = [inst for d in affected for inst in instances_by_dir[d]]
    if not instances:  # lock-bearing but unscopable -> fail-safe to the full check
        print("Affected template has no build instance -> full check (fail-safe)", file=sys.stderr)
        return full_check()

    print(f"Lock-bearing templates changed -> scoped check: {', '.join(affected)}", file=sys.stderr)
    # Each entry compiles independently, so regenerating one template is isolated from
    # other templates' and base-lock drift.
    for inst in instances:
        if not run_retry([UPDATE_DEPS, "--name", inst]):
            print(f"::error::failed to regenerate depset '{inst}' after retries", file=sys.stderr)
            return 1

    # A regenerated lock that differs from the committed one means the commit is stale.
    locks = [f"{d}/python_depset.lock" for d in affected]
    if subprocess.run(["git", "diff", "--exit-code", "--"] + locks).returncode != 0:
        print("::error::depset lock(s) out of date — run `./update_deps.sh` and commit the result:", file=sys.stderr)
        for lock in locks:
            print(f"::error::  {lock}", file=sys.stderr)
        return 1
    print("Affected depset locks are up to date.", file=sys.stderr)
    return 0


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else ""  # PR base sha; empty on push / non-PR
    head = sys.argv[2] if len(sys.argv) > 2 else "HEAD"

    # Fail-safe: anything that stops us scoping confidently -> full check.
    if not base:
        print("No base ref (push / non-PR) -> full check", file=sys.stderr)
        return full_check()
    files = changed_files(base, head)
    if files is None:
        print("Could not diff against base -> full check", file=sys.stderr)
        return full_check()

    head_cfg = yaml.safe_load(open(CONFIG).read())
    base_config_text = git_show(base, CONFIG) if CONFIG in files else None
    action, affected, reason = classify(files, head_cfg, base_config_text)

    if action == "full":
        print(f"{reason} -> full check", file=sys.stderr)
        return full_check()
    if action == "skip":
        print("No lock-bearing depset input changed -> skipping check-depsets", file=sys.stderr)
        return 0
    return scoped_check(affected, template_instances(head_cfg))


if __name__ == "__main__":
    sys.exit(main())
