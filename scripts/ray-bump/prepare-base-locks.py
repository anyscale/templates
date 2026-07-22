#!/usr/bin/env python3
"""Prepare the base depset locks for a new Ray version so a fanout can fire.

The fanout (`.github/workflows/ray-bump-fanout.yaml`) only fires a version whose
`dependencies/depsets/` already holds a *complete* base-lock set — both a
`ray_<v>_img_py*.lock` (image) and a `rayllm_<v>_*.lock` (LLM). Producing that set
is otherwise a human hand-running `workflows/upgrade-dependencies.md`. This closes
that gap: it resolves the newest stable Ray, and if we don't have its base locks
yet, edits `dependencies/template.depsets.yaml` to add the version's
`build_arg_sets` + wire them into the two base `compile` entries, then recompiles
just those locks via `update_deps.sh`, leaving the changes staged for a PR.

Version policy (minor-only): on the scheduled/auto path it acts only when the newest
stable Ray advances the *minor* (or major) over our newest complete set — templates
track minor Ray releases (~monthly), so a patch over the current minor (2.56.0 →
2.56.1) is a no-op. A new minor adopts its newest patch (2.57.z, not forced 2.57.0).
An explicit `--version` (or `--force`) bypasses the gate — a human override for a
specific target.

Copy-forward model: it clones the current newest-complete version's base-lock
matrix (the image's Python set, the LLM's (py, cuda) set), substituting the new
version, and verifies Ray published the matching deplocks at the `ray-<v>` tag. It
does NOT invent a matrix: if Ray shipped a *different* one for <v> (a py/cuda
added, dropped, or moved — as happened 2.55→2.56), it stops with 'needs human'
rather than guess. A human then runs `upgrade-dependencies.md`. The base-locks PR
is human-reviewed either way, so the job need only be correct-or-obviously-stuck.

Exit codes: 0 = nothing to do (already current / waiting on upstream / dry-run);
            10 = changes prepared (caller should open a PR);
            2  = needs human (matrix changed, recompile failed, or bad input).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

def _repo_root() -> Path:
    """Nearest ancestor dir containing BUILD.yaml (robust to where this script lives)."""
    for p in Path(__file__).resolve().parents:
        if (p / "BUILD.yaml").is_file():
            return p
    raise RuntimeError("repo root not found: no BUILD.yaml above this script")


REPO_ROOT = _repo_root()
DEPSETS = REPO_ROOT / "dependencies" / "depsets"
CONFIG = REPO_ROOT / "dependencies" / "template.depsets.yaml"
UPDATE_DEPS = REPO_ROOT / "update_deps.sh"

# The two base compile entries, keyed by their (interpolated-name) templates.
IMG_ENTRY = "ray_depset_${RAY_VERSION}_${PYTHON_VERSION}"
LLM_ENTRY = "ray_llm_depset_${RAY_VERSION}_${PYTHON_VERSION}_${CUDA_VARIANT}"

RAY_TAG = "ray-{v}"  # ray-project/ray release tag holding python/deplocks/


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


def _lock_versions(*patterns: str) -> set[str]:
    rxs = [re.compile(p) for p in patterns]
    out: set[str] = set()
    for f in DEPSETS.glob("*.lock"):
        for rx in rxs:
            if m := rx.match(f.name):
                out.add(m.group(1))
    return out


def complete_versions() -> set[str]:
    """Versions present as BOTH a ray_<v>_img_* and an LLM base lock.

    Keep the definition in sync with scripts/ray-bump/latest-depset-version.py (same contract).
    """
    img = _lock_versions(r"ray_(\d+\.\d+\.\d+)_img_")
    llm = _lock_versions(r"rayllm_(\d+\.\d+\.\d+)_", r"ray_(\d+\.\d+\.\d+)_llm_")
    return img & llm


def newest_complete() -> str | None:
    c = complete_versions()
    return max(c, key=_ver) if c else None


def is_minor_upgrade(target: str, current: str | None) -> bool:
    """Whether `target` advances (major, minor) beyond `current` — the only case the
    scheduled path auto-prepares. Patch-only bumps (same major.minor) and versions
    behind `current` return False; a missing `current` (bootstrap) returns True."""
    if current is None:
        return True
    return _ver(target)[:2] > _ver(current)[:2]


# ── Ray deplock discovery (what matrix did upstream actually ship?) ─────────


def _gh_contents(subdir: str, tag: str) -> list[str] | None:
    """File names under python/deplocks/<subdir> at ray-project/ray@<tag>.
    None if the tag or directory doesn't exist yet (404)."""
    url = f"https://api.github.com/repos/ray-project/ray/contents/python/deplocks/{subdir}?ref={tag}"
    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json", "User-Agent": "prepare-base-locks"}
    )
    if tok := (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")):
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return [it["name"] for it in json.load(r) if it["type"] == "file"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def discover_matrix(v: str) -> tuple[set[str] | None, set[tuple[str, str]]]:
    """Published (image python-shorts, LLM (py, cuda)) for Ray <v>.
    image is None when the tag/deplocks aren't published yet."""
    tag = RAY_TAG.format(v=v)
    img_files = _gh_contents("ray_img", tag)
    llm_files = _gh_contents("llm", tag) or []
    img = None
    if img_files is not None:
        img = {m.group(1) for f in img_files if (m := re.match(r"ray_img_py(\d+)\.lock$", f))}
    llm = {
        (m.group(1), m.group(2))
        for f in llm_files
        if (m := re.match(r"rayllm_py(\d+)_(cu\d+)\.lock$", f))
    }
    return img, llm


# ── config edit (ruamel round-trip: preserve comments + layout) ─────────────


def _yaml():
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    y.width = 4096  # don't wrap long sequences
    y.indent(mapping=2, sequence=4, offset=2)  # match the file's block style (no reformat churn)
    return y


def _find_entry(cfg, name_template):
    for e in cfg["depsets"]:
        if e.get("name") == name_template:
            return e
    raise RuntimeError(f"base compile entry not found: {name_template}")


def bundles_for(cfg, entry, version) -> list[dict]:
    """The build_arg_set dicts wired into `entry` for `version`, in order."""
    prefix = f"ray{compact(version)}_"
    bas = cfg["build_arg_sets"]
    return [dict(bas[name]) for name in entry["build_arg_sets"] if name.startswith(prefix)]


def bundle_name(v: str, b: dict) -> str:
    return f"ray{compact(v)}_py{b['PYTHON_SHORT']}_{b['CUDA_VARIANT']}"


def apply_edit(cfg, target: str, prev_img: list[dict], prev_llm: list[dict]) -> list[str]:
    """Add target bundles + wire them into the two base entries. Returns the
    build instance names to recompile."""
    from ruamel.yaml.comments import CommentedMap
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq

    bas = cfg["build_arg_sets"]
    instances: list[str] = []
    for entry_name, prev_bundles in ((IMG_ENTRY, prev_img), (LLM_ENTRY, prev_llm)):
        entry = _find_entry(cfg, entry_name)
        for pb in prev_bundles:
            name = bundle_name(target, pb)
            if name not in bas:
                m = CommentedMap()
                m["RAY_VERSION"] = dq(target)
                m["PYTHON_VERSION"] = dq(pb["PYTHON_VERSION"])
                m["PYTHON_SHORT"] = dq(pb["PYTHON_SHORT"])
                m["CUDA_VARIANT"] = dq(pb["CUDA_VARIANT"])
                bas[name] = m
            if name not in entry["build_arg_sets"]:
                entry["build_arg_sets"].append(name)
            # interpolate the entry's name template for --name
            inst = (
                entry_name.replace("${RAY_VERSION}", target)
                .replace("${PYTHON_VERSION}", pb["PYTHON_VERSION"])
                .replace("${CUDA_VARIANT}", pb["CUDA_VARIANT"])
            )
            if inst not in instances:
                instances.append(inst)
    return instances


def expected_outputs(target: str, prev_img: list[dict], prev_llm: list[dict]) -> list[Path]:
    out = [DEPSETS / f"ray_{target}_img_py{b['PYTHON_SHORT']}.lock" for b in prev_img]
    out += [DEPSETS / f"rayllm_{target}_py{b['PYTHON_SHORT']}_{b['CUDA_VARIANT']}.lock" for b in prev_llm]
    return out


def recompile(instances: list[str]) -> None:
    for name in instances:
        log(f"→ update_deps.sh --name {name}")
        subprocess.run(["bash", str(UPDATE_DEPS), "--name", name], cwd=REPO_ROOT, check=True)


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
        log("Already have a complete base-lock set — nothing to do.")
        set_output(status="current", version=target)
        return 0

    prev = newest_complete()
    if prev is None:
        log("error: no existing complete version to copy the matrix forward from — needs human.")
        set_output(status="needs-human", version=target)
        return 2
    log(f"Copying the base-lock matrix forward from {prev}.")

    cfg = _yaml().load(CONFIG.read_text())
    prev_img = bundles_for(cfg, _find_entry(cfg, IMG_ENTRY), prev)
    prev_llm = bundles_for(cfg, _find_entry(cfg, LLM_ENTRY), prev)
    if not prev_img or not prev_llm:
        log(f"error: couldn't read {prev}'s base matrix from the config — needs human.")
        set_output(status="needs-human", version=target)
        return 2

    # Verify Ray published the deplocks this matrix needs at ray-<target>.
    pub_img, pub_llm = discover_matrix(target)
    if pub_img is None:
        log(f"Ray {target} deplocks not published yet (no {RAY_TAG.format(v=target)} tag / ray_img dir). Waiting.")
        set_output(status="waiting", version=target)
        return 0
    need_img = {b["PYTHON_SHORT"] for b in prev_img}
    if not need_img <= pub_img:
        log(f"Image matrix changed for {target}: need py{sorted(need_img)}, Ray published py{sorted(pub_img)} — needs human.")
        set_output(status="needs-human", version=target)
        return 2
    if not pub_llm:
        log(f"Ray {target} image deplocks are up but LLM deplocks aren't yet. Waiting.")
        set_output(status="waiting", version=target)
        return 0
    need_llm = {(b["PYTHON_SHORT"], b["CUDA_VARIANT"]) for b in prev_llm}
    if not need_llm <= pub_llm:
        log(f"LLM matrix changed for {target}: need {sorted(need_llm)}, Ray published {sorted(pub_llm)} — needs human.")
        set_output(status="needs-human", version=target)
        return 2
    extra = (pub_img - need_img, pub_llm - need_llm)
    if extra[0] or extra[1]:
        log(f"note: Ray also published variants not in our matrix (img py{sorted(extra[0])}, llm {sorted(extra[1])}); "
            "not auto-added — run upgrade-dependencies.md if you want them.")

    instances = apply_edit(cfg, target, prev_img, prev_llm)
    log(f"Base locks to build: {', '.join(instances)}")

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
    try:
        recompile(instances)
    except subprocess.CalledProcessError as e:
        log(f"recompile failed ({e}) — needs human.")
        set_output(status="needs-human", version=target)
        return 2

    missing = [str(o.relative_to(REPO_ROOT)) for o in expected_outputs(target, prev_img, prev_llm) if not o.exists()]
    if missing:
        log(f"recompile did not produce: {missing} — needs human.")
        set_output(status="needs-human", version=target)
        return 2

    log(f"Prepared complete base-lock set for Ray {target}.")
    set_output(status="prepared", version=target)
    return 10


if __name__ == "__main__":
    sys.exit(main())
