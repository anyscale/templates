#!/usr/bin/env python3
"""Static dependency-delivery checks (see .claude/skills/template/references/dependencies.md).

Three failure modes that are invisible to `template-test` because CI happens to paper
over each one:

  lock-image  A lock seeded from image A while BUILD.yaml runs the template on image B.
              The lock then describes packages that aren't in the image it pins against.
  driver      A template ships a lock nothing installs, so users run on whatever the
              image happens to have. Green in CI when tests.sh installs deps separately.
  tests-pip   A bare `pip install` in tests.sh. A workspace *tracks* it and the
              runtime-env hook appends it, unpinned, to every actor's pip list; one
              unhashed entry trips pip's --require-hashes against a hashed lock and
              every runtime env for that template fails to build.

Usage: check-dep-delivery.py [lock-image|driver|tests-pip ...]   (default: all)
"""
import re
import sys
from pathlib import Path

import yaml

BUILD_YAML = Path("BUILD.yaml")
DEPSETS = Path("dependencies/template.depsets.yaml")
TESTS = Path("tests")
LOCK = "python_depset.lock"

# `ray-2.56.0-py311-cu121.freeze.txt` -> `anyscale/ray:2.56.0-py311-cu121`
FREEZE_RE = re.compile(r"^(ray(?:-llm)?)-(\d+\.\d+\.\d+.*)\.freeze\.txt$")
SEED_RE = re.compile(r"seed-image-freeze\.sh\s+(\S+)")
# An install that reads the lock as a requirements file, at any path spelling.
LOCK_INSTALL_RE = re.compile(rf"-r\s+\S*{re.escape(LOCK)}")
# ...and the same install with `-r` forgotten, which makes uv read it as a package name.
LOCK_NO_R_RE = re.compile(rf"pip\s+install\s+(?:-(?!r\b)\S+\s+)*\S*{re.escape(LOCK)}")
BARE_PIP_RE = re.compile(r"(?<!uv )\bpip\d?\s+install\b")

# Files a user actually runs. Excludes the lock itself and generated READMEs' twin.
SOURCE_GLOBS = ("*.ipynb", "*.md", "*.py", "*.sh", "*.yaml", "*.yml", "*.toml")


def templates():
    """[(name, dir, image_uri)] from BUILD.yaml."""
    entries = yaml.safe_load(BUILD_YAML.read_text())
    return [
        (e["name"], Path(e["dir"]), (e.get("cluster_env") or {}).get("image_uri", ""))
        for e in entries
    ]


def source_files(root: Path):
    """Every file a user runs, minus the nbconvert twins — those mirror a notebook's
    install line, so counting them lets an edited notebook still look installed."""
    for glob in SOURCE_GLOBS:
        for f in root.rglob(glob):
            if f.suffix == ".md" and f.with_suffix(".ipynb").exists():
                continue
            yield f


def check_lock_image():
    """Each depset's seed freeze must name the image BUILD.yaml runs that template on."""
    by_dir = {d: (name, image) for name, d, image in templates()}
    cfg = yaml.safe_load(DEPSETS.read_text())
    arg_sets = cfg["build_arg_sets"]
    problems = []

    for depset in cfg["depsets"]:
        out = Path(depset.get("output", ""))
        entry = by_dir.get(out.parent)
        if not entry or out.name != LOCK:
            continue
        name, image = entry

        for hook in depset.get("pre_hooks", []):
            m = SEED_RE.search(hook)
            if not m:
                continue
            for set_name in depset.get("build_arg_sets", []):
                freeze = m.group(1)
                for k, v in arg_sets[set_name].items():
                    freeze = freeze.replace(f"${{{k}}}", v)
                fm = FREEZE_RE.match(Path(freeze).name)
                if not fm:
                    problems.append(f"{name}: unparseable freeze name {Path(freeze).name}")
                    continue
                expected = f"anyscale/{fm.group(1)}:{fm.group(2)}"
                if expected != image:
                    problems.append(
                        f"{name}: lock is seeded from {expected} but BUILD.yaml runs {image}"
                    )
    return problems


def check_driver():
    """A template that ships a lock must install it, with `-r`, somewhere a user runs."""
    problems = []
    for name, root, _ in templates():
        if not (root / LOCK).exists():
            continue
        installs, no_r = False, []
        for f in source_files(root):
            text = f.read_text(errors="ignore")
            if LOCK_INSTALL_RE.search(text):
                installs = True
            elif LOCK_NO_R_RE.search(text):
                no_r.append(f)
        if installs:
            continue
        if no_r:
            problems.append(f"{name}: {no_r[0]} installs {LOCK} without `-r` (uv reads it as a package name)")
        else:
            problems.append(f"{name}: ships {LOCK} but nothing installs it")
    return problems


def check_tests_pip():
    """tests.sh must not bare-`pip install`. Only a template shipping a lock can be bitten
    today, so only those fail; the rest are listed, since each becomes a live break the
    moment it gains a lock."""
    problems, latent = [], []
    for name, root, _ in templates():
        script = TESTS / name / "tests.sh"
        if not script.exists():
            continue
        for n, line in enumerate(script.read_text().splitlines(), 1):
            if line.lstrip().startswith("#") or not BARE_PIP_RE.search(line):
                continue
            found = f"{name}: {script}:{n}\n      {line.strip()}"
            (problems if (root / LOCK).exists() else latent).append(found)

    if latent:
        print(f"note: {len(latent)} lock-less template(s) also bare-pip in tests.sh. Harmless "
              "until they gain a lock; convert to `uv pip install --system` when touched:")
        for f in latent:
            print(f"  - {f}")
    return problems


CHECKS = {
    "lock-image": check_lock_image,
    "driver": check_driver,
    "tests-pip": check_tests_pip,
}

if __name__ == "__main__":
    selected = sys.argv[1:] or list(CHECKS)
    failed = False
    for check in selected:
        problems = CHECKS[check]()
        if problems:
            failed = True
            print(f"\n{check}: {len(problems)} problem(s)")
            for p in problems:
                print(f"  - {p}")
    if failed:
        print("\nSee .claude/skills/template/references/dependencies.md")
    sys.exit(1 if failed else 0)
