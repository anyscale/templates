#!/usr/bin/env python3
"""Static dependency-delivery checks (see .claude/skills/template/references/dependencies.md).

Four failure modes that are invisible to `template-test` because CI happens to paper
over each one:

  depset-config   A depset entry naming a freeze for one image while BUILD.yaml runs the
                  template on another, or compiling a lock without `include_setuptools:
                  true`. The first makes the lock describe an environment the template
                  never runs in; the second lets pip collect setuptools unpinned into a
                  hashed lock, which fails every runtime env for that template.
  lock-installed  A template ships a lock and nothing installs it, so users run on
                  whatever the image happens to have -- or installs it without `-r`, so
                  uv reads the filename as a package name and hard fails. Green in CI
                  whenever tests.sh installs the deps some other way.
  bare-pip        A bare `pip install` in tests.sh. A workspace *tracks* it and the
                  runtime-env hook appends it, unpinned, to every actor's pip list; one
                  unhashed entry trips pip's hash-checking mode against a hashed lock and
                  every runtime env for that template fails to build.
  pin-style       A requirement that isn't `==`. The lock then re-resolves to whatever is
                  newest whenever it's regenerated, so a template's behaviour changes
                  without anyone editing it -- and CI passes, because it tests the drift.

Usage: check-dep-delivery.py [depset-config|lock-installed|bare-pip|pin-style ...]
                                                                       (default: all)
"""
import re
import sys
from pathlib import Path

import yaml

BUILD_YAML = Path("BUILD.yaml")
DEPSETS = Path("dependencies/template.depsets.yaml")
IMAGES = Path("dependencies/images")
TESTS = Path("tests")
LOCK = "python_depset.lock"

# Leading package name of a requirement, before any extras, specifier or URL.
REQ_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
# A pin in a compiled lock or a pip freeze, stopping before any marker or continuation.
PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")

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


def norm(name):
    """PEP 503 name, so `chembl_structure_pipeline` and `chembl-structure-pipeline` match."""
    return re.sub(r"[-_.]+", "-", name).lower()


def expand(text, build_args):
    """`ray-${RAY_VERSION}-py311.freeze.txt` -> `ray-2.56.0-py311.freeze.txt`."""
    for k, v in build_args.items():
        text = text.replace(f"${{{k}}}", v)
    return text


def templates():
    """[(name, dir, image_uri)] from BUILD.yaml."""
    entries = yaml.safe_load(BUILD_YAML.read_text())
    return [
        (e["name"], Path(e["dir"]), (e.get("cluster_env") or {}).get("image_uri", ""))
        for e in entries
    ]


def depset_targets():
    """[(dir, freeze_path, python_version)] for every template lock the depsets compile."""
    cfg = yaml.safe_load(DEPSETS.read_text())
    arg_sets = cfg["build_arg_sets"]
    targets = []
    for depset in cfg["depsets"]:
        out = Path(depset.get("output", ""))
        if out.name != LOCK:
            continue
        for hook in depset.get("pre_hooks", []):
            m = SEED_RE.search(hook)
            if not m:
                continue
            for set_name in depset.get("build_arg_sets", []):
                args = arg_sets[set_name]
                targets.append(
                    (out.parent, Path(expand(m.group(1), args)), args["PYTHON_VERSION"])
                )
    return targets


def pins(path):
    """{package: version} for every pin in a compiled lock or a pip freeze."""
    return {
        norm(m.group(1)): m.group(2)
        for line in path.read_text().splitlines()
        if (m := PIN_RE.match(line))
    }


def source_files(root: Path):
    """Every file a user runs, minus the nbconvert twins — those mirror a notebook's
    install line, so counting them lets an edited notebook still look installed."""
    for glob in SOURCE_GLOBS:
        for f in root.rglob(glob):
            if f.suffix == ".md" and f.with_suffix(".ipynb").exists():
                continue
            yield f


def check_depset_config():
    """Each depset entry must name the image its template runs on, and pin setuptools.

    Two ways an entry goes wrong, both invisible to the compile itself:

    freeze/image  The lock is *seeded* from the freeze this entry names, so every pin in
                  it describes that image. Point it at a different one than BUILD.yaml
                  runs and the lock asserts versions the runtime doesn't have -- then
                  installs them into every runtime env. `${RAY_VERSION}` interpolates on
                  a bump, but the literal py/CUDA parts don't, so moving a template
                  between images is where this slips.
    setuptools    uv drops setuptools from a lock by default, while Ray builds each
                  runtime env as a virtualenv seeding its own. A locked package wanting
                  a newer one then makes pip collect it -- unpinned, unhashed against a
                  lock full of hashes, which puts pip in hash-checking mode and fails
                  every runtime env for the template. With `include_setuptools: true`
                  it is either pinned at the image's version or absent because nothing
                  in the graph wants it.
    """
    by_dir = {d: (name, image) for name, d, image in templates()}
    cfg = yaml.safe_load(DEPSETS.read_text())
    arg_sets = cfg["build_arg_sets"]
    problems = []

    for depset in cfg["depsets"]:
        out = Path(depset.get("output", ""))
        if out.name != LOCK:
            continue

        if depset.get("include_setuptools") is not True:
            problems.append(
                f"{out.parent.name}: {DEPSETS} entry `{depset['name']}` compiles a "
                f"template lock without `include_setuptools: true`.\n"
                f"      Add it and recompile (`./update_deps.sh --name {depset['name']}`)."
            )

        entry = by_dir.get(out.parent)
        if not entry:
            continue
        name, image = entry
        for hook in depset.get("pre_hooks", []):
            m = SEED_RE.search(hook)
            if not m:
                continue
            for set_name in depset.get("build_arg_sets", []):
                freeze = expand(m.group(1), arg_sets[set_name])
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


def check_lock_installed():
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


def check_bare_pip():
    """tests.sh must not bare-`pip install`.

    Only a template shipping a lock can be bitten, so only those fail. A lock-less
    template is left alone rather than reported: for a pure workspace tutorial the
    bare install *is* the lesson, and it only becomes a break if that template ever
    gains a lock -- at which point this fails and says so.
    """
    problems = []
    for name, root, _ in templates():
        script = TESTS / name / "tests.sh"
        if not script.exists() or not (root / LOCK).exists():
            continue
        for n, line in enumerate(script.read_text().splitlines(), 1):
            if line.lstrip().startswith("#") or not BARE_PIP_RE.search(line):
                continue
            problems.append(f"{name}: {script}:{n}\n      {line.strip()}")
    return problems


def requirement_lines(path: Path):
    """[(lineno, package, spec, has_justification)] for real requirements only."""
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line, _, comment = raw.partition("#")
        spec = line.strip()
        if not spec or spec.startswith("-"):
            continue
        package = REQ_NAME_RE.match(spec)
        yield n, (package.group(1).lower() if package else spec), spec, bool(comment.strip())


def check_pin_style():
    """Every requirement is `==`, or carries a trailing comment saying why it isn't."""
    problems = []
    for name, root, _ in templates():
        req = root / "requirements.txt"
        if not req.exists():
            continue
        for n, _package, spec, justified in requirement_lines(req):
            if "==" in spec and "@" not in spec:
                continue
            if justified:
                continue
            problems.append(
                f"{name}: {req}:{n}\n      {spec}\n"
                f"      Pin it with `==` at the newest version that works on this template's "
                f"image, or add a trailing comment saying why it can't be."
            )
    return problems


CHECKS = {
    "depset-config": check_depset_config,
    "lock-installed": check_lock_installed,
    "bare-pip": check_bare_pip,
    "pin-style": check_pin_style,
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
