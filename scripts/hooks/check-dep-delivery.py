#!/usr/bin/env python3
"""Static dependency-delivery checks (see .claude/skills/template/references/dependencies.md).

Five failure modes that are invisible to `template-test` because CI happens to paper
over each one:

  lock-image  A lock seeded from image A while BUILD.yaml runs the template on image B.
              The lock then describes packages that aren't in the image it pins against.
  driver      A template ships a lock nothing installs, so users run on whatever the
              image happens to have. Green in CI when tests.sh installs deps separately.
  tests-pip   A bare `pip install` in tests.sh. A workspace *tracks* it and the
              runtime-env hook appends it, unpinned, to every actor's pip list; one
              unhashed entry trips pip's --require-hashes against a hashed lock and
              every runtime env for that template fails to build.
  pin-style   A requirement that isn't `==`. The lock then re-resolves to whatever is
              newest whenever it's regenerated, so a template's behaviour changes
              without anyone editing it -- and CI passes, because it tests the drift.
  depset-config  A template lock compiled without `include_setuptools: true`. uv then
              drops setuptools from the lock, while Ray builds each runtime env as a
              virtualenv seeding its own -- so a locked package wanting a newer one
              makes pip collect it unpinned and unhashed, which against a hashed lock
              puts pip in hash-checking mode and fails every runtime env for the
              template. With the flag on, setuptools is either pinned at the image's
              version or absent because nothing in the graph wants it.

Usage: check-dep-delivery.py [lock-image|driver|tests-pip|pin-style|depset-config ...]
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


def check_depset_config():
    """Every template lock is compiled with `include_setuptools: true`.

    uv drops setuptools from a lock by default. Ray builds each runtime env as a
    virtualenv that seeds its own setuptools, so a locked package requiring a newer
    one makes pip collect it -- unpinned, and unhashed against a lock full of hashes,
    which puts pip in hash-checking mode and fails every runtime env for the template.
    With the flag on, setuptools is either in the lock at the image's version (the
    seed decides it) or absent because nothing in the graph wants it. Both are safe;
    the flag is what makes the second case the only alternative to the first.
    """
    cfg = yaml.safe_load(DEPSETS.read_text())
    return [
        f"{Path(depset['output']).parent.name}: {DEPSETS} entry `{depset['name']}` "
        f"compiles a template lock without `include_setuptools: true`.\n"
        f"      Add it and recompile (`./update_deps.sh --name {depset['name']}`)."
        for depset in cfg["depsets"]
        if Path(depset.get("output", "")).name == LOCK
        and depset.get("include_setuptools") is not True
    ]


CHECKS = {
    "lock-image": check_lock_image,
    "driver": check_driver,
    "tests-pip": check_tests_pip,
    "pin-style": check_pin_style,
    "depset-config": check_depset_config,
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
