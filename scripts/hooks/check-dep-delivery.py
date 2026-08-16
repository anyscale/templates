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
  venv-seed   A locked package whose runtime requirement on setuptools/pip/wheel the
              runtime env already contradicts. Ray builds that env with a virtualenv
              that seeds its own setuptools, and pip never collects a requirement the
              seeded copy already satisfies -- so only a lower bound *above* the seed
              forces an upgrade, and that upgrade resolves unpinned to newest, has no
              hash in the lock, and fails --require-hashes. Sixteen templates ship a
              package that requires setuptools and fifteen are fine, so the bound is
              what matters, never the presence.

Usage: check-dep-delivery.py [lock-image|driver|tests-pip|pin-style|venv-seed ...]
                                                                       (default: all)
       check-dep-delivery.py --refresh   rebuild dependencies/venv-seed-cache.txt from
                                         PyPI (network; the checks themselves never fetch)
"""
import re
import sys
from pathlib import Path

import yaml
from packaging.requirements import InvalidRequirement, Requirement

BUILD_YAML = Path("BUILD.yaml")
DEPSETS = Path("dependencies/template.depsets.yaml")
PIN_EXCEPTIONS = Path("dependencies/loose-pins-allowlist.txt")
IMAGES = Path("dependencies/images")
SEED_CACHE = Path("dependencies/venv-seed-cache.txt")
TESTS = Path("tests")
LOCK = "python_depset.lock"

# The packages a virtualenv seeds into the env it creates, so pip finds them already
# installed. raydepsets strips setuptools from every lock by default; pip and wheel it
# leaves alone, but the same "already satisfied, never collected" rule governs all three.
SEEDED = ("pip", "setuptools", "wheel")

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

# Python versions to test a dependency's marker against when deciding, at refresh time,
# whether it can ever apply at runtime. Anything false for all of them is extras-only.
PY_CANDIDATES = ("3.9", "3.10", "3.11", "3.12", "3.13", "3.14")


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
    """Every requirement is `==`, or carries a trailing comment saying why it isn't.

    The allowlist grandfathers what predates the rule so the check is green today; it is
    a backlog, not a config, and every entry removed is a template that stopped drifting.
    """
    allowed = set()
    if PIN_EXCEPTIONS.exists():
        for line in PIN_EXCEPTIONS.read_text().splitlines():
            entry = line.split("#")[0].strip()
            if entry:
                allowed.add(entry)

    problems, grandfathered, stale = [], 0, []
    roots = {name: root for name, root, _ in templates()}
    for name, root in roots.items():
        req = root / "requirements.txt"
        if not req.exists():
            continue
        for n, package, spec, justified in requirement_lines(req):
            if "==" in spec and "@" not in spec:
                continue
            if justified:
                continue
            if f"{name}/{package}" in allowed:
                grandfathered += 1
                continue
            problems.append(
                f"{name}: {req}:{n}\n      {spec}\n"
                f"      Pin it with `==` at the newest version that works on this template's "
                f"image, or add a trailing comment saying why it can't be."
            )

    for entry in sorted(allowed):
        tmpl, _, package = entry.partition("/")
        req = roots.get(tmpl, Path("/nonexistent")) / "requirements.txt"
        if not req.exists() or not any(
            p == package and not ("==" in s and "@" not in s) and not j
            for _, p, s, j in requirement_lines(req)
        ):
            stale.append(entry)

    if grandfathered:
        print(f"note: {grandfathered} loose pin(s) still grandfathered in {PIN_EXCEPTIONS}. "
              "Each one re-resolves to whatever is newest when its lock is rebuilt.")
    if stale:
        print(f"note: {len(stale)} allowlist entry/entries no longer needed — delete them:")
        for entry in stale:
            print(f"  - {entry}")
    return problems


def marker_env(python_version):
    """A Linux cluster node running `python_version`, with no extra requested — so a
    requirement gated behind `extra == "dev"` evaluates false rather than raising."""
    return {
        "python_version": python_version,
        "python_full_version": f"{python_version}.0",
        "implementation_version": f"{python_version}.0",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "sys_platform": "linux",
        "os_name": "posix",
        "extra": "",
    }


def marker_applies(marker, python_version):
    """A marker we can't evaluate is treated as live: over-reporting is recoverable,
    silently dropping a requirement is the bug this check exists to catch."""
    if marker is None:
        return True
    try:
        return bool(marker.evaluate(marker_env(python_version)))
    except Exception:
        return True


def read_seed_cache():
    """({(virtualenv, python, package): version}, {(package, version): [requirement]}).

    A `requires` key with an empty list is a positive record of "this version declares
    none" — it is what keeps an unseen version distinguishable from a checked one.
    """
    seeds, requires = {}, {}
    if not SEED_CACHE.exists():
        return seeds, requires
    for raw in SEED_CACHE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        kind, _, rest = line.partition(" ")
        if kind == "seed":
            venv, python, package, version = rest.split(" ", 3)
            seeds[(venv, python, norm(package))] = version
        elif kind == "requires":
            package, version, spec = rest.split(" ", 2)
            entry = requires.setdefault((norm(package), version), [])
            if spec != "-":
                entry.append(spec)
    return seeds, requires


def check_venv_seed():
    """No locked package may require a seeded package at a version the runtime env's
    virtualenv doesn't already provide, unless the lock pins that package too.

    The seed is never assumed: the images' virtualenv version comes from their freezes,
    and its bundled versions from dependencies/venv-seed-cache.txt. A freeze naming a
    virtualenv the cache has no seeds for is a failure, not a fallback.
    """
    seeds, requires = read_seed_cache()
    if not seeds:
        return [
            f"{SEED_CACHE} is missing or has no seed versions, so what the runtime env "
            f"already provides is unknown.\n"
            f"      Run `python3 scripts/hooks/check-dep-delivery.py --refresh`."
        ]

    known_requirers = {package for package, _ in requires}
    names = {d: name for name, d, _ in templates()}
    freezes, problems = {}, []

    for tdir, freeze, python in depset_targets():
        lock = tdir / LOCK
        name = names.get(tdir, str(tdir))
        if not lock.exists():
            continue
        if not freeze.exists():
            problems.append(f"{name}: depset seeds from {freeze}, which doesn't exist")
            continue

        if freeze not in freezes:
            freezes[freeze] = pins(freeze)
        image = freezes[freeze]
        venv = image.get("virtualenv")
        if not venv:
            problems.append(
                f"{name}: {freeze} pins no virtualenv, so the version Ray's runtime env "
                f"seeds into every actor can't be established."
            )
            continue
        if not any(cached == venv for cached, _, _ in seeds):
            problems.append(
                f"{name}: {freeze} ships virtualenv=={venv}, which {SEED_CACHE} has no "
                f"seed versions for.\n"
                f"      Run `python3 scripts/hooks/check-dep-delivery.py --refresh`."
            )
            continue

        locked = pins(lock)
        for package, version in sorted(locked.items()):
            if package not in known_requirers:
                continue
            specs = requires.get((package, version))
            if specs is None:
                problems.append(
                    f"{name}: {lock} pins {package}=={version}, a package that requires a "
                    f"seeded package at some versions, but {SEED_CACHE} has no metadata for "
                    f"this one.\n"
                    f"      Run `python3 scripts/hooks/check-dep-delivery.py --refresh`."
                )
                continue

            for raw in specs:
                try:
                    req = Requirement(raw)
                except InvalidRequirement:
                    continue
                if not marker_applies(req.marker, python):
                    continue

                seeded = norm(req.name)
                provided = seeds.get((venv, python, seeded))
                if provided == "-":
                    # virtualenv bundles nothing for this pair, so the image's own copy
                    # is what shows through the env's system-site-packages.
                    provided, origin = image.get(seeded), f"{freeze.name} ships"
                else:
                    origin = f"virtualenv {venv} seeds"
                if provided is None:
                    problems.append(
                        f"{name}: neither virtualenv {venv} nor {freeze.name} provides "
                        f"{seeded}, and {SEED_CACHE} records no seed for python {python}.\n"
                        f"      Run `python3 scripts/hooks/check-dep-delivery.py --refresh`."
                    )
                    continue

                if req.specifier.contains(provided, prereleases=True):
                    continue
                if seeded in locked:
                    continue

                problems.append(
                    f"{name}: {lock}\n"
                    f"      {package}=={version} requires `{raw}`, and {origin} "
                    f"{seeded} {provided}, which doesn't satisfy it.\n"
                    f"      pip leaves an already-satisfied requirement alone, so only a bound "
                    f"the seed misses forces it to collect {seeded} — unpinned, resolving to "
                    f"newest, with no hash in the lock. --require-hashes then fails every "
                    f"runtime env for this template.\n"
                    f"      Emit {seeded} into the lock (`include_{seeded}: true` on this "
                    f"depset in {DEPSETS}) and recompile."
                )
    return problems


def refresh():
    """Rebuild SEED_CACHE from PyPI and the virtualenv wheels the images ship.

    Network-bound and manual, so the hook never fetches. Entries are merged, never
    dropped: released metadata is immutable, and a version that leaves the locks today
    is one `git revert` away from coming back.
    """
    import ast
    import io
    import json
    import urllib.request
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    seeds, requires = read_seed_cache()

    def pypi(name, version):
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.load(resp)

    virtualenvs = set()
    for freeze in sorted(IMAGES.glob("*.freeze.txt")):
        version = pins(freeze).get("virtualenv")
        if version:
            virtualenvs.add(version)
    print(f"images ship virtualenv {', '.join(sorted(virtualenvs)) or '(none found)'}")

    for venv in sorted(virtualenvs):
        meta = pypi("virtualenv", venv)
        url = next(f["url"] for f in meta["urls"] if f["packagetype"] == "bdist_wheel")
        with urllib.request.urlopen(url, timeout=120) as resp:
            wheel = zipfile.ZipFile(io.BytesIO(resp.read()))
        src = wheel.read("virtualenv/seed/wheels/embed/__init__.py").decode()
        bundle = next(
            ast.literal_eval(node.value)
            for node in ast.parse(src).body
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "BUNDLE_SUPPORT" for t in node.targets)
        )
        for python, dists in bundle.items():
            for package in SEEDED:
                filename = dists.get(package)
                # `setuptools-75.8.0-py3-none-any.whl` -> `75.8.0`
                seeds[(venv, python, package)] = filename.split("-")[1] if filename else "-"
        print(f"  virtualenv {venv}: seeds for python {', '.join(bundle)}")

    wanted = set()
    for lock in sorted(Path("templates").glob(f"*/{LOCK}")):
        wanted |= set(pins(lock).items())

    def fetch(pair):
        name, version = pair
        # A lock can pin a local version (`2.7.0+cu128`) served by the torch index; PyPI
        # only knows the upstream release, whose requirements are the ones that matter.
        for candidate in (version, version.split("+")[0]):
            try:
                meta = pypi(name, candidate)
            except Exception:
                continue
            found = []
            for raw in meta["info"].get("requires_dist") or []:
                try:
                    req = Requirement(raw)
                except InvalidRequirement:
                    continue
                if norm(req.name) not in SEEDED:
                    continue
                # Keep anything that can apply on some Python; that drops requirements
                # reachable only through an extra, which nothing installs at runtime.
                if any(marker_applies(req.marker, py) for py in PY_CANDIDATES):
                    found.append(raw)
            return pair, found
        return pair, None

    with ThreadPoolExecutor(max_workers=16) as pool:
        fetched = dict(pool.map(fetch, sorted(wanted)))

    missing = [p for p, reqs in fetched.items() if reqs is None]
    for name, version in sorted(missing):
        print(f"  warning: no PyPI metadata for {name}=={version}; leaving it uncached")

    # A package already tracked stays tracked, so a version that stopped requiring
    # anything is still recorded as checked rather than silently becoming unknown.
    requirers = {name for (name, _), reqs in fetched.items() if reqs}
    requirers |= {name for name, _ in requires}
    for pair, reqs in fetched.items():
        if reqs is not None and pair[0] in requirers:
            requires[pair] = reqs

    lines = [
        "# Cached package metadata for the `venv-seed` check in scripts/hooks/check-dep-delivery.py.",
        "# Generated — do not hand-edit. Refresh with:",
        "#",
        "#   python3 scripts/hooks/check-dep-delivery.py --refresh",
        "#",
        "# `seed <virtualenv> <python> <package> <version>` — what that virtualenv installs into",
        "# every env it creates, read from its own BUNDLE_SUPPORT. Ray builds each runtime env",
        "# this way, and the seeded copy shadows the image's, so it is the version pip sees as",
        "# already installed. `-` means that virtualenv bundles nothing for the pair and the",
        "# image's own copy shows through instead. Which virtualenv the images ship comes from",
        "# dependencies/images/*.freeze.txt; a freeze naming one this file has no seeds for fails",
        "# the check rather than falling back to a guess.",
        "#",
        "# `requires <package> <version> <requirement>` — every runtime requirement on pip,",
        "# setuptools or wheel declared by a package some lock pins, verbatim from PyPI, with `-`",
        "# recording that the version declares none. Requirements reachable only through an extra",
        "# are dropped: nothing installs them. Only packages that require one of these somewhere",
        "# are tracked, so an unlisted version of a listed package fails the check as unknown.",
        "",
    ]
    lines += [
        f"seed {venv} {python} {package} {version}"
        for (venv, python, package), version in sorted(seeds.items())
    ]
    lines.append("")
    for (name, version), reqs in sorted(requires.items()):
        for raw in sorted(reqs) or ["-"]:
            lines.append(f"requires {name} {version} {raw}")

    SEED_CACHE.write_text("\n".join(lines) + "\n")
    tracked = len({name for name, _ in requires})
    print(
        f"wrote {SEED_CACHE}: {len(seeds)} seed versions, "
        f"{len(requires)} package versions across {tracked} package(s) that require one"
    )


CHECKS = {
    "lock-image": check_lock_image,
    "driver": check_driver,
    "tests-pip": check_tests_pip,
    "pin-style": check_pin_style,
    "venv-seed": check_venv_seed,
}

if __name__ == "__main__":
    if "--refresh" in sys.argv[1:]:
        refresh()
        sys.exit(0)
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
