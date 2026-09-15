"""Compile a WDL task's declared environment into a Ray ``runtime_env``.

The mirror image of :mod:`wdl_on_ray.resources`. That module turns ``runtime {}`` into a Ray
*resource* request; this one turns it into a Ray *environment*. Both are pure functions on
miniwdl's normalized ``runtime_values``, so both are testable without a cluster.

Why this exists
---------------
An OCI image is how WDL names a tool environment. Ray has its own way to name one --
``runtime_env``, so the ``native`` container runtime compiles between them. Where
``container_runtime = none`` demotes ``runtime.docker`` to a comment and requires every task's
tools to be present in one cluster image, ``native`` resolves each task to just the tools it
invokes and lets Ray materialize them: cached per node by content hash, ref-counted, and
garbage-collected, with no container started at any point.

What this is *not*
------------------
Environment isolation, not filesystem isolation. Ray provides no mount, PID or network
namespace, so a task still sees the host filesystem exactly as under ``none`` (Ray's own
"namespaces" are actor-name scopes, unrelated to the OS kind). What changes is that a task
gets only the tools it declares instead of every tool in the image.

It is also *closed-world*, which is the real trade against ``podman``. An OCI image carries
whatever its author put in it; a wheel manifest carries only what has been packaged. So a task
invoking something the manifest does not provide resolves to nothing for that name, which is
why unresolved names are reported before the task runs (see :class:`Resolved.unresolved`).

Resolution precedence
---------------------
Highest wins, and wins *wholesale* instead of merging: Ray rejects a ``runtime_env`` naming
more than one of ``pip``/``uv``/``conda``, so a partial merge could assemble an invalid env
out of two individually valid ones.

1. ``runtime { ray_runtime_env: '{"pip": [...]}' }``: per-task escape hatch, spelled like the
   existing ``ray_resources`` key.
2. ``[ray] image_env_map``: maps a ``runtime.docker`` value to a ``runtime_env``. Honours the
   image the WDL declares without editing the WDL.
3. Derived from the manifest: scan the task's command for executables, look each up in the
   toolchain, emit the wheels that provide them.
4. Nothing. The task then behaves exactly as it does under ``none``, which is the right
   fallback: the tool may well be in the cluster image already.

Two notes on the emitted shape, both of which cost real debugging time to find:

* ``pip_install_options`` belongs *inside* the ``pip`` dict. As a sibling ``runtime_env`` key it
  is accepted and silently ignored, and the install then goes to the network instead of the
  wheel directory. Same for ``uv_pip_install_options`` inside ``uv``.
* ``virtualenv`` (and ``pip``) must be importable in every node's base Python. Both the ``pip``
  and ``uv`` plugins require it and then materialize a virtualenv, cloning the base environment
  when the driver already sits in one, so a base env without ``pip`` yields a clone without
  ``pip``. The stock ``anyscale/ray`` image satisfies both.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Prefix of every tool distribution name. Must agree with the wheel names
#: ``tools/build_wheels.sh`` emits (``wdl-on-ray-tools-<name>``); the manifest is
#: deliberately readable by independent consumers, so the prefix is stated here
#: rather than imported from the build tooling.
DIST_PREFIX = "wdl-on-ray-tools"

#: ``wdl_on_ray/envs.py`` -> template root -> the toolchain manifest. Correct for the template
#: directory a workspace clones and for the uploaded ``working_dir`` a Ray job ships, and simply
#: absent when the package is installed as a wheel on its own (as it is inside this template's
#: image), in which case set ``[ray] tool_manifest``.
#:
#: Only ``native`` reads this. Under ``none``, what this template runs, the tools come from
#: the image, so a manifest that doesn't resolve here costs nothing.
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "tools" / "manifest.toml"

#: Names supplied by any plausible base environment: shell keywords, builtins and coreutils.
#: Maintained by hand, and stable in a way tool versions never were; an entry is added when a
#: task starts using a new *shell* utility, not when a tool is upgraded. Over-inclusion here
#: costs a missed warning; under-inclusion costs a spurious one. Neither can fail a run, since
#: an unresolved name only means "no wheel for this", and the base environment may supply it.
BASE_ENVIRONMENT = frozenset(
    """
    awk basename bash cat cd chmod cksum comm cp cut date df dirname do done du echo elif else
    env esac eval exit export expr fi find for grep gunzip gzip head if join local ln ls
    mkdir mktemp mv nproc paste printf pwd read return rev rm sed seq set sh shift sleep sort
    source split tail tar tee test then touch tr true uname uniq unset wc while xargs zcat
    """.split()
)

#: The leading word of a command position: start of a segment, after a pipe, ``&&`` or ``;``.
_LEADING_WORD = re.compile(r"([A-Za-z_][A-Za-z0-9_.-]*)")


def invoked_executables(command: str) -> set[str]:
    """Every name ``command`` runs in command position.

    Deliberately a heuristic and not a shell parse, and two filters do the heavy lifting or
    the result is dominated by things that merely look like commands:

    ``continuations``
        canu and flye are invoked across many backslash-continued lines whose arguments
        (``genomeSize=...``, ``-j ...``) each start a line. A line continued after a *pipe* is
        the exception and does start a command, which is how ``CallAssemblyVariants`` reaches
        ``paftools.js``.
    ``assignments``
        ``word=`` is a shell assignment or a canu-style parameter, never a command.

    The input is the *rendered* command miniwdl hands the backend, so there are no
    ``~{}`` placeholders to confuse it.
    """
    found: set[str] = set()
    skip_next = False
    for line in command.splitlines():
        stripped = line.strip()
        skipping, skip_next = (
            skip_next,
            stripped.endswith("\\") and not stripped[:-1].rstrip().endswith("|"),
        )
        if not stripped or stripped.startswith("#") or skipping:
            continue
        for raw in re.split(r"\||&&|;|\$\(", stripped):
            segment = raw.strip()
            match = _LEADING_WORD.match(segment)
            if match and not segment[match.end() :].startswith("="):
                found.add(match.group(1))
    return found


#: The executable miniwdl's own input downloader shells out to, per URI scheme.
#:
#: These are not the pipeline's tools and are not in the manifest, so nothing here declares or
#: checks them, and under ``native`` there is no container to supply them either. miniwdl
#: implements each downloader as a synthesised WDL *task* carrying a ``docker`` image, on the
#: assumption the binary comes from that image; with no container it has to already be on the
#: worker. Verified against ``WDL.runtime.download._downloader``:
#:
#:     s3  -> awscli_downloader   `aws s3 cp`
#:     gs  -> gsutil_downloader   `gsutil -q cp`
#:     http(s)/ftp -> aria2c_downloader  `aria2c`
#:
#: The stock ``anyscale/ray`` image happens to ship ``aws`` at /usr/local/bin/aws, which is the
#: only reason s3:// inputs localize at all in this mode. `gsutil` and `aria2c` are not there.
DOWNLOADER_EXECUTABLES = {
    "s3": "aws",
    "gs": "gsutil",
    "http": "aria2c",
    "https": "aria2c",
    "ftp": "aria2c",
}


def input_uri_schemes(values: object) -> set[str]:
    """Every URI scheme appearing in a nested inputs structure.

    Walks the JSON instead of typed WDL values, so it can run against an inputs file before
    anything is parsed or a workflow is loaded.
    """
    found: set[str] = set()
    if isinstance(values, str):
        head, sep, _ = values.partition("://")
        # RFC 3986: a scheme starts with a letter, then letters, digits, "+", "-" or ".".
        # `head.isalpha()` looks equivalent and is not: it rejects "s3", which is the one
        # scheme this exists for.
        if sep and head and head[0].isalpha() and all(c.isalnum() or c in "+-." for c in head):
            found.add(head.lower())
    elif isinstance(values, dict):
        for value in values.values():
            found |= input_uri_schemes(value)
    elif isinstance(values, (list, tuple)):
        for value in values:
            found |= input_uri_schemes(value)
    return found


def missing_downloaders(values: object) -> dict[str, str]:
    """``{scheme: executable}`` for schemes in the inputs whose downloader is not on PATH.

    The failure this prevents is a late one: miniwdl localizes a task's inputs *inside* the task,
    so a missing ``aws`` surfaces as exit 127 from a synthesised ``aws_s3_cp`` task several
    directories deep in a run, long after the workflow started and with nothing naming the
    scheme. Cheap to answer up front instead.
    """
    import shutil

    missing: dict[str, str] = {}
    for scheme in sorted(input_uri_schemes(values)):
        executable = DOWNLOADER_EXECUTABLES.get(scheme)
        if executable and not shutil.which(executable):
            missing[scheme] = executable
    return missing


def load_tool_index(manifest_path: str | Path | None = None) -> dict[str, tuple[str, str]]:
    """Map every executable the toolchain provides to ``(distribution, version)``.

    Reads only the two fields dispatch needs, ``version`` and the keys of ``provides``,
    because a task's environment does not care where a payload came from or what it hashed to.
    Returns an empty map when the manifest is absent, which turns manifest derivation off
    instead of failing the run.
    """
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    index: dict[str, tuple[str, str]] = {}
    for name, table in raw.get("tools", {}).items():
        # The manifest pin may carry upstream's build number as a PEP 440 local segment (the
        # JRE's is "21.0.12+8"), which the distribution drops; a local segment would have to
        # be percent-encoded in the URLs these wheels are referenced by. So pin against what
        # the wheel actually declares, which the manifest version is a prefix of.
        version = str(table.get("version", "")).split("+", 1)[0]
        for provided in table.get("provides", {}):
            index[str(provided)] = (f"{DIST_PREFIX}-{name}", version)
    return index


@dataclass(frozen=True)
class Resolved:
    """The environment for one task, plus enough context to explain it in a log."""

    runtime_env: dict[str, Any] = field(default_factory=dict)
    source: str = "none"
    """Which precedence rule produced this: ``runtime``, ``image_env_map``, ``manifest`` or
    ``none`` for the package modes; ``task_image_map`` or ``cluster-image`` for
    ``container_runtime = ray``."""
    requirements: tuple[str, ...] = ()
    """Distributions requested, when derived from the manifest."""
    unresolved: tuple[str, ...] = ()
    """Names the command invokes that neither the base environment nor the manifest accounts
    for. Reported as a warning: the cluster image may still supply them, and an unpackaged
    tool must not be a hard failure."""

    def describe(self) -> dict[str, Any]:
        """Log-friendly summary, in the shape :func:`RayRequest.describe` uses."""
        out: dict[str, Any] = {"env_source": self.source}
        if self.requirements:
            out["requirements"] = list(self.requirements)
        if self.unresolved:
            out["unprovided_commands"] = list(self.unresolved)
        # The image is the single most useful thing to see in a dispatch log under
        # `container_runtime = ray`: it is the answer to "what actually ran".
        if "image_uri" in self.runtime_env:
            out["image_uri"] = self.runtime_env["image_uri"]
        if not self.runtime_env:
            out["runtime_env"] = "(none)"
        return out


def _install_options(installer: str, wheel_dir: str, *, offline: bool) -> dict[str, Any]:
    """The installer-specific half of the emitted env.

    ``--find-links`` points at the wheelhouse; a package index stays available alongside it,
    which matters more than it looks. A ``kind = pypi`` tool (``quast`` is one) ships a wheel
    that is nothing but a dependency edge onto real distributions on PyPI, so resolving with
    ``--no-index`` would fail for it while succeeding for every tool whose payload
    ``tools/build_wheels.sh`` builds.

    ``offline`` adds ``--no-index`` for an air-gapped cluster. It is opt-in because it only
    works when every transitive dependency has been vendored into the wheelhouse too.

    Ray's own defaults are restated because supplying the option list replaces them wholesale
    instead of extending them.
    """
    if installer == "uv":
        options = ["--no-cache"]
    else:
        options = ["--disable-pip-version-check", "--no-cache-dir"]
    if offline:
        options.append("--no-index")
    if wheel_dir:
        options += ["--find-links", wheel_dir]
    key = "uv_pip_install_options" if installer == "uv" else "pip_install_options"
    return {key: options}


def build_env(
    requirements: list[str],
    *,
    installer: str = "pip",
    wheel_dir: str = "",
    offline: bool = False,
) -> dict[str, Any]:
    """Wrap a requirement list in the dict form of the ``pip`` or ``uv`` field."""
    field_name = "uv" if installer == "uv" else "pip"
    options = _install_options(installer, wheel_dir, offline=offline)
    return {field_name: {"packages": requirements, **options}}


#: ``runtime_env`` keys Ray refuses to accept alongside ``image_uri``. The image is
#: expected to be self-contained, so a package or working-directory field would be
#: describing an environment the image already fixed.
IMAGE_INCOMPATIBLE_KEYS = ("pip", "uv", "conda", "working_dir", "py_modules", "py_executable")


def validate_runtime_env(runtime_env: dict[str, Any]) -> None:
    """Reject ``runtime_env`` shapes Ray will refuse, before a task is dispatched.

    Ray raises on these too, but a task that fails at scheduling time inside a
    workflow surfaces as a failed WDL task with a Ray traceback attached, several
    layers away from the config line that caused it. Checking here means the message
    names the key and the fix.
    """
    if not runtime_env:
        return
    if "image_uri" in runtime_env:
        clashing = [k for k in IMAGE_INCOMPATIBLE_KEYS if k in runtime_env]
        if clashing:
            raise ValueError(
                "a Ray runtime_env cannot combine 'image_uri' with "
                + ", ".join(repr(k) for k in clashing)
                + ". An image_uri environment must be self-contained; bake those"
                " dependencies into the image instead. 'env_vars' is allowed."
            )
    if "pip" in runtime_env and "conda" in runtime_env:
        raise ValueError("a Ray runtime_env cannot name both 'pip' and 'conda'")


def resolve_image(
    runtime_values: dict[str, Any],
    *,
    task_image_map: dict[str, str] | None = None,
    fallback: str = "error",
) -> Resolved:
    """Decide the per-task image ``runtime_env`` for ``container_runtime = ray``.

    Precedence, highest first:

    1. ``runtime { ray_runtime_env: '{"image_uri": ...}' }``: the per-task escape
       hatch, same key the package modes use.
    2. ``[ray] task_image_map`` keyed on ``runtime.docker``, then on ``"*"``.
    3. ``fallback``: ``cluster`` runs the task in the cluster image with no
       ``image_uri``; ``error`` refuses.

    :param runtime_values: miniwdl's normalized ``runtime {}`` values.
    :param task_image_map: ``runtime.docker`` value -> image URI Ray should run.
    :param fallback: ``error`` or ``cluster``; see
        :attr:`wdl_on_ray.config.RayConfig.task_image_fallback`.
    """
    explicit = runtime_values.get("ray_runtime_env")
    if explicit not in (None, ""):
        parsed = json.loads(explicit) if isinstance(explicit, str) else explicit
        if not isinstance(parsed, dict):
            raise ValueError(f"runtime.ray_runtime_env must be a JSON object, got {explicit!r}")
        validate_runtime_env(parsed)
        return Resolved(runtime_env=parsed, source="runtime")

    image = str(runtime_values.get("docker", "") or "")
    mapping = task_image_map or {}
    mapped = mapping.get(image) or mapping.get("*")

    if not mapped:
        if fallback == "cluster":
            # Deliberate, configured, and logged as such: this task runs in whatever
            # the cluster booted with, exactly as it would under `none`.
            return Resolved(source="cluster-image")
        raise ValueError(
            f"container_runtime=ray has no image for a task declaring docker={image!r}. "
            "Add it to [ray] task_image_map (or add a '*' entry), or set "
            "[ray] task_image_fallback = cluster to run unmapped tasks in the cluster "
            "image. Refusing by default because silently falling back would give this "
            "task the advisory-image-tag behaviour of container_runtime=none, which is "
            "the thing this runtime exists to avoid."
        )

    env = {"image_uri": mapped}
    validate_runtime_env(env)
    return Resolved(runtime_env=env, source="task_image_map")


def resolve(
    runtime_values: dict[str, Any],
    command: str,
    *,
    wheel_dir: str = "",
    image_env_map: dict[str, Any] | None = None,
    installer: str = "pip",
    manifest_path: str | Path | None = None,
    extra_requirements: tuple[str, ...] = (),
    offline: bool = False,
) -> Resolved:
    """Decide the Ray ``runtime_env`` for one WDL task.

    :param runtime_values: miniwdl's normalized ``runtime {}`` values.
    :param command: the *rendered* task command, scanned for the executables it invokes.
    :param wheel_dir: where the tool wheels are staged; empty means "resolve from an index".
    :param image_env_map: ``runtime.docker`` value -> ``runtime_env``.
    :param installer: ``pip`` or ``uv``.
    :param offline: resolve with ``--no-index``, for a cluster with no egress. Requires every
        transitive dependency to be in ``wheel_dir`` as well.
    :param manifest_path: override for the toolchain manifest location.
    :param extra_requirements: distributions added to every derived environment.
    """
    explicit = runtime_values.get("ray_runtime_env")
    if explicit not in (None, ""):
        parsed = json.loads(explicit) if isinstance(explicit, str) else explicit
        if not isinstance(parsed, dict):
            raise ValueError(f"runtime.ray_runtime_env must be a JSON object, got {explicit!r}")
        validate_runtime_env(parsed)
        return Resolved(runtime_env=parsed, source="runtime")

    image = str(runtime_values.get("docker", "") or "")
    mapped = (image_env_map or {}).get(image)
    if mapped:
        if not isinstance(mapped, dict):
            raise ValueError(
                f"[ray] image_env_map entry for {image!r} must be a JSON object, got {mapped!r}"
            )
        validate_runtime_env(mapped)
        return Resolved(runtime_env=dict(mapped), source="image_env_map")

    index = load_tool_index(manifest_path)
    invoked = invoked_executables(command)
    candidates = sorted(invoked - BASE_ENVIRONMENT)

    requirements: list[str] = []
    unresolved: list[str] = []
    for name in candidates:
        entry = index.get(name)
        if entry is None:
            unresolved.append(name)
            continue
        dist, version = entry
        pinned = f"{dist}=={version}" if version else dist
        if pinned not in requirements:
            requirements.append(pinned)

    for extra in extra_requirements:
        if extra not in requirements:
            requirements.append(extra)

    if not requirements:
        return Resolved(source="none", unresolved=tuple(unresolved))

    return Resolved(
        runtime_env=build_env(
            requirements, installer=installer, wheel_dir=wheel_dir, offline=offline
        ),
        source="manifest",
        requirements=tuple(requirements),
        unresolved=tuple(unresolved),
    )
