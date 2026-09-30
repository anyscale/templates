"""Compile a WDL task's declared environment into a Ray ``runtime_env``."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Must match the wheel names ``tools/build_wheels.sh`` emits.
DIST_PREFIX = "wdl-on-ray-tools"

#: Absent when the package is installed as a wheel on its own; set ``[ray] tool_manifest`` then.
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "tools" / "manifest.toml"

#: Shell keywords, builtins and coreutils that any base environment supplies. Hand-maintained.
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
    """Every name the rendered ``command`` runs in command position. A heuristic, not a parse."""
    found: set[str] = set()
    skip_next = False
    for line in command.splitlines():
        stripped = line.strip()
        # A continuation line holds arguments, except after a pipe, where a command starts.
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


#: What miniwdl's input downloaders (``WDL.runtime.download``) shell out to, per URI scheme.
#: Without a container these must already be on the worker.
DOWNLOADER_EXECUTABLES = {
    "s3": "aws",
    "gs": "gsutil",
    "http": "aria2c",
    "https": "aria2c",
    "ftp": "aria2c",
}


def input_uri_schemes(values: object) -> set[str]:
    """Every URI scheme in a nested inputs structure, read from raw JSON before any parsing."""
    found: set[str] = set()
    if isinstance(values, str):
        head, sep, _ = values.partition("://")
        # RFC 3986 scheme. Not head.isalpha(): that rejects "s3".
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
    """``{scheme: executable}`` for input schemes whose downloader is not on PATH."""
    import shutil

    missing: dict[str, str] = {}
    for scheme in sorted(input_uri_schemes(values)):
        executable = DOWNLOADER_EXECUTABLES.get(scheme)
        if executable and not shutil.which(executable):
            missing[scheme] = executable
    return missing


def load_tool_index(manifest_path: str | Path | None = None) -> dict[str, tuple[str, str]]:
    """Map each executable the manifest provides to ``(distribution, version)``; {} without one."""
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    index: dict[str, tuple[str, str]] = {}
    for name, table in raw.get("tools", {}).items():
        # Wheels drop a PEP 440 local segment (the JRE's "21.0.12+8"), so pin what they declare.
        version = str(table.get("version", "")).split("+", 1)[0]
        for provided in table.get("provides", {}):
            index[str(provided)] = (f"{DIST_PREFIX}-{name}", version)
    return index


@dataclass(frozen=True)
class Resolved:
    """The environment for one task, plus enough context to explain it in a log."""

    runtime_env: dict[str, Any] = field(default_factory=dict)
    #: The precedence rule that produced it, e.g. ``manifest`` or ``task_image_map``.
    source: str = "none"
    requirements: tuple[str, ...] = ()
    #: Invoked names that neither the base environment nor the manifest accounts for.
    unresolved: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"env_source": self.source}
        if self.requirements:
            out["requirements"] = list(self.requirements)
        if self.unresolved:
            out["unprovided_commands"] = list(self.unresolved)
        if "image_uri" in self.runtime_env:
            out["image_uri"] = self.runtime_env["image_uri"]
        if not self.runtime_env:
            out["runtime_env"] = "(none)"
        return out


def _install_options(installer: str, wheel_dir: str, *, offline: bool) -> dict[str, Any]:
    # Ray's defaults are restated: a supplied option list replaces them rather than extending them.
    if installer == "uv":
        options = ["--no-cache"]
    else:
        options = ["--disable-pip-version-check", "--no-cache-dir"]
    # The index stays on unless offline: a kind = pypi wheel is only a dependency edge onto PyPI.
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
    # The install options go inside that dict; as a sibling runtime_env key Ray ignores them.
    field_name = "uv" if installer == "uv" else "pip"
    options = _install_options(installer, wheel_dir, offline=offline)
    return {field_name: {"packages": requirements, **options}}


#: ``runtime_env`` keys Ray refuses alongside ``image_uri``.
IMAGE_INCOMPATIBLE_KEYS = ("pip", "uv", "conda", "working_dir", "py_modules", "py_executable")


def validate_runtime_env(runtime_env: dict[str, Any]) -> None:
    """Reject ``runtime_env`` shapes Ray would refuse, before a task is dispatched."""
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
    """The per-task image ``runtime_env`` under ``container_runtime = ray``."""
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
    """Decide the Ray ``runtime_env`` for one WDL task."""
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
