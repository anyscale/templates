"""Parse the ``#RAY`` header the plugin writes into ``.command.run``, like SLURM's ``#SBATCH``."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field

_HEADER_RE = re.compile(r"^\s*#RAY\s+(?P<key>-{1,2}[A-Za-z][\w-]*)\s*(?P<value>.*?)\s*$")

# Stop at the script body, so a `#RAY` echoed later (a heredoc, a script: block) is not read.
_BODY_RE = re.compile(r"^\s*[^#\s]")

_MAX_HEADER_LINES = 200


class DirectiveError(ValueError):
    """A ``#RAY`` header line could not be understood."""


@dataclass(frozen=True)
class Directives:
    """The parsed ``#RAY`` header of one job script."""

    name: str = ""
    cpus: float = 1.0
    memory_mb: int = 0
    gpus: float = 0.0
    accelerator: str = ""
    resources: str = ""
    image: str = ""
    runtime: str = ""
    time_minutes: int = 0
    extra: dict[str, str] = field(default_factory=dict)


def _as_float(key: str, value: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise DirectiveError(f"#RAY {key}: expected a number, got {value!r}") from None


def _as_int(key: str, value: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise DirectiveError(f"#RAY {key}: expected an integer, got {value!r}") from None


def scan(lines: Iterable[str]) -> dict[str, str]:
    """Collect ``#RAY`` key/value pairs from the head of a job script, stopping at the body."""
    seen: dict[str, str] = {}
    for lineno, line in enumerate(lines):
        if lineno >= _MAX_HEADER_LINES:
            break
        if _BODY_RE.match(line):
            break
        match = _HEADER_RE.match(line)
        if match:
            seen[match.group("key").lstrip("-")] = match.group("value")
    return seen


def _build(seen: dict[str, str]) -> Directives:
    known = {
        "name",
        "cpus",
        "memory",
        "gpus",
        "accelerator",
        "resources",
        "image",
        "runtime",
        "time",
    }
    return Directives(
        name=seen.get("name", ""),
        cpus=_as_float("-cpus", seen["cpus"]) if seen.get("cpus") else 1.0,
        memory_mb=_as_int("-memory", seen["memory"]) if seen.get("memory") else 0,
        gpus=_as_float("-gpus", seen["gpus"]) if seen.get("gpus") else 0.0,
        accelerator=seen.get("accelerator", ""),
        resources=seen.get("resources", ""),
        image=seen.get("image", ""),
        runtime=seen.get("runtime", ""),
        time_minutes=_as_int("-time", seen["time"]) if seen.get("time") else 0,
        extra={k: v for k, v in seen.items() if k not in known},
    )


def parse_header(text: str) -> Directives:
    """Parse the ``#RAY`` header block out of a job script's text."""
    return _build(scan(text.splitlines()))


def parse_script(path: str) -> Directives:
    """Parse the ``#RAY`` header of the job script at *path*."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return _build(scan(handle))


def render_command(script: str) -> list[str]:
    """The argv that runs a Nextflow job script."""
    return ["/bin/bash", "-ue", script]


def format_header(directives: Directives) -> str:
    """Render *directives* back to header text, the inverse of :func:`parse_header`."""
    lines = []
    for key, value in (
        ("name", directives.name),
        ("cpus", directives.cpus),
        ("memory", directives.memory_mb),
        ("gpus", directives.gpus),
        ("accelerator", directives.accelerator),
        ("resources", directives.resources),
        ("image", directives.image),
        ("runtime", directives.runtime),
        ("time", directives.time_minutes),
    ):
        if value:
            lines.append(f"#RAY -{key} {shlex.quote(str(value)) if ' ' in str(value) else value}")
    return "\n".join(lines)
