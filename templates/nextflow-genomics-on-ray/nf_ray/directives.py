"""Read the ``#RAY`` header out of a Nextflow job script.

Nextflow's ``AbstractGridExecutor`` builds a job script (``.command.run``) and
prefixes it with one header line per directive, using the token the executor
declares in ``getHeaderToken()``. For SLURM that produces ``#SBATCH -c 4``; for
this executor it produces::

    #!/bin/bash
    #RAY -name nf-FASTP_HG002
    #RAY -cpus 6
    #RAY -memory 36864
    #RAY -gpus 1
    #RAY -accelerator nvidia-l4
    #RAY -resources {"nvme":1}

Parsing the header rather than accepting command-line flags is deliberate: it
keeps the submit call to ``nf-ray submit .command.run`` regardless of how many
directives a process declares, and it means the resource request is recorded in
the task's own work directory, where anyone debugging the run will look first.

One value per line. The value is the entire remainder of the line, so values
containing spaces survive -- but the Groovy side still emits JSON without
spaces, because ``AbstractGridExecutor`` pairs directive tokens up two at a time
and a bare space inside a value would split it across two header lines.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field

#: ``#RAY -key value`` -- leading whitespace tolerated, ``value`` optional so a
#: flag-style directive (``-preemptible``) parses to the empty string.
_HEADER_RE = re.compile(r"^\s*#RAY\s+(?P<key>-{1,2}[A-Za-z][\w-]*)\s*(?P<value>.*?)\s*$")

#: Stop scanning once the script body starts. Nextflow puts every header line in
#: the first block, so a `#RAY` string appearing later (in a heredoc, or in a
#: user's `script:` block that happens to echo one) must not be picked up.
_BODY_RE = re.compile(r"^\s*[^#\s]")

#: How many leading lines to scan before giving up on finding a header. Nextflow
#: emits its headers immediately after the shebang; this is slack, not a target.
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
    """Directives this version does not recognize, kept rather than dropped.
    Nothing acts on them yet."""


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
    """Collect ``#RAY`` key/value pairs from the head of a job script.

    The single place both bounds are enforced, so neither can drift from the
    other: stop at the first body line, and never scan more than
    :data:`_MAX_HEADER_LINES`. Takes an iterable so :func:`parse_script` can hand
    it a file object and read no more of the file than is actually scanned -- a
    job script embeds the process's whole ``script:`` block, which for a genomics
    tool invocation runs to hundreds of lines.
    """
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
    """The argv that runs a Nextflow job script.

    Nextflow's wrapper redirects the task's own stdout/stderr to
    ``.command.out``/``.command.err`` and writes ``.exitcode`` itself, so there
    is nothing to arrange here beyond invoking it under bash.
    """
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
