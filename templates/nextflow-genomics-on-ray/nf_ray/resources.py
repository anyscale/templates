"""Translate Nextflow directives into a Ray resource request, and reject what no node can run.

On Ray an unsatisfiable request pends forever. ``time`` is recorded but not enforced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from nf_ray.directives import Directives

# GCE-style names -> Ray accelerator_type. Unknown values pass through, so Ray's names work too.
ACCELERATOR_ALIASES = {
    "nvidia-tesla-k80": "K80",
    "nvidia-tesla-p4": "P4",
    "nvidia-tesla-p100": "P100",
    "nvidia-tesla-t4": "T4",
    "nvidia-tesla-v100": "V100",
    "nvidia-tesla-a100": "A100",
    "nvidia-a100-80gb": "A100-80G",
    "nvidia-l4": "L4",
    "nvidia-l40s": "L40S",
    "nvidia-a10g": "A10G",
    "nvidia-h100-80gb": "H100",
    "nvidia-h100-mega-80gb": "H100",
    "nvidia-h200-141gb": "H200",
}

# "A GPU", no model. As an accelerator_type these would match no node.
_ANY_GPU = frozenset({"nvidia.com/gpu", "gpu", "amd.com/gpu", "nvidia", "true", "1"})

MB = 1 << 20


class Unschedulable(RuntimeError):
    """A request no node in the cluster can satisfy."""


@dataclass(frozen=True)
class NodeLimits:
    """The largest single node on offer; a task runs on one node, so the cluster total is moot."""

    cpus: float = 0.0
    memory_bytes: int = 0
    gpus: float = 0.0
    source: str = "unknown"

    def known(self) -> bool:
        return self.cpus > 0 or self.memory_bytes > 0


@dataclass(frozen=True)
class RayRequest:
    """A resource request, shaped for ``ray.remote(...).options(**kwargs)``."""

    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory: int | None = None
    resources: dict[str, float] = field(default_factory=dict)
    accelerator_type: str | None = None
    time_minutes: int = 0

    def options(self) -> dict[str, Any]:
        """The subset ``ray.remote(...).options()`` accepts."""
        opts: dict[str, Any] = {"num_cpus": self.num_cpus}
        if self.num_gpus:
            opts["num_gpus"] = self.num_gpus
        if self.memory:
            opts["memory"] = self.memory
        if self.resources:
            opts["resources"] = dict(self.resources)
        if self.accelerator_type:
            opts["accelerator_type"] = self.accelerator_type
        return opts

    def describe(self) -> dict[str, Any]:
        """Log-friendly summary, written next to the task in its work directory."""
        out: dict[str, Any] = {"num_cpus": self.num_cpus}
        if self.num_gpus:
            out["num_gpus"] = self.num_gpus
        if self.memory:
            out["memory_bytes"] = self.memory
        if self.resources:
            out["resources"] = dict(self.resources)
        if self.accelerator_type:
            out["accelerator_type"] = self.accelerator_type
        if self.time_minutes:
            out["time_minutes_ignored"] = self.time_minutes
        return out


def ray_accelerator_type(value: Any) -> str | None:
    """Map a Nextflow ``accelerator`` ``type:`` onto a Ray ``accelerator_type``."""
    if not value:
        return None
    name = str(value).strip()
    if name.lower() in _ANY_GPU:
        return None
    return ACCELERATOR_ALIASES.get(name.lower(), name) or None


def _custom_resources(raw: str) -> dict[str, float]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exn:
        raise ValueError(f"ext.ray_resources is not valid JSON: {raw!r} ({exn})") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"ext.ray_resources must be a JSON object, got {raw!r}")
    return {str(k): float(v) for k, v in parsed.items()}


def build_request(
    directives: Directives,
    *,
    extra_resources: dict[str, float] | None = None,
    default_accelerator_type: str = "",
) -> RayRequest:
    """Build the Ray request for one Nextflow task."""
    num_cpus = float(directives.cpus or 1)
    memory = directives.memory_mb * MB if directives.memory_mb else None

    num_gpus = float(directives.gpus or 0)
    accelerator = ray_accelerator_type(directives.accelerator)
    if num_gpus and accelerator is None and default_accelerator_type:
        accelerator = ray_accelerator_type(default_accelerator_type)
    if not num_gpus:
        # An accelerator_type with no GPU request is unschedulable noise.
        accelerator = None

    resources = dict(extra_resources or {})
    resources.update(_custom_resources(directives.resources))

    return RayRequest(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory=memory,
        resources=resources,
        accelerator_type=accelerator,
        time_minutes=directives.time_minutes,
    )


def _gib(value: float) -> str:
    return f"{value / (1 << 30):.1f} GiB"


def check_schedulable(
    request: RayRequest,
    limits: NodeLimits,
    *,
    process: str = "",
    clamp: bool = False,
) -> RayRequest:
    """Reject -- or, if *clamp*, shrink -- a request no single node can satisfy."""
    if not limits.known():
        # Nothing known yet (no worker alive): let the autoscaler decide.
        return request

    label = f"process {process!r}" if process else "this task"
    problems = []
    if request.num_cpus > limits.cpus:
        problems.append(
            f"cpus {request.num_cpus:g} > {limits.cpus:g} available on the largest node"
        )
    if request.memory and request.memory > limits.memory_bytes:
        problems.append(
            f"memory {_gib(request.memory)} > {_gib(limits.memory_bytes)} "
            "available on the largest node"
        )
    if request.num_gpus > limits.gpus:
        problems.append(
            f"gpus {request.num_gpus:g} > {limits.gpus:g} available on the largest node"
        )

    if not problems:
        return request

    if clamp:
        return RayRequest(
            num_cpus=min(request.num_cpus, limits.cpus) or 1.0,
            num_gpus=min(request.num_gpus, limits.gpus),
            memory=min(request.memory, limits.memory_bytes) if request.memory else None,
            resources=dict(request.resources),
            accelerator_type=request.accelerator_type if limits.gpus else None,
            time_minutes=request.time_minutes,
        )

    raise Unschedulable(
        f"{label} asks for more than any node in this cluster can provide:\n"
        + "".join(f"  - {p}\n" for p in problems)
        + f"  (node ceiling read from: {limits.source})\n"
        "On Ray an unsatisfiable request pends forever rather than failing, so it is\n"
        "rejected here instead. Fix it in one of two places:\n"
        "  - raise the instance type in configs/nextflow-genomics-on-ray/{aws,gce}.yaml\n"
        "  - lower the request for this process label in pipeline/conf/base.config\n"
        "Set NF_RAY_CLAMP_RESOURCES=1 to clamp to the largest node instead. That is\n"
        "fine for a smoke test and wrong for a real run: a tool told it has more\n"
        "memory than it gets is killed by the kernel partway through the task."
    )
