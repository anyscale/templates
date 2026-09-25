"""Translate Nextflow process directives into a Ray resource request.

The mapping itself is short, because Nextflow's resource vocabulary and Ray's
overlap almost exactly:

===========================================  ==========================================
Nextflow                                     Ray
===========================================  ==========================================
``cpus 12``                                  ``num_cpus=12``
``memory 36.GB``                             ``memory=36<<30``
``accelerator 1, type: 'nvidia-l4'``         ``num_gpus=1, accelerator_type='L4'``
``ext.ray_resources = '{"nvme":1}'``         ``resources={'nvme': 1}``
``ext.image = '<uri>'``                      ``runtime_env={'image_uri': '<uri>'}``
``time 8.h``                                 nothing -- see below
===========================================  ==========================================

Two of those rows carry the interesting decisions.

``time`` has no Ray equivalent. A grid scheduler enforces a wall-clock limit and
kills the job; Ray does not. So the directive is parsed, recorded as
``time_minutes_ignored`` in the request the daemon logs and writes beside the
task, and otherwise ignored, and the README says so.

``memory`` is where a Nextflow pipeline meets an autoscaling cluster, and it is
the one place this module refuses to be quiet. On Ray, a request no node can
satisfy waits indefinitely instead of failing: the autoscaler finds no instance
type that fits, the task stays pending, and Nextflow gets nothing back.
nf-core's ``base.config`` ships ``process_high_memory`` at 200 GB and scales every
request by ``task.attempt``, so this happens in practice. So
:func:`check_schedulable` fails fast with the two numbers and the two files the
reader can change. Set ``NF_RAY_CLAMP_RESOURCES=1`` to clamp to the largest node
instead, which is occasionally what you want for a smoke test and never what you
want for a real run -- a tool told it has 200 GB and given 60 will be killed by
the kernel, some way into the task, with a less obvious message than this one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from nf_ray.directives import Directives

#: Accelerator spellings seen in Nextflow pipelines, mapped onto Ray
#: ``accelerator_type`` values. The Kubernetes device-plugin form
#: (``nvidia.com/gpu``) names no model and so maps to "any GPU", like a bare
#: ``accelerator 1``; GCE-style names map to Ray's. Unrecognized values pass
#: through unchanged, so Ray's own spellings ("A10G", "L40S", ...) work directly in
#: a `type:` field.
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

#: Values that mean "a GPU" without naming a model. Passing any of these through
#: as an `accelerator_type` would make the task unschedulable on every node.
_ANY_GPU = frozenset({"nvidia.com/gpu", "gpu", "amd.com/gpu", "nvidia", "true", "1"})

MB = 1 << 20


class Unschedulable(RuntimeError):
    """A request no node in the cluster can satisfy.

    Raised at submit time rather than left to pend, because Nextflow gets nothing
    back from a Ray task that never schedules.
    """


@dataclass(frozen=True)
class NodeLimits:
    """The largest single node the cluster can currently offer.

    One task is one process on one node, so the ceiling that matters is the
    largest *node*, never the cluster total: a 4-node cluster with 32 GB each
    cannot run a 64 GB task. :mod:`nf_ray.daemon` builds it from the declared
    ``NF_RAY_MAX_NODE_*`` values when they are set, so a cold cluster does not
    reject a request the autoscaler can satisfy by starting a node, and otherwise
    from the largest node ``ray.nodes()`` has shown so far.
    """

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
    """Parsed from Nextflow's ``time`` directive. Not enforced: Ray has no
    wall-clock limit. Carried so the request recorded beside the task shows it was
    ignored."""

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
    """Parse the ``ext.ray_resources`` escape hatch."""
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
    """Build the Ray request for one Nextflow task.

    :param directives: the parsed ``#RAY`` header.
    :param extra_resources: custom resources demanded of every task, from
        ``ray.extraResources`` (``NF_RAY_EXTRA_RESOURCES``). Useful for pinning a
        whole pipeline onto a labelled node group.
    :param default_accelerator_type: applied to GPU tasks that name no model, such
        as a bare ``accelerator 1`` or ``nvidia.com/gpu``.
    """
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
    """Reject -- or, if *clamp*, shrink -- a request no single node can satisfy.

    Returns the request to submit. Raises :class:`Unschedulable` otherwise, with
    both numbers and both files a reader can act on.
    """
    if not limits.known():
        # No node information yet (a cluster with zero workers alive). Submitting
        # is the right call: the autoscaler is what decides, and a wrong refusal
        # here would break the cold-start case entirely.
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
