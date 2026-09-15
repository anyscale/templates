"""Translate a WDL ``runtime {}`` section into a Ray resource request.

miniwdl has already normalized the ``runtime`` expressions into
``TaskContainer.runtime_values`` by the time we see them: ``cpu`` is an int,
``memory`` has become ``memory_reservation``/``memory_limit`` in bytes, and both
have been clamped to the ceiling our backend reported from
``detect_resource_limits``. What's left is deciding which Ray resources to
demand, which is where the WDL spec and the wider WDL ecosystem disagree enough
to need explicit handling:

* ``gpu`` is a *Boolean* in the WDL 1.1+ spec: it says "this task wants a GPU"
  without saying how many.
* ``gpuCount``/``gpuType``/``nvidiaDriverVersion`` are Cromwell's Google-backend
  extensions. The Broad long-read pipelines use them (``MedakaPolish`` asks for
  one ``nvidia-tesla-t4``), so honouring them is what makes real-world WDL run
  unchanged.
* ``disks`` is likewise a Cromwell-ism (``"local-disk 500 HDD"``). Ray has no
  disk resource, so we parse it for logging and optional node selection rather
  than silently dropping it.

Anything Ray-specific that WDL has no vocabulary for can be passed through with
a ``ray_resources`` entry in ``runtime {}`` (a JSON object).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

#: Cromwell/GCE accelerator names mapped onto Ray ``accelerator_type`` values.
#: Unrecognized names are passed through unchanged, so Ray's own spellings
#: ("A10G", "L40S", ...) work directly in a WDL ``gpuType``.
GPU_TYPE_TO_RAY_ACCELERATOR = {
    "nvidia-tesla-k80": "K80",
    "nvidia-tesla-p4": "P4",
    "nvidia-tesla-p100": "P100",
    "nvidia-tesla-t4": "T4",
    "nvidia-tesla-v100": "V100",
    "nvidia-tesla-a100": "A100",
    "nvidia-a100-80gb": "A100-80G",
    "nvidia-l4": "L4",
    "nvidia-h100-80gb": "H100",
    "nvidia-h100-mega-80gb": "H100",
    "nvidia-h200-141gb": "H200",
}

#: ``disks: "local-disk 500 HDD"`` and the ``"/mnt/foo 500 SSD"`` form.
_DISKS_RE = re.compile(r"(?:^|\s)(?P<size>\d+)\s+(?P<type>HDD|SSD|LOCAL)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RayRequest:
    """A resource request, shaped for ``ray.remote(...).options(**kwargs)``."""

    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory: int | None = None
    resources: dict[str, float] = field(default_factory=dict)
    accelerator_type: str | None = None
    disk_gb: int | None = None
    """Parsed from ``runtime.disks``. Not a Ray resource, carried for logging
    and for :data:`RayRequest.resources` when the operator opts in."""

    def options(self) -> dict[str, Any]:
        """The subset that ``ray.remote(...).options()`` accepts."""
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
        """Log-friendly summary (used in the miniwdl task log)."""
        out: dict[str, Any] = {"num_cpus": self.num_cpus}
        if self.num_gpus:
            out["num_gpus"] = self.num_gpus
        if self.memory:
            out["memory_bytes"] = self.memory
        if self.resources:
            out["resources"] = dict(self.resources)
        if self.accelerator_type:
            out["accelerator_type"] = self.accelerator_type
        if self.disk_gb:
            out["requested_disk_gb"] = self.disk_gb
        return out


def parse_disk_gb(disks: Any) -> int | None:
    """Extract the largest size (GB) from a Cromwell-style ``runtime.disks``.

    Returns ``None`` when the value doesn't parse, since ``disks`` is free-form
    in practice and a bad parse must not fail the run.
    """
    if disks is None:
        return None
    if isinstance(disks, (int, float)):
        return int(disks)
    sizes = [int(m.group("size")) for m in _DISKS_RE.finditer(str(disks))]
    return max(sizes) if sizes else None


def ray_accelerator_type(gpu_type: Any) -> str | None:
    """Map a WDL ``gpuType`` onto a Ray ``accelerator_type``."""
    if not gpu_type:
        return None
    name = str(gpu_type).strip()
    return GPU_TYPE_TO_RAY_ACCELERATOR.get(name.lower(), name) or None


def _custom_resources(runtime_values: dict[str, Any]) -> dict[str, float]:
    """Read a ``ray_resources`` escape hatch out of ``runtime {}``."""
    raw = runtime_values.get("ray_resources")
    if raw in (None, ""):
        return {}
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError(f"runtime.ray_resources must be a JSON object, got {raw!r}")
    return {str(k): float(v) for k, v in parsed.items()}


def build_request(
    runtime_values: dict[str, Any],
    *,
    reserve_memory: bool = True,
    extra_resources: dict[str, float] | None = None,
    default_accelerator_type: str = "",
    disk_resource_name: str = "",
) -> RayRequest:
    """Build the Ray request for one WDL task.

    :param runtime_values: miniwdl's normalized ``runtime {}`` values.
    :param reserve_memory: request ``runtime.memory`` as Ray's ``memory``
        resource, so Ray won't overcommit a node's RAM.
    :param extra_resources: custom resources demanded of every task.
    :param default_accelerator_type: used for GPU tasks that don't name a type.
    :param disk_resource_name: when set, ``runtime.disks`` is also demanded as
        this custom Ray resource, letting operators steer disk-hungry tasks onto
        node groups labelled with it.
    """
    # miniwdl omits `cpu` entirely when the task's runtime section does not set
    # it; WDL's default is one core.
    num_cpus = float(runtime_values.get("cpu", 1) or 1)

    memory: int | None = None
    if reserve_memory:
        reservation = int(runtime_values.get("memory_reservation", 0) or 0)
        memory = reservation or None

    # `gpu` (spec, Boolean) and `gpuCount` (Cromwell, Int) can both appear; take
    # whichever asks for more so neither is silently ignored.
    num_gpus = 0.0
    if runtime_values.get("gpu"):
        num_gpus = 1.0
    gpu_count = runtime_values.get("gpuCount")
    if gpu_count is not None:
        num_gpus = max(num_gpus, float(gpu_count))

    accelerator = ray_accelerator_type(
        runtime_values.get("gpuType") or runtime_values.get("acceleratorType")
    )
    if num_gpus and accelerator is None and default_accelerator_type:
        accelerator = default_accelerator_type
    if not num_gpus:
        # An accelerator_type without a GPU request is unschedulable noise.
        accelerator = None

    resources = dict(extra_resources or {})
    resources.update(_custom_resources(runtime_values))

    disk_gb = parse_disk_gb(runtime_values.get("disks"))
    if disk_gb and disk_resource_name:
        resources[disk_resource_name] = float(disk_gb)

    return RayRequest(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory=memory,
        resources=resources,
        accelerator_type=accelerator,
        disk_gb=disk_gb,
    )
