"""Translate a WDL ``runtime {}`` section (Cromwell extensions too) into a Ray resource request."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

#: Cromwell/GCE accelerator names -> Ray ``accelerator_type``; others pass through ("A10G", "L40S").
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
    #: From ``runtime.disks``: logged, and a resource only with ``disk_resource_name``.
    disk_gb: int | None = None

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
    """Largest size (GB) in a Cromwell-style ``runtime.disks``; None if it does not parse."""
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
    """Build the Ray request for one WDL task from miniwdl's normalized runtime values."""
    # miniwdl omits `cpu` when the task does not set it; WDL's default is one core.
    num_cpus = float(runtime_values.get("cpu", 1) or 1)

    memory: int | None = None
    if reserve_memory:
        reservation = int(runtime_values.get("memory_reservation", 0) or 0)
        memory = reservation or None

    # `gpu` (spec, Boolean) and `gpuCount` (Cromwell, Int) can both appear; take the larger.
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
