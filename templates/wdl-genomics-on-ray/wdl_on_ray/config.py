"""Typed access to the ``[ray]`` section of miniwdl's configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from WDL.runtime.config import Loader

SECTION = "ray"

CONTAINER_RUNTIMES = (
    "auto",
    "podman",
    "docker",
    "apptainer",
    "singularity",
    "none",
    "native",
    "ray",
)

ENV_INSTALLERS = ("pip", "uv")

#: For a task not in ``task_image_map``: ``error`` refuses it, ``cluster`` runs the cluster image.
TASK_IMAGE_FALLBACKS = ("error", "cluster")

#: Where a task's command runs: a Ray task per WDL task, or in this process.
DISPATCH_MODES = ("ray", "inprocess")

#: How to derive the per-task ceiling miniwdl clamps ``runtime.cpu``/``runtime.memory`` to.
LIMIT_SOURCES = ("max_node", "cluster", "local")


def _raw(cfg: Loader, key: str, default: str) -> str:
    # miniwdl reads a section it does not know only with a fallback on every read.
    return cfg.get(SECTION, key, default=default).strip()


def _bool(cfg: Loader, key: str, default: bool) -> bool:
    value = _raw(cfg, key, "true" if default else "false").lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"[{SECTION}] {key}: expected a boolean, got {value!r}")


def _int(cfg: Loader, key: str, default: int) -> int:
    return int(_raw(cfg, key, str(default)))


def _list(cfg: Loader, key: str, default: list[str]) -> list[str]:
    value = _raw(cfg, key, "")
    if not value:
        return list(default)
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(f"[{SECTION}] {key}: expected a JSON list, got {value!r}")
        return [str(item) for item in parsed]
    return value.split()


def _dict(cfg: Loader, key: str, default: dict[str, Any]) -> dict[str, Any]:
    value = _raw(cfg, key, "")
    if not value:
        return dict(default)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"[{SECTION}] {key}: expected a JSON object, got {value!r}")
    return parsed


def _choice(cfg: Loader, key: str, default: str, allowed: tuple[str, ...]) -> str:
    value = _raw(cfg, key, default).lower()
    if value not in allowed:
        raise ValueError(f"[{SECTION}] {key}: expected one of {', '.join(allowed)}, got {value!r}")
    return value


@dataclass(frozen=True)
class RayConfig:
    """Resolved ``[ray]`` settings."""

    address: str
    namespace: str
    container_runtime: str
    #: e.g. ``["sudo", "podman"]``; empty means the runtime's own default.
    container_exe: list[str]
    limit_source: str
    #: 0 derives it. Set it to the worker shape when workers may not be up at startup.
    max_cpu: int
    max_memory_bytes: int
    #: Request ``runtime.memory`` from Ray, so it does not pack a node past its RAM.
    reserve_memory: bool
    #: ``DEFAULT`` or ``SPREAD``.
    scheduling_strategy: str
    #: 0: miniwdl's own retries reset the working directory, where Ray's would reuse a dirty one.
    task_max_retries: int
    #: Demanded by every task, e.g. ``{"wdl_node": 1}`` to confine WDL work to a node group.
    extra_resources: dict[str, float]
    extra_container_args: list[str]
    #: For GPU tasks whose WDL names none.
    accelerator_type: str
    image_pull_timeout: int
    #: Node-local, so N tasks landing on one node at once pull once.
    pull_lock_dir: str
    #: Tiny image that hands rootless-container outputs back to the invoking user.
    chown_image: str
    sif_cache_dir: str
    #: If set, ``runtime.disks`` is also demanded as this custom resource; else it is only logged.
    disk_resource_name: str
    dispatch: str
    #: For ``native``: a shared directory or a ``--find-links`` index; empty resolves from an index.
    tool_wheel_dir: str
    #: ``runtime.docker`` -> ``runtime_env`` for ``native``; wins over deriving it from the command.
    image_env_map: dict[str, Any]
    #: ``--no-index``; a ``kind = pypi`` tool then needs its dependencies vendored too.
    env_offline: bool
    env_installer: str
    tool_manifest: str
    #: Added to every environment ``native`` derives, not to explicitly named ones.
    env_extra_requirements: list[str]
    #: ``runtime.docker`` -> image URI for ``container_runtime = ray``; ``"*"`` matches the rest.
    task_image_map: dict[str, str]
    task_image_fallback: str


def load(cfg: Loader) -> RayConfig:
    """Read the ``[ray]`` section, applying defaults and validating values."""
    return RayConfig(
        address=_raw(cfg, "address", "auto"),
        namespace=_raw(cfg, "namespace", "wdl-on-ray"),
        container_runtime=_choice(cfg, "container_runtime", "auto", CONTAINER_RUNTIMES),
        container_exe=_list(cfg, "container_exe", []),
        limit_source=_choice(cfg, "limit_source", "max_node", LIMIT_SOURCES),
        max_cpu=_int(cfg, "max_cpu", 0),
        max_memory_bytes=_int(cfg, "max_memory_bytes", 0),
        reserve_memory=_bool(cfg, "reserve_memory", True),
        scheduling_strategy=_raw(cfg, "scheduling_strategy", "DEFAULT"),
        task_max_retries=_int(cfg, "task_max_retries", 0),
        extra_resources={str(k): float(v) for k, v in _dict(cfg, "extra_resources", {}).items()},
        extra_container_args=_list(cfg, "extra_container_args", []),
        accelerator_type=_raw(cfg, "accelerator_type", ""),
        image_pull_timeout=_int(cfg, "image_pull_timeout", 3600),
        pull_lock_dir=_raw(cfg, "pull_lock_dir", "/tmp/wdl-on-ray/pull-locks"),
        chown_image=_raw(cfg, "chown_image", "docker.io/library/alpine:3"),
        sif_cache_dir=_raw(cfg, "sif_cache_dir", "/tmp/wdl-on-ray/sif"),
        disk_resource_name=_raw(cfg, "disk_resource_name", ""),
        dispatch=_choice(cfg, "dispatch", "ray", DISPATCH_MODES),
        tool_wheel_dir=_raw(cfg, "tool_wheel_dir", ""),
        image_env_map=_dict(cfg, "image_env_map", {}),
        env_offline=_bool(cfg, "env_offline", False),
        env_installer=_choice(cfg, "env_installer", "pip", ENV_INSTALLERS),
        tool_manifest=_raw(cfg, "tool_manifest", ""),
        env_extra_requirements=_list(cfg, "env_extra_requirements", []),
        task_image_map={str(k): str(v) for k, v in _dict(cfg, "task_image_map", {}).items()},
        task_image_fallback=_choice(cfg, "task_image_fallback", "error", TASK_IMAGE_FALLBACKS),
    )
