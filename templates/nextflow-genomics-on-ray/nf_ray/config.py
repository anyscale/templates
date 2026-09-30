"""Executor settings from ``NF_RAY_*`` env vars, the one channel that reaches Nextflow, the
daemon and every per-submit CLI. The plugin forwards a ``ray { ... }`` scope into them.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

SHARED_STORAGE_PREFIXES = (
    "/mnt/cluster_storage",
    "/mnt/shared_storage",
    "/mnt/user_storage",
    "/mnt/shared",
    "/mnt/nfs",
    "/mnt/efs",
    "/mnt/fsx",
)

_TRUE = frozenset({"1", "true", "yes", "on"})

# Node-local: unix sockets and flock are unreliable on NFS. Not $TMPDIR, which a notebook
# may point at shared storage.
SOCKET_DIR = "/tmp"

# Shared by the dataclass and from_env, so the two cannot drift.
DEFAULT_TASK_MAX_RETRIES = 0
DEFAULT_POLL_INTERVAL = 0.25
DEFAULT_IDLE_TIMEOUT = 3600.0
DEFAULT_SCHEDULING_STRATEGY = "DEFAULT"
DEFAULT_IMAGE_FALLBACK = "error"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"NF_RAY_{name}", default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    return raw.lower() in _TRUE if raw else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"NF_RAY_{name} must be an integer, got {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"NF_RAY_{name} must be a number, got {raw!r}") from None


def _env_json(name: str, default: Any) -> Any:
    raw = _env(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exn:
        raise ValueError(f"NF_RAY_{name} must be valid JSON, got {raw!r} ({exn})") from None


@dataclass(frozen=True)
class Config:
    """Resolved configuration for one pipeline run."""

    address: str = "auto"
    namespace: str = "nf-ray"
    socket_path: str = ""
    work_dir: str = ""
    # Off: a tool told it has more memory than it gets dies mid-task, with a worse message.
    clamp_resources: bool = False
    # Zero: Nextflow owns retries, and a hidden Ray retry would defeat task.attempt scaling.
    task_max_retries: int = DEFAULT_TASK_MAX_RETRIES
    scheduling_strategy: str = DEFAULT_SCHEDULING_STRATEGY
    default_accelerator: str = ""
    extra_resources: dict[str, float] = field(default_factory=dict)
    runtime_env: dict[str, Any] = field(default_factory=dict)
    # Declared image -> launchable image; "*" is a catch-all.
    image_map: dict[str, str] = field(default_factory=dict)
    image_fallback: str = DEFAULT_IMAGE_FALLBACK
    poll_interval: float = DEFAULT_POLL_INTERVAL
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT
    placement_tsv: str = ""
    log_level: str = "INFO"
    # Declared, not observed: ray.nodes() misses workers the autoscaler has yet to start.
    # Zero falls back to the largest node seen alive.
    max_node_cpus: float = 0.0
    max_node_memory_gb: float = 0.0
    max_node_gpus: float = 0.0

    @classmethod
    def from_env(cls, work_dir: str = "") -> Config:
        resolved_work_dir = work_dir or _env("WORK_DIR") or os.getcwd()
        socket_path = _env("SOCKET") or default_socket_path(resolved_work_dir)
        placement = _env("PLACEMENT_TSV") or os.path.join(
            resolved_work_dir, "nf_ray_placement.tsv"
        )

        extra = _env_json("EXTRA_RESOURCES", {}) or {}
        if not isinstance(extra, dict):
            raise ValueError("NF_RAY_EXTRA_RESOURCES must be a JSON object")

        runtime_env = _env_json("RUNTIME_ENV", {}) or {}
        if not isinstance(runtime_env, dict):
            raise ValueError("NF_RAY_RUNTIME_ENV must be a JSON object")

        image_map = _env_json("IMAGE_MAP", {}) or {}
        if not isinstance(image_map, dict):
            raise ValueError("NF_RAY_IMAGE_MAP must be a JSON object")

        fallback = (_env("IMAGE_FALLBACK") or DEFAULT_IMAGE_FALLBACK).lower()
        if fallback not in ("error", "ignore"):
            raise ValueError(
                f"NF_RAY_IMAGE_FALLBACK must be 'error' or 'ignore', got {fallback!r}"
            )

        strategy = (_env("SCHEDULING_STRATEGY") or DEFAULT_SCHEDULING_STRATEGY).upper()
        if strategy not in ("DEFAULT", "SPREAD"):
            raise ValueError(
                f"NF_RAY_SCHEDULING_STRATEGY must be 'DEFAULT' or 'SPREAD', got {strategy!r}"
            )

        return cls(
            address=_env("ADDRESS") or "auto",
            namespace=_env("NAMESPACE") or "nf-ray",
            socket_path=socket_path,
            work_dir=resolved_work_dir,
            clamp_resources=_env_bool("CLAMP_RESOURCES"),
            task_max_retries=_env_int("TASK_MAX_RETRIES", DEFAULT_TASK_MAX_RETRIES),
            scheduling_strategy=strategy,
            default_accelerator=_env("DEFAULT_ACCELERATOR"),
            extra_resources={str(k): float(v) for k, v in extra.items()},
            runtime_env=runtime_env,
            image_map={str(k): str(v) for k, v in image_map.items()},
            image_fallback=fallback,
            poll_interval=_env_float("POLL_INTERVAL", DEFAULT_POLL_INTERVAL),
            idle_timeout=_env_float("IDLE_TIMEOUT", DEFAULT_IDLE_TIMEOUT),
            placement_tsv=placement,
            log_level=(_env("LOG_LEVEL") or "INFO").upper(),
            max_node_cpus=_env_float("MAX_NODE_CPUS", 0.0),
            max_node_memory_gb=_env_float("MAX_NODE_MEMORY_GB", 0.0),
            max_node_gpus=_env_float("MAX_NODE_GPUS", 0.0),
        )


def default_socket_path(work_dir: str) -> str:
    """The daemon socket for *work_dir*: node-local, hashed to fit ``sun_path``'s 108 bytes."""
    digest = hashlib.sha256(os.path.abspath(work_dir).encode()).hexdigest()[:16]
    return os.path.join(SOCKET_DIR, f"nf-ray-{digest}.sock")


def is_shared_storage(path: str) -> bool:
    """Whether *path* is on storage every node sees at the same absolute path."""
    resolved = os.path.abspath(path)
    return any(
        resolved == prefix or resolved.startswith(prefix.rstrip("/") + "/")
        for prefix in SHARED_STORAGE_PREFIXES
    )


def shared_storage_warning(work_dir: str) -> str:
    """Warning for a workDir other nodes cannot read, or ``""``; not gated on current node count."""
    if is_shared_storage(work_dir):
        return ""
    return (
        f"workDir {work_dir!r} is not on shared storage.\n"
        "Nextflow stages inputs by absolute path and a grid executor has no other way to\n"
        "move them, so as soon as this cluster has a second node, tasks there will fail\n"
        "to find files the first node wrote -- and Nextflow will report it as a missing\n"
        "output, not as a storage problem.\n"
        f"Use one of: {', '.join(SHARED_STORAGE_PREFIXES[:3])}\n"
        "  workDir = '/mnt/cluster_storage/nf-work'   # nextflow.config"
    )
