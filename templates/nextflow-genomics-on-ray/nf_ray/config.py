"""Typed configuration for the Ray side of the executor.

Every setting arrives as an ``NF_RAY_*`` environment variable. That is not a
compromise, it is the only channel that reaches all three places the settings
have to be visible: the Nextflow JVM (which spawns ``nf-ray``), the daemon (a
separate long-lived process), and the CLI (a fresh process per submit). A
Nextflow config scope would reach only the first.

The Groovy plugin forwards a ``ray { ... }`` scope into this namespace, so all
three of these are spellings of the same setting::

    ray { clampResources = true }              # nextflow.config
    env { NF_RAY_CLAMP_RESOURCES = 'true' }    # nextflow.config
    NF_RAY_CLAMP_RESOURCES=true nextflow run   # shell

Defaults worth their comments are commented at the field.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

#: Storage that every node sees at the same absolute path. Nextflow stages inputs
#: by path and a grid executor has no other way to move them, so a work directory
#: outside this list means the second process in the pipeline gets "no such file"
#: on a different node -- silently, and only once the cluster has more than one.
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

#: Where the daemon's socket goes by default. Node-local on purpose: the socket
#: only ever connects processes on the head node (Nextflow, the CLI it spawns, the
#: daemon), and the work directory is on NFS, where a unix socket may not be
#: supported at all and the lock file beside it relies on flock over the network.
#: A fixed /tmp rather than $TMPDIR, because a notebook that points TMPDIR at
#: shared storage would otherwise move the socket straight back onto it.
SOCKET_DIR = "/tmp"

# Defaults live here, once, and are referenced by both the dataclass field and
# `from_env`. Writing the literal in both places is how a default silently
# becomes two defaults: the dataclass says one thing, the env reader another,
# and whichever the test happens to exercise is the one that stays honest.
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
    """Unix socket the CLI uses to reach the daemon. Defaults to
    ``/tmp/nf-ray-<hash of work_dir>.sock`` (see :func:`default_socket_path`), so
    two concurrent pipelines in different work directories get their own daemon
    without coordinating, and the socket stays off NFS."""

    work_dir: str = ""
    """Nextflow's ``workDir``. Used to site the socket and to warn when it is not
    on shared storage."""

    clamp_resources: bool = False
    """Shrink an unsatisfiable request to the largest node instead of rejecting
    it. Off, because a tool told it has more memory than it gets dies partway
    through the task with a worse message. See :mod:`nf_ray.resources`."""

    task_max_retries: int = DEFAULT_TASK_MAX_RETRIES
    """Ray-level retries. Zero on purpose: Nextflow's ``maxRetries`` and
    ``errorStrategy`` are the pipeline's retry policy, and a Ray retry underneath
    them is invisible to Nextflow -- it would silently multiply the real attempt
    count and defeat ``task.attempt``-scaled resource requests."""

    scheduling_strategy: str = DEFAULT_SCHEDULING_STRATEGY
    """``SPREAD`` is useful for demonstrating placement and for I/O-bound fan-out;
    ``DEFAULT`` packs, which is what you want when the autoscaler is paying by
    the node."""

    default_accelerator: str = ""
    """Applied to GPU tasks that name no model. nf-core modules say
    ``nvidia.com/gpu``, which names none, so without this a GPU process would be
    scheduled onto any accelerator the cluster happens to have."""

    extra_resources: dict[str, float] = field(default_factory=dict)
    runtime_env: dict[str, Any] = field(default_factory=dict)

    image_map: dict[str, str] = field(default_factory=dict)
    """Declared image -> image actually launchable on this cluster. Ray's
    ``image_uri`` requires the image's Ray and Python to match the cluster's to
    the patch, so a pipeline's own ``ext.image`` usually cannot be used verbatim.
    ``"*"`` is honoured as a catch-all."""

    image_fallback: str = DEFAULT_IMAGE_FALLBACK
    """What to do when ``ext.image`` has no mapping: ``error`` or ``ignore``
    (run in the cluster's own image). ``error``, because silently ignoring a
    declared image runs the task against different software than it asked for."""

    poll_interval: float = DEFAULT_POLL_INTERVAL
    """How often the daemon reaps finished tasks. Nextflow polls status on its own
    ``queueStatInterval``; this only bounds how stale the answer can be."""

    idle_timeout: float = DEFAULT_IDLE_TIMEOUT
    """Shut the daemon down after this long with no client contact, so a killed
    Nextflow run cannot leave a Ray driver holding the cluster open."""

    placement_tsv: str = ""
    """Consolidated placement record. Defaults to
    ``<work_dir>/nf_ray_placement.tsv``."""

    log_level: str = "INFO"

    max_node_cpus: float = 0.0
    max_node_memory_gb: float = 0.0
    max_node_gpus: float = 0.0
    """The largest worker this cluster can *grow* to, declared rather than
    discovered.

    ``ray.nodes()`` only reports nodes that are already alive, so on a cluster
    whose big worker group sits at ``min_nodes: 0`` the observed ceiling is
    whatever happens to be running -- which would reject a 72 GB request that the
    autoscaler could satisfy in two minutes. The autoscaler's configured node
    types are not reachable through a public API, so the template's ``ray``
    profile states them instead, copied from the compute config:

        env {
          NF_RAY_MAX_NODE_CPUS      = '16'
          NF_RAY_MAX_NODE_MEMORY_GB = '80'
          NF_RAY_MAX_NODE_GPUS      = '1'
        }

    This is the same coupling the compute configs' own comments describe from the
    other side -- the instance type and the pipeline's resource request are one
    decision, not two -- so writing it down in both places is the point, not
    duplication. Left at zero, the ceiling falls back to the largest node seen
    alive so far, and an all-zero ceiling disables the check."""

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
    """The daemon socket for *work_dir*: node-local, and short.

    Keyed by a hash of the absolute work directory, which keeps "one daemon per
    work directory" without putting the socket inside it. The hash also keeps the
    path at a fixed 33 characters, well inside the 108 bytes ``sun_path`` allows;
    ``<work_dir>/.nf-ray.sock`` grew with the work directory and failed to bind
    past that limit.
    """
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
    """The warning for a work directory no other node can read, or ``""``.

    Deliberately **not** gated on how many nodes are alive right now. A cluster
    whose worker group has ``min_nodes: 0`` looks single-node until the moment the
    autoscaler adds the second node, which is exactly the moment this becomes a
    problem -- so a check that stays quiet while the cluster is small is quiet
    precisely when it is needed.
    """
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
