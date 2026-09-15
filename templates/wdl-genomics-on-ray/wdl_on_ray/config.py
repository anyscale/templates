"""Typed access to the ``[ray]`` section of miniwdl's configuration.

miniwdl's :class:`WDL.runtime.config.Loader` merges, in decreasing priority:
explicit overrides, ``MINIWDL__<SECTION>__<KEY>`` environment variables, a
``miniwdl.cfg`` file, and the built-in defaults. Sections that miniwdl doesn't
know about, like ours, work fine as long as every read supplies a fallback,
which is what the helpers here do. So all of these are equivalent ways to pick
the container runtime:

    MINIWDL__RAY__CONTAINER_RUNTIME=podman miniwdl run ...
    miniwdl run --cfg my.cfg ...        # with [ray] container_runtime = podman
    wdl-on-ray run --container-runtime podman ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from WDL.runtime.config import Loader

SECTION = "ray"

#: Container runtimes this backend knows how to drive. ``none`` runs task
#: commands directly on the Ray worker with no nested container, and ``native``
#: does the same but has Ray supply each task's tools; see
#: :mod:`wdl_on_ray.runtimes` for why those modes exist.
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

#: How ``container_runtime = native`` installs a task's tool wheels.
ENV_INSTALLERS = ("pip", "uv")

#: What ``container_runtime = ray`` does with a task whose ``runtime.docker`` is not in
#: ``task_image_map``. ``error`` refuses to dispatch it; ``cluster`` runs it in the
#: cluster image without an ``image_uri``. See :attr:`RayConfig.task_image_fallback`.
TASK_IMAGE_FALLBACKS = ("error", "cluster")

#: Where a task's command runs. ``ray`` submits a Ray task per WDL task, which is the
#: backend's whole point and what this template uses. ``inprocess`` runs it here instead,
#: for a caller that has already scheduled the graph so this process *is* the Ray task.
DISPATCH_MODES = ("ray", "inprocess")

#: How to derive the per-task ceiling that miniwdl clamps ``runtime.cpu`` and
#: ``runtime.memory`` against.
LIMIT_SOURCES = ("max_node", "cluster", "local")


def _raw(cfg: Loader, key: str, default: str) -> str:
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
    """Ray cluster address. ``auto`` honours ``RAY_ADDRESS`` and otherwise starts
    a local instance; anything else is passed to ``ray.init(address=...)``."""

    namespace: str

    container_runtime: str
    """One of :data:`CONTAINER_RUNTIMES`. ``auto`` probes the cluster once, at
    startup, for the first usable runtime."""

    container_exe: list[str]
    """Override the executable for the chosen runtime, e.g. ``["sudo", "podman"]``.
    Empty means "use the runtime's own default"."""

    limit_source: str
    """See :data:`LIMIT_SOURCES`. ``max_node`` is the default and the only one
    that's generally correct: a single WDL task runs inside a single container on
    a single node, so the ceiling that matters is the *largest node*, not the sum
    across the cluster (a 64-CPU request is unschedulable on 8 x 8-CPU nodes no
    matter what the cluster total says)."""

    max_cpu: int
    """Hard override for the per-task CPU ceiling; 0 means "derive it from
    ``limit_source``". Set this when the cluster autoscales from a small head
    node to large workers, since the workers don't exist yet at startup."""

    max_memory_bytes: int
    """Hard override for the per-task memory ceiling; 0 means "derive it"."""

    reserve_memory: bool
    """Whether to pass ``runtime.memory`` to Ray as a ``memory`` resource
    request. On by default: it keeps Ray from packing more tasks onto a node than
    its RAM can serve."""

    scheduling_strategy: str
    """Ray scheduling strategy for task placement: ``DEFAULT`` or ``SPREAD``."""

    task_max_retries: int
    """Ray-level retries on *worker or node failure*. Defaults to 0 because
    miniwdl already implements WDL's ``runtime.maxRetries``/``preemptible``, and
    it resets the task's working directory between attempts, whereas a Ray-level retry
    would instead re-enter a dirty directory."""

    extra_resources: dict[str, float]
    """Custom Ray resources demanded by *every* task, e.g. ``{"wdl_node": 1}`` to
    confine WDL work to a labelled node group."""

    extra_container_args: list[str]
    """Extra arguments spliced into every container run invocation."""

    accelerator_type: str
    """Ray accelerator type demanded by every GPU task, when the WDL doesn't name
    one itself."""

    image_pull_timeout: int
    """Seconds to allow for an image pull on a worker node."""

    pull_lock_dir: str
    """Node-local directory for image-pull locks, so that N tasks landing on one
    node at once produce one pull instead of N."""

    chown_image: str
    """Tiny image used to hand rootless-container output files back to the
    invoking user. See :meth:`wdl_on_ray.runtimes.ContainerRuntime.chown_argv`."""

    sif_cache_dir: str
    """Where Apptainer/Singularity SIF conversions are cached. Pointing this at
    shared storage makes each image a once-per-cluster conversion instead of
    once-per-node."""

    disk_resource_name: str
    """When set, ``runtime.disks`` is also demanded as this custom Ray resource,
    so disk-hungry tasks can be steered onto node groups labelled with it. Empty
    (the default) means ``runtime.disks`` only gets logged."""

    dispatch: str
    """One of :data:`DISPATCH_MODES`. Leave at ``ray``: ``inprocess`` exists for a
    caller that has already scheduled the workflow graph itself, so that *this*
    process is the Ray task; see :meth:`wdl_on_ray.backend.RayContainer._run_inprocess`."""

    tool_wheel_dir: str
    """Where ``container_runtime = native`` finds the tool wheels: a directory on
    the shared storage this backend already requires, or a published
    ``--find-links`` index. Empty means resolve from a package index alone. Set
    ``env_offline`` as well for a cluster with no egress."""

    image_env_map: dict[str, Any]
    """``runtime.docker`` value -> Ray ``runtime_env``, for ``native`` mode. Lets
    a task's declared image select an environment without editing the WDL. Takes
    precedence over deriving the environment from the task's command."""

    env_offline: bool
    """Resolve tool wheels with ``--no-index``, for a cluster with no egress. Off by
    default: a ``kind = pypi`` tool such as ``quast`` ships only a dependency edge onto
    real distributions, so an index has to stay reachable unless those have been
    vendored into ``tool_wheel_dir`` too."""

    env_installer: str
    """One of :data:`ENV_INSTALLERS`. ``uv`` is faster; ``pip`` is the default
    because it is what Ray's own plugin reaches for first. Both require
    ``virtualenv`` (and ``pip``) in every node's base Python."""

    tool_manifest: str
    """Override the location of ``tools/manifest.toml``. Needed only when the
    package is installed without the template directory alongside it (as it is
    inside this template's image), since the default is discovered relative to
    this module."""

    env_extra_requirements: list[str]
    """Distributions added to every environment ``native`` mode *derives*, e.g. to
    ship a library each task's command expects. Per task, not per wheel, so
    a task needing no tool wheels still gets them. An environment named explicitly
    by ``runtime.ray_runtime_env`` or ``image_env_map`` is left exactly as given."""

    task_image_map: dict[str, str]
    """``runtime.docker`` value -> the image URI Ray should run that task in, for
    ``container_runtime = ray``.

    A map and not a passthrough, because the two values name different things. The
    WDL declares a portable image (``us.gcr.io/broad-dsp-lrma/lr-flye:2.8.3``); Ray
    needs one whose Ray and Python versions match the cluster's exactly, which means
    an image rebuilt on the cluster's own base. The map is where that correspondence
    is written down and version-controlled, so the WDL keeps declaring what it means
    and the deployment supplies what will run.

    Keys are matched exactly against ``runtime.docker``. An entry may also be the
    literal ``"*"``, which matches any image not otherwise listed."""

    task_image_fallback: str
    """What ``container_runtime = ray`` does with a task whose ``runtime.docker`` has
    no entry in :attr:`task_image_map`. One of :data:`TASK_IMAGE_FALLBACKS`.

    ``error`` (the default) refuses to dispatch. That is deliberate and is the whole
    reason to choose this runtime: silently running an unmapped task in the cluster
    image is exactly the "declared tag is advisory" behaviour of ``none``, and
    getting it by accident, on one task out of nine, in a run that otherwise looks
    isolated, is worse than getting it on purpose.

    ``cluster`` opts back into that, per deployment and in writing: unmapped tasks
    run in the cluster image with no ``image_uri``. Reasonable when most tasks are
    shell built-ins and only a few need a real toolchain."""


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
