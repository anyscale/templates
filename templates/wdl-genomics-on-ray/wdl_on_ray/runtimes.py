"""Command-line shapes for the container runtimes this backend can drive.

Each :class:`ContainerRuntime` is a pure argv builder: it turns a
:class:`RunSpec` into a list of strings. Nothing here touches the filesystem,
imports Ray, or holds miniwdl state, which keeps it unit-testable and keeps the
worker side of the dispatch (:mod:`wdl_on_ray.job`) free of policy: the worker
only ever executes argv lists handed to it.

Six runtimes are supported; the first four in the order ``auto`` prefers them:

``podman``
    Rootless-capable and the usual choice on a workstation or an HPC-ish node.
``docker``
    Plain ``docker run``. Note this is *not* miniwdl's own ``docker_swarm``
    backend: Swarm does its own cluster scheduling, which is precisely the job
    Ray is doing here, so we drive the plain CLI instead.
``apptainer``/``singularity``
    Unprivileged containers on clusters that don't allow a container daemon.
``none``
    No nested container at all: the task command runs directly in the Ray
    worker's own environment. This exists because the common Ray deployments
    (Anyscale, KubeRay) *already* run Ray inside a container, and nesting one
    more usually requires privileges the pod or instance doesn't have. In this
    mode the WDL ``runtime.docker`` value is only advisory (it's logged, not
    honoured), so the tools each task invokes have to be present in the cluster
    image. The template's ``Dockerfile`` is how the bundled pipeline does that.
``native``
    ``none``'s dispatch, but Ray supplies each task's tools through a
    ``runtime_env`` instead of requiring them all in one image, so
    ``runtime.docker`` becomes a resolvable specification again instead of a
    comment, still without starting a container. The argv is identical to
    ``none``; everything specific to the mode lives in :mod:`wdl_on_ray.envs`,
    which also documents where the isolation stops.
``ray``
    Per-task container images, via Ray's own ``runtime_env={"image_uri": ...}``.
    Ray starts the nested container *around the worker process* before the task
    runs, so this backend never invokes a container CLI, which is what makes it
    work in the deployments where ``podman``/``docker`` cannot. This is the mode
    that restores the thing ``none`` gives up: each task runs in an image, so
    ``runtime.docker`` describes what actually executed. See
    :class:`RayImageRuntime` for the constraints that come with it.
"""

from __future__ import annotations

import os
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

#: ``(container_path, host_path, writable)``, miniwdl's ``prepare_mounts()`` shape.
Mount = tuple[str, str, bool]

#: Placeholder element the driver emits where GPU flags belong. The worker
#: expands it once it can see the ``CUDA_VISIBLE_DEVICES`` that Ray assigned,
#: which is the only place the concrete device list is known.
GPU_ARGS_SENTINEL = "@@WDL_ON_RAY_GPU_ARGS@@"


@dataclass(frozen=True)
class RunSpec:
    """Everything needed to formulate one container run."""

    image: str
    container_dir: str
    workdir: str
    """Working directory *as the container sees it*."""
    entry: list[str]
    """Command to exec inside the container, e.g. ``["/bin/sh", "-c", "..."]``."""
    mounts: list[Mount] = field(default_factory=list)
    cpu: float = 0.0
    memory_limit: int = 0
    num_gpus: float = 0.0
    privileged: bool = False
    network: str | None = None
    as_user: tuple[int, int] | None = None
    extra_args: list[str] = field(default_factory=list)
    scratch_mounts: list[Mount] = field(default_factory=list)
    """Additional writable mounts (e.g. ``/tmp``) that the runtime needs but that
    aren't part of the WDL task's own I/O."""


class ContainerRuntime(ABC):
    """Argv builder for one container CLI."""

    name: str
    #: Whether the task command runs in its own filesystem namespace. ``False``
    #: for :class:`NoContainerRuntime`, which the backend uses to decide whether
    #: it must translate container paths to host paths at all.
    isolated: bool = True
    #: ``True`` when output files land owned by a subordinate uid and need
    #: handing back to the invoking user after the run.
    needs_chown: bool = False
    #: ``True`` when the task's environment is supplied by a Ray ``runtime_env``
    #: and not by an image. Declarative on purpose: resolving the env needs
    #: the toolchain manifest and miniwdl's runtime values, neither of which may
    #: be imported here (this module ships *by value* into every Ray task and is
    #: stdlib-only by contract). So the flag lives here and the resolution lives
    #: in :mod:`wdl_on_ray.envs`, which only the driver imports.
    provides_env: bool = False
    #: ``True`` when that ``runtime_env`` is an *image* rather than a package set,
    #: i.e. ``runtime.docker`` resolves to ``{"image_uri": ...}``. Splits the two
    #: ``provides_env`` modes without :mod:`wdl_on_ray.envs` having to know runtime
    #: class names.
    env_is_image: bool = False

    @property
    @abstractmethod
    def default_exe(self) -> list[str]:
        """Default executable, used when ``[ray] container_exe`` is unset."""

    def version_argv(self, exe: list[str]) -> list[str]:
        """Probe used by ``auto`` detection and by startup validation."""
        return [*exe, "--version"]

    def image_ref(self, image: str, *, cache_dir: str | None = None) -> str:
        """Runtime-specific spelling of a Docker image reference."""
        del cache_dir
        return image

    @abstractmethod
    def pull_argv(self, exe: list[str], image_ref: str, *, source: str) -> list[str] | None:
        """Command that makes ``image_ref`` available locally, or ``None`` if
        nothing needs doing."""

    def image_present_argv(self, exe: list[str], image_ref: str) -> list[str] | None:
        """Cheap check for "already local", to skip a redundant pull."""
        return None

    @abstractmethod
    def run_argv(self, exe: list[str], spec: RunSpec) -> list[str]:
        """The full run invocation."""

    def gpu_args(self, devices: str | None, num_gpus: float) -> list[str]:
        """Flags exposing GPUs to the container.

        ``devices`` is the ``CUDA_VISIBLE_DEVICES`` Ray assigned to the worker;
        honouring it (instead of exposing every GPU) is what keeps two tasks
        sharing a multi-GPU node from both grabbing device 0.
        """
        del devices, num_gpus
        return []

    def chown_argv(
        self,
        exe: list[str],
        *,
        image: str,
        host_dir: str,
        container_dir: str,
        target: str,
        uid: int,
        gid: int,
    ) -> list[str] | None:
        """Command that returns ownership of task outputs to ``uid:gid``."""
        return None

    @staticmethod
    def _bind_arg(container_path: str, host_path: str, writable: bool) -> str:
        if ":" in container_path or ":" in host_path:
            raise ValueError(
                f"cannot bind-mount a path containing ':' ({host_path} -> {container_path})"
            )
        return f"{host_path}:{container_path}" + ("" if writable else ":ro")


class _OciRuntime(ContainerRuntime):
    """Shared shape for the Docker-compatible CLIs (``podman``, ``docker``)."""

    needs_chown = True

    def pull_argv(self, exe: list[str], image_ref: str, *, source: str) -> list[str] | None:
        del source
        return [*exe, "pull", image_ref]

    def image_present_argv(self, exe: list[str], image_ref: str) -> list[str] | None:
        return [*exe, "image", "exists", image_ref]

    def run_argv(self, exe: list[str], spec: RunSpec) -> list[str]:
        argv = [*exe, "run", "--rm", "--workdir", spec.workdir]
        if spec.cpu > 0:
            argv += ["--cpus", str(spec.cpu)]
        if spec.memory_limit > 0:
            argv += ["--memory", str(spec.memory_limit)]
        if spec.network is not None:
            argv += ["--network", spec.network]
        if spec.as_user is not None:
            argv += ["--user", f"{spec.as_user[0]}:{spec.as_user[1]}"]
        if spec.privileged:
            argv.append("--privileged")
        if spec.num_gpus:
            argv.append(GPU_ARGS_SENTINEL)
        argv += spec.extra_args
        for container_path, host_path, writable in [*spec.mounts, *spec.scratch_mounts]:
            argv += ["-v", self._bind_arg(container_path, host_path, writable)]
        argv.append(spec.image)
        argv += spec.entry
        return argv

    def chown_argv(
        self,
        exe: list[str],
        *,
        image: str,
        host_dir: str,
        container_dir: str,
        target: str,
        uid: int,
        gid: int,
    ) -> list[str] | None:
        # Rootless podman/docker write outputs as a subordinate uid, which the
        # invoking user then cannot read. miniwdl's own podman backend solves
        # this by chowning through a throwaway container; do the same, but on the
        # worker node where the files actually are.
        quoted = shlex.quote(target)
        script = (
            f"(find {quoted} -type d -print0 && find {quoted} -type f -print0"
            f" && find {quoted} -type l -print0)"
            f" | xargs -0 -r -P 10 chown -Ph {uid}:{gid}"
        )
        return [
            *exe,
            "run",
            "--rm",
            "-v",
            f"{host_dir}:{container_dir}",
            image,
            "/bin/sh",
            "-eo",
            "pipefail",
            "-c",
            script,
        ]


class PodmanRuntime(_OciRuntime):
    name = "podman"

    @property
    def default_exe(self) -> list[str]:
        return ["podman"]

    def gpu_args(self, devices: str | None, num_gpus: float) -> list[str]:
        # CDI is podman's supported path for NVIDIA devices. `nvidia.com/gpu=N`
        # selects an individual device by index.
        if devices:
            return [
                arg
                for index in devices.split(",")
                if index.strip()
                for arg in ("--device", f"nvidia.com/gpu={index.strip()}")
            ]
        return ["--device", "nvidia.com/gpu=all"]


class DockerRuntime(_OciRuntime):
    name = "docker"

    @property
    def default_exe(self) -> list[str]:
        return ["docker"]

    def image_present_argv(self, exe: list[str], image_ref: str) -> list[str] | None:
        return [*exe, "image", "inspect", image_ref]

    def gpu_args(self, devices: str | None, num_gpus: float) -> list[str]:
        if devices:
            return ["--gpus", f'"device={devices}"']
        if num_gpus:
            return ["--gpus", str(int(num_gpus))]
        return []


class ApptainerRuntime(ContainerRuntime):
    """Apptainer / Singularity.

    Images are converted to SIF files. Pointing ``[ray] sif_cache_dir`` at shared
    storage turns the conversion into a once-per-cluster cost instead of
    once-per-node.
    """

    name = "apptainer"

    @property
    def default_exe(self) -> list[str]:
        return ["apptainer"]

    def image_ref(self, image: str, *, cache_dir: str | None = None) -> str:
        if not cache_dir:
            return "docker://" + image
        sanitized = image.replace("/", "_").replace(":", "_")
        return os.path.join(cache_dir, sanitized + ".sif")

    def pull_argv(self, exe: list[str], image_ref: str, *, source: str) -> list[str] | None:
        if not image_ref.endswith(".sif"):
            # Running a docker:// URI directly lets apptainer manage its own
            # cache; no separate pull step.
            return None
        return [*exe, "pull", image_ref, "docker://" + source]

    def image_present_argv(self, exe: list[str], image_ref: str) -> list[str] | None:
        if image_ref.endswith(".sif"):
            return ["test", "-f", image_ref]
        return None

    def run_argv(self, exe: list[str], spec: RunSpec) -> list[str]:
        argv = [*exe, "exec", "--containall", "--pwd", spec.workdir]
        if spec.privileged:
            argv += ["--add-caps", "all"]
        if spec.num_gpus:
            argv.append(GPU_ARGS_SENTINEL)
        argv += spec.extra_args
        for container_path, host_path, writable in [*spec.mounts, *spec.scratch_mounts]:
            argv += ["--bind", self._bind_arg(container_path, host_path, writable)]
        argv.append(spec.image)
        argv += spec.entry
        return argv

    def gpu_args(self, devices: str | None, num_gpus: float) -> list[str]:
        # --nv shares the host's driver and device nodes; the container inherits
        # CUDA_VISIBLE_DEVICES, so device selection is already correct.
        return ["--nv"]


class SingularityRuntime(ApptainerRuntime):
    name = "singularity"

    @property
    def default_exe(self) -> list[str]:
        return ["singularity"]


class NoContainerRuntime(ContainerRuntime):
    """Run the task command directly on the Ray worker.

    ``runtime.docker`` is not honoured here: the required tools must already be
    on the worker. In exchange, this is the only mode that works unmodified
    inside an already-containerized Ray deployment.
    """

    name = "none"
    isolated = False

    @property
    def default_exe(self) -> list[str]:
        return []

    def version_argv(self, exe: list[str]) -> list[str]:
        return ["/bin/sh", "-c", "exit 0"]

    def pull_argv(self, exe: list[str], image_ref: str, *, source: str) -> list[str] | None:
        return None

    def run_argv(self, exe: list[str], spec: RunSpec) -> list[str]:
        return list(spec.entry)


class NativeRuntime(NoContainerRuntime):
    """No container either, but Ray supplies the task's tools.

    Identical to :class:`NoContainerRuntime` in everything this module decides: the argv is
    still just the task command, because the isolation does not come from the invocation. It
    comes from *how the Ray task is submitted*: the backend attaches a ``runtime_env`` built by
    :mod:`wdl_on_ray.envs`, and Ray materializes the tools per node before the command runs.

    So this is the mode that gives ``runtime.docker`` meaning again without starting a
    container: where ``none`` needs every task's tools in one cluster image, this resolves each
    task to only the tools it invokes. What it still does not provide is a filesystem
    namespace; see :mod:`wdl_on_ray.envs` for where that line falls.
    """

    name = "native"
    provides_env = True


class RayImageRuntime(NoContainerRuntime):
    """Per-task container images, started by Ray rather than by this backend.

    Ray's ``runtime_env`` accepts an ``image_uri``, and when it is set Ray runs the
    *worker process itself* inside that image before handing it any work. So the
    task command still executes as a plain subprocess of that worker (the argv is
    identical to :class:`NoContainerRuntime`) and this backend never invokes a
    container CLI at all. That indirection is the whole point: it is why per-task
    images are available in deployments where ``podman run`` cannot start, because
    the nesting is done by the platform's own machinery instead of ours.

    What this buys over ``none``: ``runtime.docker`` names something that actually
    ran. Each task gets its own image, so the pipeline's declared provenance is the
    executed provenance, and a fifty-tool pipeline does not need one image holding
    all fifty. That is the trade ``none`` makes and the reason it cannot be used
    where per-task image provenance is a requirement.

    Four constraints come with it, all of them Ray's or the platform's rather than
    this backend's, and all of them load-bearing when building an image matrix:

    * **Version lockstep.** The nested image's Ray and Python must match the host
      image's exactly, Python down to the patch level. Task images therefore have
      to be built *from* the cluster's base image, not from an arbitrary upstream
      one. A pipeline's declared images (``lr-flye:2.8.3`` and friends) will not run
      unmodified; they have to be rebuilt on the matching base and mapped. See
      ``[ray] task_image_map``.
    * **No package fields alongside it.** Ray rejects a ``runtime_env`` that names
      ``image_uri`` together with ``pip``, ``uv``, ``conda`` or ``working_dir``, so
      each image must be self-contained. ``env_vars`` is allowed and is used.
      :func:`wdl_on_ray.envs.validate_runtime_env` enforces this before dispatch
      rather than letting Ray fail the task later.
    * **Shared storage must reach inside.** miniwdl's model is filesystem-mediated:
      task B reads task A's outputs by path. Those paths live on the cluster's
      shared mount, and the nested container has to see them at the same location.
      This holds where the platform propagates node mounts into the worker
      container; ``wdl-on-ray doctor`` checks it, because a run that gets this wrong
      fails confusingly, one task late.
    * **Privileges.** On Kubernetes-backed clouds the ray container has to run
      privileged for the nested worker container to start.

    ``isolated`` stays ``False``, which reads oddly and is correct. It governs
    whether *this backend* must translate between container and host paths, and it
    must not: the worker is inside the container, sees the node's mounts at their
    node paths, and miniwdl's absolute paths stay valid unchanged. The task does get
    a filesystem namespace; it just is not one this module has to compensate for.
    """

    name = "ray"
    provides_env = True
    env_is_image = True


_REGISTRY: dict[str, type[ContainerRuntime]] = {
    "podman": PodmanRuntime,
    "docker": DockerRuntime,
    "apptainer": ApptainerRuntime,
    "singularity": SingularityRuntime,
    "none": NoContainerRuntime,
    "native": NativeRuntime,
    "ray": RayImageRuntime,
}

#: Order in which ``container_runtime = auto`` probes for something usable.
#:
#: ``native`` and ``ray`` are deliberately absent. ``auto`` picks the first runtime whose probe
#: succeeds, and a runtime needing no executable always succeeds, so listing either would
#: silently capture every ``auto`` user the moment no container CLI was found, which is precisely
#: the case where the existing fallback to ``none`` is the *less* surprising answer. ``ray``
#: additionally needs an image map that only its operator can supply, so a silent selection would
#: turn a missing CLI into an unrelated configuration error. Opt in by name.
AUTO_ORDER = ("podman", "docker", "apptainer", "singularity")


def get(name: str) -> ContainerRuntime:
    """Instantiate a runtime by name."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"unknown container runtime {name!r}; expected one of {', '.join(_REGISTRY)}"
        ) from None


def expand_gpu_args(
    argv: list[str], runtime: ContainerRuntime, devices: str | None, num_gpus: float
) -> list[str]:
    """Replace :data:`GPU_ARGS_SENTINEL` with the runtime's real GPU flags."""
    if GPU_ARGS_SENTINEL not in argv:
        return argv
    replacement = runtime.gpu_args(devices, num_gpus)
    out: list[str] = []
    for item in argv:
        if item == GPU_ARGS_SENTINEL:
            out.extend(replacement)
        else:
            out.append(item)
    return out
