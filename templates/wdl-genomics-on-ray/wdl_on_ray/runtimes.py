"""Pure argv builders for the container runtimes. Stdlib-only: it ships by value in every task."""

from __future__ import annotations

import os
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

#: ``(container_path, host_path, writable)``, miniwdl's ``prepare_mounts()`` shape.
Mount = tuple[str, str, bool]

#: Expanded on the worker, the only place Ray's ``CUDA_VISIBLE_DEVICES`` is known.
GPU_ARGS_SENTINEL = "@@WDL_ON_RAY_GPU_ARGS@@"


@dataclass(frozen=True)
class RunSpec:
    """Everything needed to formulate one container run."""

    image: str
    container_dir: str
    #: As the container sees it.
    workdir: str
    entry: list[str]
    mounts: list[Mount] = field(default_factory=list)
    cpu: float = 0.0
    memory_limit: int = 0
    num_gpus: float = 0.0
    privileged: bool = False
    network: str | None = None
    as_user: tuple[int, int] | None = None
    extra_args: list[str] = field(default_factory=list)
    #: Writable mounts the runtime needs (``/tmp``) beyond the task's own I/O.
    scratch_mounts: list[Mount] = field(default_factory=list)


class ContainerRuntime(ABC):
    """Argv builder for one container CLI."""

    name: str
    #: Own filesystem namespace? If not, container paths are host paths.
    isolated: bool = True
    #: Outputs land owned by a subordinate uid and need handing back.
    needs_chown: bool = False
    #: A Ray ``runtime_env`` supplies the tools. Only a flag: :mod:`envs` resolves it, driver-side.
    provides_env: bool = False
    #: That ``runtime_env`` is an image (``image_uri``), not a package set.
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
        """Command that makes ``image_ref`` available locally, or ``None``."""

    def image_present_argv(self, exe: list[str], image_ref: str) -> list[str] | None:
        """Cheap check for "already local", to skip a redundant pull."""
        return None

    @abstractmethod
    def run_argv(self, exe: list[str], spec: RunSpec) -> list[str]:
        """The full run invocation."""

    def gpu_args(self, devices: str | None, num_gpus: float) -> list[str]:
        """Flags exposing only the GPUs Ray assigned (``devices``) to the container."""
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
        # Rootless runtimes write outputs as a subordinate uid. Chown them back through a
        # throwaway container, as miniwdl's podman backend does, on the node holding the files.
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
        # CDI; `nvidia.com/gpu=N` selects one device by index.
        if devices:
            return [
                arg
                for index in devices.split(",")
                if index.strip()
                for arg in ("--device", f"nvidia.com/gpu={index.strip()}")
            ]
        return ["--device", "nvidia.com/gpu=all"]


class DockerRuntime(_OciRuntime):
    # The plain CLI, not miniwdl's docker_swarm: Swarm would schedule, and that is Ray's job here.
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
    """Apptainer / Singularity. A shared ``sif_cache_dir`` converts each image once per cluster."""

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
            # Run the docker:// URI directly and let apptainer cache it.
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
        # The container inherits CUDA_VISIBLE_DEVICES, so device selection is already right.
        return ["--nv"]


class SingularityRuntime(ApptainerRuntime):
    name = "singularity"

    @property
    def default_exe(self) -> list[str]:
        return ["singularity"]


class NoContainerRuntime(ContainerRuntime):
    """Run the task command directly on the Ray worker; its tools must already be there."""

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
    """No container either, but Ray supplies each task's tools via a ``runtime_env`` from envs."""

    name = "native"
    provides_env = True


class RayImageRuntime(NoContainerRuntime):
    """Per-task images, started by Ray around the worker via ``runtime_env`` ``image_uri``."""

    # Images must match the cluster's Ray and Python to the patch, be self-contained (no
    # pip/uv/conda/working_dir), and see the shared mount at the same path.
    # isolated stays False: the worker is inside the image, so host paths stay valid.
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

#: ``native`` and ``ray`` are opt-in: needing no executable, either would always win the probe.
AUTO_ORDER = ("podman", "docker", "apptainer", "singularity")


def get(name: str) -> ContainerRuntime:
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
