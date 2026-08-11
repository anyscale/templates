"""A miniwdl container backend that runs each WDL task as a Ray task.

Registered as the ``ray`` entry in miniwdl's ``miniwdl.plugin.container_backend``
group, so it is selected with ``[scheduler] container_backend = ray``.

How it fits together
--------------------
miniwdl's task runner hands a :class:`~WDL.runtime.task_container.TaskContainer`
a work directory and a shell command, and expects the command to run with the
task's inputs mounted and its outputs left behind on disk. The stock backends
satisfy that with a container on the local machine. This one keeps the exact
same filesystem contract but *dispatches* the container to a Ray task, so it
lands on whichever node Ray picks, which is what turns a workflow into
something a cluster can absorb.

Three consequences follow, and they drive most of the code below.

1. The run directory must be visible from every node. miniwdl's model is
   filesystem-mediated: it writes the command file, bind-mounts inputs, and reads
   outputs and stderr back from the same paths. We don't try to hide that behind
   object-store staging; we require shared storage and check for it at startup
   (``/mnt/cluster_storage`` on Anyscale, an NFS/EFS/Lustre mount elsewhere).
   This keeps miniwdl's call cache, output globs and live stderr tailing working
   untouched.

2. The resource ceiling is one node, not the whole cluster. miniwdl clamps
   ``runtime.cpu``/``runtime.memory`` to whatever ``detect_resource_limits``
   reports. A WDL task is one container on one node, so reporting the cluster
   *total* would let a task ask for 64 CPUs on a fleet of 8-CPU nodes and then
   sit unschedulable forever. See :meth:`RayContainer.detect_resource_limits`.

3. Node loss is a WDL-level interruption. Ray reports a dead worker or a
   reclaimed spot node as a task error. Mapping those onto miniwdl's
   ``Interrupted`` makes WDL's own ``runtime.preemptible`` counter do the
   retrying, with a clean working directory each attempt, which is what
   pipelines like the Broad's (``preemptible_tries: 3``) already expect.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, cast

from WDL import Error, Type
from WDL._util import PygtailLogger
from WDL._util import StructuredLogMessage as _
from WDL.runtime import config as wdl_config
from WDL.runtime.backend.cli_subprocess import SubprocessBase
from WDL.runtime.error import DownloadFailed, Interrupted, Terminated

from wdl_on_ray import config as ray_config
from wdl_on_ray import envs, resources, runtimes
from wdl_on_ray import job as ray_job

#: Path prefixes that are shared across the nodes of a cluster. Anyscale mounts
#: the first two; the rest are conventional mount points for NFS/EFS/FSx.
SHARED_STORAGE_PREFIXES = (
    "/mnt/cluster_storage",
    "/mnt/shared_storage",
    "/mnt/user_storage",
    "/mnt/shared",
    "/mnt/nfs",
    "/mnt/efs",
    "/mnt/fsx",
)

#: Grace period between a polite and a forced ``ray.cancel``.
_CANCEL_GRACE_SECONDS = 15.0

#: Ray errors that mean "the node or worker went away", not "the task failed".
_INTERRUPTION_ERRORS = (
    "NodeDiedError",
    "WorkerCrashedError",
    "LocalRayletDiedError",
    "OwnerDiedError",
    "ObjectLostError",
    "ObjectFetchTimedOutError",
)


def _is_interruption(exn: BaseException) -> bool:
    """Did the node or worker go away, as opposed to the task failing on its merits?

    ``isinstance``, not a name comparison. ``ObjectLostError`` has subclasses, and on
    ray 2.56 three of them (``ObjectReconstructionFailedError``, which has its own
    subclasses, plus ``ReferenceCountingAssertionError`` and ``ObjectFreedError``) are
    not named in the tuple above. Those are raised when an object is lost *because* the
    node holding it died, which is the reclaimed-spot case: it should spend the WDL's
    ``runtime.preemptible`` budget. Matching on the exact name let them through as
    ordinary failures, where they consumed ``maxRetries`` instead and bypassed the
    pipeline's own preemption policy entirely.

    The name tuple stays as a fallback, so a Ray release that renames or adds a class
    still degrades to the previous behaviour rather than to nothing.
    """
    from ray import exceptions as ray_exceptions

    classes = tuple(
        cls
        for cls in (getattr(ray_exceptions, name, None) for name in _INTERRUPTION_ERRORS)
        if isinstance(cls, type)
    )
    return (classes and isinstance(exn, classes)) or type(exn).__name__ in _INTERRUPTION_ERRORS


#: Whether :func:`connect` has run in this process.
_connected = False

#: Modules the Ray worker needs in order to run a task. Serialized *by value*
#: (see :func:`_pickle_worker_modules_by_value`), so both are deliberately
#: stdlib-only: adding a miniwdl or Ray import to either would drag it into
#: every task's payload.
_WORKER_MODULES = (ray_job, runtimes)


def _pickle_worker_modules_by_value() -> None:
    """Ship the worker-side code inside each task instead of importing it there.

    Otherwise cloudpickle serializes the task function and its ``ContainerJob``
    argument *by reference* (module path plus qualname) and every Ray worker
    has to be able to ``import wdl_on_ray``. Arranging that is a surprising amount
    of deployment surface: installing the package in the cluster image works, but
    on an uploaded ``working_dir`` it does not, and the resulting error names an
    argument-deserialization failure, giving no hint of the real cause. Attaching the
    package via ``ray.init(runtime_env={"py_modules": ...})`` works too, until the
    submitting job declares ``py_modules`` of its own, and Ray then refuses to merge
    two declarations of the same field and the run dies at the first task.

    Serializing by value sidesteps all of it: the worker needs nothing but Ray.
    The cost is a few KB of module bytecode per task, against a container launch.

    Idempotent, and safe when the package *is* installed: by-value simply wins.
    """
    from ray.cloudpickle import register_pickle_by_value

    for module in _WORKER_MODULES:
        register_pickle_by_value(module)


def connect(ray_cfg: ray_config.RayConfig, logger: logging.Logger) -> None:
    """Connect to Ray. Idempotent.

    The single place this package calls ``ray.init()``, so that connection-time
    decisions are made once and cannot be pre-empted by something else touching
    Ray first.
    """
    global _connected
    import ray

    _pickle_worker_modules_by_value()

    if ray.is_initialized():
        _connected = True
        return

    kwargs: dict[str, Any] = {
        "namespace": ray_cfg.namespace,
        "ignore_reinit_error": True,
        # miniwdl owns the console; Ray worker stdout would interleave with (and
        # drown out) the per-task logs.
        "log_to_driver": False,
    }
    address = ray_cfg.address
    if address and address != "auto":
        kwargs["address"] = address
    elif os.environ.get("RAY_ADDRESS"):
        kwargs["address"] = os.environ["RAY_ADDRESS"]

    ray.init(**kwargs)
    _connected = True
    logger.notice(  # type: ignore[attr-defined]
        _(
            "connected to Ray",
            address=kwargs.get("address", "local"),
            nodes=len([n for n in ray.nodes() if n.get("Alive")]),
            cluster_cpus=int(ray.cluster_resources().get("CPU", 0)),
            cluster_gpus=int(ray.cluster_resources().get("GPU", 0)),
        )
    )


class RayContainer(SubprocessBase):
    """Dispatch WDL task containers onto a Ray cluster.

    Inherits :class:`~WDL.runtime.backend.cli_subprocess.SubprocessBase` for its
    mount preparation and input-copy bookkeeping, and replaces
    :meth:`_run`, the part that would otherwise spawn a local subprocess.
    """

    _ray_cfg: ray_config.RayConfig
    _runtime: runtimes.ContainerRuntime
    _exe: list[str]
    _limits: dict[str, int] | None = None
    _limits_lock = threading.Lock()
    _sif_cache_dir: str | None = None
    _checked_shared_run_dir = False

    # ------------------------------------------------------------------ startup

    @classmethod
    def global_init(cls, cfg: wdl_config.Loader, logger: logging.Logger) -> None:
        cls._ray_cfg = ray_config.load(cfg)
        cls._runtime = cls._resolve_runtime(cls._ray_cfg, logger)
        cls._exe = cls._ray_cfg.container_exe or cls._runtime.default_exe

        if isinstance(cls._runtime, runtimes.ApptainerRuntime):
            cls._sif_cache_dir = cls._ray_cfg.sif_cache_dir or None
            if cls._sif_cache_dir:
                os.makedirs(cls._sif_cache_dir, exist_ok=True)

        cls._init_ray(logger)
        limits = cls.detect_resource_limits(cfg, logger)

        logger.notice(  # type: ignore[attr-defined]
            _(
                "Ray container backend initialized",
                container_runtime=cls._runtime.name,
                exe=" ".join(cls._exe) or "(none)",
                task_cpu_limit=limits["cpu"],
                task_mem_bytes_limit=limits["mem_bytes"],
                limit_source=cls._ray_cfg.limit_source,
            )
        )
        if cls._runtime.env_is_image:
            cls._warn_unless_task_images_usable(logger)
            logger.notice(  # type: ignore[attr-defined]
                _(
                    "container_runtime=ray: each task runs in its own image, started by Ray"
                    " around the worker process. runtime.docker is resolved through"
                    " [ray] task_image_map, so what ran is what the map says, not what the"
                    " WDL declares",
                    mapped_images=len(cls._ray_cfg.task_image_map),
                    unmapped_tasks=cls._ray_cfg.task_image_fallback,
                )
            )
        elif cls._runtime.provides_env:
            cls._warn_unless_env_plugin_usable(logger)
            cls._warn_unless_wheelhouse_usable(logger)
            logger.notice(  # type: ignore[attr-defined]
                _(
                    "container_runtime=native: each task's tools come from a Ray runtime_env."
                    " Tasks share the workers' filesystem, so this isolates environments, not"
                    " filesystems",
                    installer=cls._ray_cfg.env_installer,
                    tool_wheel_dir=cls._ray_cfg.tool_wheel_dir or "(resolve from an index)",
                )
            )
        elif not cls._runtime.isolated:
            logger.warning(
                "container_runtime=none: WDL runtime.docker is advisory only, and each task's"
                " tools must already be present on the Ray workers"
            )

    @classmethod
    def _warn_unless_task_images_usable(cls, logger: logging.Logger) -> None:
        """Check ``container_runtime = ray``'s configuration before the first task.

        Two things are worth catching here rather than one task in. An empty image
        map with the default ``error`` fallback cannot dispatch *anything*, so it is
        a configuration mistake rather than a per-task condition and is raised as
        one. And the version lockstep (a task image's Ray and Python must match the
        cluster's exactly, Python to the patch) cannot be checked from the driver
        without pulling every image, so the next best thing is to state the versions
        an image has to be built against, in the run log, where the person building
        them will look.
        """
        if not cls._ray_cfg.task_image_map and cls._ray_cfg.task_image_fallback == "error":
            raise Error.RuntimeError(
                "container_runtime=ray with an empty [ray] task_image_map: every task would"
                " fail to resolve an image. Map each runtime.docker value to an image URI"
                " (a '*' entry catches the rest), or set [ray] task_image_fallback = cluster"
                " to run unmapped tasks in the cluster image."
            )

        import platform

        import ray

        logger.notice(  # type: ignore[attr-defined]
            _(
                "task images must be built on a base matching these versions exactly"
                " (Python to the patch level) or Ray will refuse to start the worker",
                ray_version=ray.__version__,
                python_version=platform.python_version(),
            )
        )

        # miniwdl passes files between tasks by path, so every task image has to see
        # the run directory at the same absolute path the driver wrote it to. That
        # holds when the platform propagates the node's mounts into the nested worker
        # container, and fails one task late when it does not, as a missing input
        # file, naming a path that plainly exists.
        logger.info(
            "container_runtime=ray requires the run directory to be visible at the same path"
            " inside each task image; a task that cannot see its inputs is the symptom when"
            " it is not"
        )

    @classmethod
    def _warn_unless_wheelhouse_usable(cls, logger: logging.Logger) -> None:
        """Check the wheel directory before any task tries to install from it.

        A ``tool_wheel_dir`` that is missing, empty, or not visible from this node makes every
        task's environment resolve against a package index instead, where the pinned
        ``wdl-on-ray-tools-*`` versions do not exist, so the first task fails inside Ray's
        runtime_env setup, nowhere near anything that names the directory. Cheap to check,
        and the mistake is easy to make: a wheelhouse left somewhere ``.gitignore`` excludes
        never reaches the cluster at all.

        A URL is left alone (only a local path can be inspected) and an empty setting is a
        deliberate "resolve from an index", not a mistake.
        """
        wheel_dir = cls._ray_cfg.tool_wheel_dir
        if not wheel_dir or "://" in wheel_dir:
            return
        import glob

        if not os.path.isdir(wheel_dir):
            logger.warning(
                _(
                    "[ray] tool_wheel_dir does not exist on this node; every task will resolve"
                    " its tools from a package index, where the pinned versions are not",
                    tool_wheel_dir=wheel_dir,
                )
            )
            return
        wheels = glob.glob(os.path.join(wheel_dir, "*.whl"))
        if not wheels:
            logger.warning(_("[ray] tool_wheel_dir holds no wheels", tool_wheel_dir=wheel_dir))
            return
        logger.info(_("tool wheelhouse", tool_wheel_dir=wheel_dir, wheels=len(wheels)))

    @staticmethod
    def _warn_unless_env_plugin_usable(logger: logging.Logger) -> None:
        """Check the two things Ray's ``pip``/``uv`` plugins need, before any task runs.

        Both import ``virtualenv`` and then materialize one, cloning the base environment when
        the driver already sits in a venv, so a base environment without ``pip`` produces a
        clone without ``pip`` and the install fails. Neither failure is discovered until the
        first task dispatches, and both report as a ``RuntimeEnvSetupError`` several frames
        deep, so checking here converts a confusing mid-run failure into a startup warning.

        Only a warning, and only about *this* node: the driver's environment is a good proxy for
        the workers' on a homogeneous cluster and no guarantee on any other.
        """
        import importlib.util

        missing = [name for name in ("virtualenv", "pip") if not importlib.util.find_spec(name)]
        if missing:
            logger.warning(
                _(
                    "container_runtime=native needs these importable in every node's base"
                    " Python; without them the first task fails with RuntimeEnvSetupError",
                    missing=missing,
                )
            )

    @classmethod
    def _resolve_runtime(
        cls, ray_cfg: ray_config.RayConfig, logger: logging.Logger
    ) -> runtimes.ContainerRuntime:
        """Pick a container runtime, probing the host when configured ``auto``.

        The probe necessarily runs on the node hosting the workflow driver. On a
        heterogeneous cluster where the workers differ, name the runtime
        explicitly with ``[ray] container_runtime`` instead of relying on
        ``auto``.
        """
        if ray_cfg.container_runtime != "auto":
            candidate = runtimes.get(ray_cfg.container_runtime)
            exe = ray_cfg.container_exe or candidate.default_exe
            if candidate.isolated and not cls._probe(candidate, exe):
                raise Error.RuntimeError(
                    f"container_runtime={candidate.name} is configured but"
                    f" `{' '.join(candidate.version_argv(exe))}` did not succeed;"
                    " verify the installation or set [ray] container_runtime"
                )
            return candidate

        for name in runtimes.AUTO_ORDER:
            candidate = runtimes.get(name)
            if cls._probe(candidate, candidate.default_exe):
                logger.info(_("auto-detected container runtime", runtime=name))
                return candidate

        logger.warning(
            "no container runtime found on this node (tried"
            f" {', '.join(runtimes.AUTO_ORDER)}); falling back to container_runtime=none,"
            " which runs task commands directly on the Ray workers"
        )
        return runtimes.get("none")

    @staticmethod
    def _probe(runtime: runtimes.ContainerRuntime, exe: list[str]) -> bool:
        if not runtime.isolated:
            return True
        try:
            return (
                subprocess.run(
                    runtime.version_argv(exe),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                    check=False,
                ).returncode
                == 0
            )
        except (OSError, subprocess.SubprocessError):
            return False

    @classmethod
    def _init_ray(cls, logger: logging.Logger) -> None:
        connect(cls._ray_cfg, logger)

    @classmethod
    def detect_resource_limits(
        cls, cfg: wdl_config.Loader, logger: logging.Logger
    ) -> dict[str, int]:
        """Report the ceiling miniwdl clamps ``runtime.cpu``/``memory`` against.

        Deliberately *not* the cluster total. A WDL task runs as a single
        container on a single node, so admitting a request larger than any node
        can satisfy produces a task that Ray will never schedule. ``max_node``
        (the default) therefore reports the largest live node.

        The one case that needs an operator override is an autoscaling cluster
        that starts with only a small head node: the big workers don't exist yet,
        so ``max_node`` under-reports. Set ``[ray] max_cpu`` /
        ``[ray] max_memory_bytes`` to the shape of the worker group in that case.
        """
        with cls._limits_lock:
            if cls._limits is not None:
                return cls._limits

            ray_cfg = getattr(cls, "_ray_cfg", None) or ray_config.load(cfg)
            cpu, mem = cls._probe_limits(ray_cfg, logger)

            if ray_cfg.max_cpu > 0:
                cpu = ray_cfg.max_cpu
            if ray_cfg.max_memory_bytes > 0:
                mem = ray_cfg.max_memory_bytes

            cls._limits = {"cpu": max(1, int(cpu)), "mem_bytes": max(1, int(mem))}
            return cls._limits

    @classmethod
    def _probe_limits(
        cls, ray_cfg: ray_config.RayConfig, logger: logging.Logger
    ) -> tuple[float, float]:
        if ray_cfg.limit_source == "local":
            import multiprocessing

            import psutil

            return multiprocessing.cpu_count(), psutil.virtual_memory().total

        import ray

        cls._init_ray(logger)
        if ray_cfg.limit_source == "cluster":
            total = ray.cluster_resources()
            return total.get("CPU", 1.0), total.get("memory", 0.0)

        nodes = [n for n in ray.nodes() if n.get("Alive")]
        cpu = max((n.get("Resources", {}).get("CPU", 0.0) for n in nodes), default=0.0)
        mem = max((n.get("Resources", {}).get("memory", 0.0) for n in nodes), default=0.0)
        if not cpu:
            # A cluster with no CPU-bearing node is either mid-scale-up or
            # GPU-only; fall back to the totals instead of clamping to 1.
            total = ray.cluster_resources()
            cpu, mem = total.get("CPU", 1.0), total.get("memory", 0.0)
        return cpu, mem

    # ------------------------------------------------------------- per-instance

    def __init__(self, cfg: wdl_config.Loader, run_id: str, host_dir: str) -> None:
        super().__init__(cfg, run_id, host_dir)
        if not self._runtime.isolated:
            # No filesystem namespace, so container paths *are* host paths.
            # Setting this before add_paths() runs means miniwdl's own path
            # mapping resolves to real locations with no translation.
            self.container_dir = self.host_dir
        # First task construction is the earliest moment that knows both the run
        # location and the live cluster shape, so the shared-storage check fires
        # here, once. (A benign race could emit it twice; that costs a duplicate
        # log line, not a lock.)
        if not RayContainer._checked_shared_run_dir:
            RayContainer._checked_shared_run_dir = True
            warn_if_not_shared(host_dir, logging.getLogger("wdl-on-ray"))

    def process_runtime(self, logger: logging.Logger, runtime_eval: dict[str, Any]) -> None:
        """Extend miniwdl's ``runtime {}`` handling with the keys Ray can use.

        The WDL spec only has a Boolean ``gpu``. Real pipelines, including the
        one this template ships, use Cromwell's Google-backend extensions
        (``gpuCount``, ``gpuType``, ``disks``), so those are read here rather
        than dropped, which is what lets such a WDL run on Ray unedited.
        """
        super().process_runtime(logger, runtime_eval)
        ans = self.runtime_values

        if "gpuCount" in runtime_eval:
            ans["gpuCount"] = max(0, runtime_eval["gpuCount"].coerce(Type.Int()).value)
        for key in ("gpuType", "acceleratorType"):
            if key in runtime_eval:
                ans[key] = runtime_eval[key].coerce(Type.String()).value
        if "disks" in runtime_eval:
            ans["disks"] = runtime_eval["disks"].coerce(Type.String()).value
        if "ray_resources" in runtime_eval:
            ans["ray_resources"] = runtime_eval["ray_resources"].coerce(Type.String()).value
        if "ray_runtime_env" in runtime_eval:
            ans["ray_runtime_env"] = runtime_eval["ray_runtime_env"].coerce(Type.String()).value

    @property
    def cli_name(self) -> str:
        """Names the container CLI in log messages and log filenames.

        Required by :class:`SubprocessBase`; here it tracks whichever runtime was
        configured or auto-detected.
        """
        return self._runtime.name

    @property
    def cli_exe(self) -> list[str]:
        return list(self._exe)

    def reset(self, logger: logging.Logger) -> None:
        """Prepare a fresh working directory for a retry.

        miniwdl advances the host-side work directory (``work`` -> ``work2``)
        between attempts so the failed attempt stays around for inspection, and
        relies on the bind mount to keep the *container* path at
        ``{container_dir}/work`` regardless. Without a container there is no
        mount to do that, and the task command, already rendered with absolute
        paths that say ``work``, would keep writing into the previous attempt's
        directory. Redirecting ``work`` as a symlink to the current attempt
        restores the invariant while still preserving attempt 1 (as ``work1``).
        """
        super().reset(logger)
        if self._runtime.isolated:
            return
        stable = os.path.join(self.host_dir, "work")
        if os.path.islink(stable):
            os.unlink(stable)
        elif os.path.isdir(stable):
            os.rename(stable, os.path.join(self.host_dir, "work1"))
        os.symlink(self.host_work_dir(), stable)

    def _ray_request(self) -> resources.RayRequest:
        return resources.build_request(
            self.runtime_values,
            reserve_memory=self._ray_cfg.reserve_memory,
            extra_resources=self._ray_cfg.extra_resources,
            default_accelerator_type=self._ray_cfg.accelerator_type,
            disk_resource_name=self._ray_cfg.disk_resource_name,
        )

    def _ray_env(self, command: str) -> envs.Resolved:
        """The Ray ``runtime_env`` for this task, for runtimes that supply one.

        Two shapes, split on :attr:`ContainerRuntime.env_is_image`. ``ray`` resolves
        ``runtime.docker`` to an ``image_uri`` and needs nothing from the command;
        ``native`` derives a package set and needs the *rendered* command, because
        the environment comes from the executables the task actually invokes. See
        :mod:`wdl_on_ray.envs`.
        """
        if self._runtime.env_is_image:
            return envs.resolve_image(
                self.runtime_values,
                task_image_map=self._ray_cfg.task_image_map,
                fallback=self._ray_cfg.task_image_fallback,
            )
        return envs.resolve(
            self.runtime_values,
            command,
            wheel_dir=self._ray_cfg.tool_wheel_dir,
            image_env_map=self._ray_cfg.image_env_map,
            installer=self._ray_cfg.env_installer,
            offline=self._ray_cfg.env_offline,
            manifest_path=self._ray_cfg.tool_manifest or None,
            extra_requirements=tuple(self._ray_cfg.env_extra_requirements),
        )

    # ------------------------------------------------------------- invocation

    def _image_ref(self) -> tuple[str, str]:
        """``(source, local_ref)`` for this task's image.

        ``source`` is the WDL ``runtime.docker`` value; ``local_ref`` is how the
        chosen runtime names it locally (identical for OCI CLIs, a ``.sif`` path
        or ``docker://`` URI for Apptainer).
        """
        source = self.runtime_values.get(
            "docker", self.cfg.get_dict("task_runtime", "defaults")["docker"]
        )
        return source, self._runtime.image_ref(source, cache_dir=self._sif_cache_dir)

    def _link_inputs(self, logger: logging.Logger) -> None:
        """Stand in for bind mounts when running without a container.

        Symlinks, because genomics inputs are routinely tens of GB,
        and both ends of the link are on the shared filesystem the cluster
        already requires.
        """
        if not self._bind_input_files:
            return  # copy_input_files() already put real files in place
        linked = 0
        for host_path, container_path in self.input_path_map.items():
            src = host_path.rstrip("/")
            dest = self.host_work_path(container_path).rstrip("/")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.lexists(dest):
                if os.path.islink(dest) and os.readlink(dest) == src:
                    continue
                os.unlink(dest)
            os.symlink(src, dest)
            linked += 1
        logger.info(_("linked task inputs", count=linked, mode="symlink"))

    def _run_invocation(self, logger: logging.Logger, cleanup: ExitStack, image: str) -> list[str]:
        """Container invocation *without* the trailing command, per the
        :class:`SubprocessBase` contract."""
        return self._build_argv(logger, cleanup, image, entry=[])

    def _build_argv(
        self, logger: logging.Logger, cleanup: ExitStack, image: str, entry: list[str]
    ) -> list[str]:
        request = self._ray_request()
        spec = runtimes.RunSpec(
            image=image,
            container_dir=self.container_dir,
            workdir=os.path.join(self.container_dir, "work"),
            entry=entry,
            cpu=float(self.runtime_values.get("cpu", 0) or 0),
            memory_limit=int(self.runtime_values.get("memory_limit", 0) or 0),
            num_gpus=request.num_gpus,
            privileged=bool(self.runtime_values.get("privileged", False)),
            network=self.runtime_values.get("docker_network"),
            as_user=(
                (os.geteuid(), os.getegid())
                if self.cfg.get_bool("task_runtime", "as_user")
                else None
            ),
            extra_args=list(self._ray_cfg.extra_container_args),
            mounts=self.prepare_mounts() if self._runtime.isolated else [],
            scratch_mounts=(
                self._apptainer_scratch(cleanup)
                if isinstance(self._runtime, runtimes.ApptainerRuntime)
                else []
            ),
        )
        return self._runtime.run_argv(self._exe, spec)

    def _apptainer_scratch(self, cleanup: ExitStack) -> list[runtimes.Mount]:
        """Give Apptainer real directories for ``/tmp`` and ``/var/tmp``.

        Its in-memory session directory is small and easily overrun by
        bioinformatics tools; miniwdl's own Singularity backend does the same.
        """
        tempdir = cleanup.enter_context(
            tempfile.TemporaryDirectory(prefix="_apptainer_tmpdir_", dir=self.host_dir)
        )
        os.mkdir(os.path.join(tempdir, "tmp"))
        os.mkdir(os.path.join(tempdir, "var_tmp"))
        return [
            ("/tmp", os.path.join(tempdir, "tmp"), True),
            ("/var/tmp", os.path.join(tempdir, "var_tmp"), True),
        ]

    def _entry(self) -> list[str]:
        """Shell invocation that runs the task's command file.

        With a container, miniwdl's own relative-path convention works because
        ``stdout.txt``/``stderr.txt``/``command`` are individually bind-mounted.
        Without one there are no mounts, so the redirections have to name the
        real host paths (which also keeps retries, ``stdout2.txt`` and friends,
        pointing at the right files).
        """
        shell = self.cfg.get("task_runtime", "command_shell")
        if self._runtime.isolated:
            return ["/bin/sh", "-c", f"{shell} ../command >> ../stdout.txt 2>> ../stderr.txt"]
        return [
            "/bin/sh",
            "-c",
            f"{shell} {shlex.quote(os.path.join(self.host_dir, 'command'))}"
            f" >> {shlex.quote(self.host_stdout_txt())}"
            f" 2>> {shlex.quote(self.host_stderr_txt())}",
        ]

    def _write_command_file(self, command: str) -> str:
        """Materialize the task command, prefixed with its environment.

        Same approach as miniwdl's subprocess backends: exporting inside the
        script sidesteps both command-line length limits and the inconsistent
        quoting of ``--env-file`` across runtimes.
        """
        path = os.path.join(self.host_dir, "command")
        with open(path, "w") as outfile:
            for key, value in self.runtime_values.get("env", {}).items():
                outfile.write(f"export {key}={shlex.quote(value)}\n")
            outfile.write(command)
        return path

    # --------------------------------------------------------------- execution

    def _run(self, logger: logging.Logger, terminating: Callable[[], bool], command: str) -> int:
        import ray

        with ExitStack() as cleanup:
            request = self._ray_request()
            source, image_ref = self._image_ref()
            self._write_command_file(command)

            if self._runtime.isolated:
                argv = self._build_argv(logger, cleanup, image_ref, self._entry())
            else:
                # prepare_mounts() is what normally creates these; without it we
                # make the stream files and input links ourselves.
                for stream in (self.host_stdout_txt(), self.host_stderr_txt()):
                    if not os.path.exists(stream):
                        self.touch_mount_point(stream)
                self._link_inputs(logger)
                argv = self._build_argv(logger, cleanup, image_ref, self._entry())

            cli_log_filename = os.path.join(self.host_dir, f"{self.cli_name}.log.txt")
            placement_path = os.path.join(self.host_dir, "ray_placement.json")

            # Create the log now, on the driver. The worker appends to it, but the
            # driver starts tailing it immediately, and if the task fails before
            # ever running (unschedulable, a lost node, a bad argument) the file
            # would never exist, and Pygtail's FileNotFoundError buries the actual
            # error under an unrelated traceback.
            with open(cli_log_filename, "a"):
                pass

            job = ray_job.ContainerJob(
                runtime_name=self._runtime.name,
                run_argv=argv,
                cwd=self.host_work_dir() if not self._runtime.isolated else self.host_dir,
                cli_log_path=cli_log_filename,
                exe=list(self._exe),
                image_ref=image_ref if self._runtime.isolated else None,
                image_source=source,
                pull_argv=(
                    self._runtime.pull_argv(self._exe, image_ref, source=source)
                    if self._runtime.isolated
                    else None
                ),
                image_present_argv=self._runtime.image_present_argv(self._exe, image_ref),
                chown_argv=self._chown_argv(),
                pull_lock_dir=self._ray_cfg.pull_lock_dir,
                pull_timeout=self._ray_cfg.image_pull_timeout,
                num_gpus=request.num_gpus,
                env=self._container_env(),
            )

            resolved = self._ray_env(command) if self._runtime.provides_env else envs.Resolved()

            # Three different things `image` can mean, so say which. Under an OCI or
            # apptainer runtime it is what this backend runs; under `ray` it is the
            # key the image map was looked up with, and `resolved.describe()` carries
            # the image that actually starts; under `none`/`native` nothing uses it.
            if self._runtime.isolated:
                image_note = source
            elif self._runtime.env_is_image:
                image_note = f"{source} (declared; mapped below)"
            else:
                image_note = "(not used)"

            logger.info(
                _(
                    "dispatching task to Ray",
                    image=image_note,
                    **request.describe(),
                    **(resolved.describe() if self._runtime.provides_env else {}),
                )
            )
            if resolved.unresolved:
                # Not an error: the cluster image may well supply these, and an unpackaged
                # tool must not fail a run that would otherwise work. But it is the single
                # most likely cause of a later exit 127, so it is worth saying out loud.
                logger.warning(
                    _(
                        "no tool wheel provides these commands; they must already be on the"
                        " workers or the task will fail with exit 127",
                        commands=list(resolved.unresolved),
                    )
                )

            if self._ray_cfg.dispatch == "inprocess":
                # Stay inside `cleanup` for the same reason the Ray path does.
                return self._run_inprocess(logger, job)

            options: dict[str, Any] = {
                **request.options(),
                "max_retries": self._ray_cfg.task_max_retries,
            }
            if resolved.runtime_env:
                options["runtime_env"] = resolved.runtime_env
            if self._ray_cfg.scheduling_strategy.upper() != "DEFAULT":
                options["scheduling_strategy"] = self._ray_cfg.scheduling_strategy.upper()

            remote = ray.remote(ray_job.execute_and_record).options(**options)
            ref = remote.remote(job, placement_path)
            # Stay inside `cleanup` while the task runs: it owns the Apptainer
            # scratch directories the container is mounting.
            exit_code = self._await(logger, terminating, ref, cli_log_filename, placement_path)
        return exit_code

    def _run_inprocess(self, logger: logging.Logger, job: ray_job.ContainerJob) -> int:
        """Run the task command here, without submitting a Ray task.

        For a caller that has already scheduled the workflow graph itself, so that *this*
        process is the Ray task. Dispatching again would submit a task from inside a task:
        correct, but it would double the scheduling and hold two workers' resources for one
        WDL task. This template never selects it (``[ray] dispatch`` stays at ``ray``) but
        the branch is kept so the backend behaves the same here as it does upstream.

        Everything else stays shared with the dispatching path (the argv, the environment,
        the input symlinks, the retry directory handling) so this is a change of *where* the
        command runs, nothing more.
        """
        with ExitStack() as cleanup:
            poll_stderr = cleanup.enter_context(self.poll_stderr_context(logger))
            cleanup.enter_context(self.task_running_context())
            result = ray_job.execute(job)
            # The command has already exited, so this drains what is left.
            poll_stderr()

        if result.chown_error:
            logger.error(
                _(
                    "post-task chown failed; outputs may be unreadable."
                    " Consider [file_io] chown = false",
                    error=result.chown_error,
                )
            )
        logger.info(
            _(
                "task complete (in-process)",
                exit_code=result.exit_code,
                seconds_running=round(result.seconds_running, 1),
            )
        )
        return result.exit_code

    def _chown_argv(self) -> list[str] | None:
        if not (self._runtime.needs_chown and self.cfg.get_bool("file_io", "chown")):
            return None
        if self.cfg.get_bool("task_runtime", "as_user") or (
            os.geteuid() == 0 and os.getegid() == 0
        ):
            return None
        work = os.path.join(
            self.container_dir, f"work{self.try_counter if self.try_counter > 1 else ''}"
        )
        return self._runtime.chown_argv(
            self._exe,
            image=self._ray_cfg.chown_image,
            host_dir=self.host_dir,
            container_dir=self.container_dir,
            target=work,
            uid=os.geteuid(),
            gid=os.getegid(),
        )

    def _container_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if self._sif_cache_dir:
            env["APPTAINER_CACHEDIR"] = self._sif_cache_dir
            env["SINGULARITY_CACHEDIR"] = self._sif_cache_dir
        return env

    def _await(
        self,
        logger: logging.Logger,
        terminating: Callable[[], bool],
        ref: Any,
        cli_log_filename: str,
        placement_path: str,
    ) -> int:
        """Block on the Ray task, tailing its streams and honouring termination."""
        import ray

        cli_logger = logger.getChild(self._runtime.name or "ray")
        with ExitStack() as cleanup:
            poll_stderr = cleanup.enter_context(self.poll_stderr_context(logger))
            poll_cli_log = cleanup.enter_context(
                PygtailLogger(
                    logger,
                    cli_log_filename,
                    lambda msg: cli_logger.info(msg.rstrip()),
                    level=logging.INFO,
                )
            )

            queued_since = time.monotonic()
            running = cleanup.enter_context(ExitStack())
            started = False
            cancelled_at: float | None = None
            while True:
                done, _pending = ray.wait([ref], timeout=1.0)
                if done:
                    break
                if not started and os.path.exists(placement_path):
                    # Only now does the task hold real resources, so this is the
                    # right moment to count it in the status bar's "running".
                    running.enter_context(self.task_running_context())
                    started = True
                    logger.info(
                        _(
                            "task started on Ray worker",
                            seconds_queued=round(time.monotonic() - queued_since, 1),
                            **self._read_placement(placement_path),
                        )
                    )
                if terminating() and cancelled_at is None:
                    logger.notice(  # type: ignore[attr-defined]
                        "cancelling Ray task after termination signal"
                    )
                    ray.cancel(ref)
                    cancelled_at = time.monotonic()
                elif (
                    cancelled_at is not None
                    and time.monotonic() - cancelled_at > _CANCEL_GRACE_SECONDS
                ):
                    ray.cancel(ref, force=True)
                    cancelled_at = time.monotonic()
                poll_stderr()
                poll_cli_log()

            if not started:
                running.enter_context(self.task_running_context())
            result = self._collect(logger, ref)
            # Final drain, so a task that wrote and exited immediately still gets
            # its stderr into the log.
            poll_stderr()
            poll_cli_log()

        if terminating():
            raise Terminated()
        if result.chown_error:
            logger.error(
                _(
                    "post-task chown failed; outputs may be unreadable."
                    " Consider [file_io] chown = false",
                    error=result.chown_error,
                )
            )
        logger.info(
            _(
                "Ray task complete",
                exit_code=result.exit_code,
                node_ip=result.node_ip,
                seconds_running=round(result.seconds_running, 1),
                seconds_pulling=round(result.seconds_pulling, 1),
                pulled_image=result.pulled,
            )
        )
        return result.exit_code

    @staticmethod
    def _read_placement(path: str) -> dict[str, Any]:
        import json

        try:
            with open(path) as src:
                data = json.load(src)
            return {"node_ip": data.get("node_ip", ""), "node_id": data.get("node_id", "")}
        except (OSError, ValueError):
            return {}

    def _collect(self, logger: logging.Logger, ref: Any) -> ray_job.JobResult:
        """Turn a Ray outcome into either a result or the right miniwdl error."""
        import ray

        try:
            result = ray.get(ref)
            assert isinstance(result, ray_job.JobResult)
            return result
        except ray.exceptions.TaskCancelledError:
            raise Terminated() from None
        except ray.exceptions.RayTaskError as exn:
            # Our worker code raised. That is a genuine failure of this task --
            # not an interruption, so it must not consume a `preemptible` try.
            cause = exn.cause if isinstance(getattr(exn, "cause", None), BaseException) else exn
            if isinstance(cause, ray_job.PullFailed) or "PullFailed" in str(exn):
                logger.error(_("image pull failed on Ray worker", error=str(cause)))
                raise DownloadFailed(self._image_ref()[0]) from None
            self.failure_info = {"ray_error": str(exn)}
            logger.error(_("Ray task raised", error=str(exn)))
            raise Error.RuntimeError(f"Ray task failed: {exn}") from None
        except Exception as exn:
            # Node loss, worker crash, raylet death, lost objects: the task never
            # got to fail on its own merits. Reporting it as Interrupted is what
            # routes it to WDL's runtime.preemptible retry budget, the correct
            # behaviour on reclaimed spot capacity.
            name = type(exn).__name__
            if _is_interruption(exn):
                logger.warning(_("Ray worker or node lost", error=name, detail=str(exn)))
                raise Interrupted(f"Ray {name}") from None
            raise


def register_backend(cfg: wdl_config.Loader) -> None:
    """Make the ``ray`` backend selectable without an installed distribution.

    Normally miniwdl finds this class through the ``miniwdl.plugin.container_backend``
    entry point declared in pyproject.toml, but entry points only exist for an
    *installed* package. On a Ray cluster the code frequently arrives as an
    uploaded ``working_dir`` on ``sys.path`` instead, with nothing installed, and
    discovery then silently comes up empty: the run fails with "missing backend
    ray", which points at nothing useful.

    Registering explicitly closes that gap and costs nothing when the entry point
    *did* resolve. Discovery still runs first, so the built-in backends
    (docker_swarm, singularity, podman, udocker) stay available; miniwdl only
    performs discovery when the registry is empty, so seeding it blindly would
    hide them.
    """
    from WDL.runtime import task_container

    with task_container._backends_lock:
        if not task_container._backends:
            for name, plugin in wdl_config.load_plugins(cfg, "container_backend"):
                # load_plugins is typed as yielding callables because it serves
                # several plugin groups; for this group every entry is a
                # TaskContainer subclass.
                task_container._backends[name] = cast("type[task_container.TaskContainer]", plugin)
        task_container._backends["ray"] = RayContainer


def warn_if_not_shared(run_dir: str, logger: logging.Logger) -> None:
    """Warn when the run directory may not be visible from every node.

    miniwdl passes files between tasks by path: task B reads task A's outputs from
    A's working directory. So a run directory that only exists on the driver's node
    produces a workflow that starts fine and fails at the *second* task, with a
    missing-input error naming a path that plainly exists, one of the more
    expensive ways to lose an hour.

    The check does not fail the run, because it cannot tell the difference between a
    genuinely local setup and a shared mount at a path this list does not know. It is
    deliberately noisy in the ambiguous case instead.

    Node count is only used to pick the severity, never to skip the check. An
    autoscaling cluster with ``min_nodes: 1``, which is this template's shape,
    has one node at startup and three by the time the assemblies dispatch, so a
    check gated on "more than one node right now" is silent precisely when it is
    needed. Earlier it was gated that way.
    """
    import ray

    if any(os.path.abspath(run_dir).startswith(p) for p in SHARED_STORAGE_PREFIXES):
        return

    nodes = [n for n in ray.nodes() if n.get("Alive")] if ray.is_initialized() else []
    multi_node = len(nodes) > 1

    message = _(
        "run directory is not on a recognized shared filesystem; any task scheduled off"
        " the driver's node will fail to see its inputs. This is safe only if the cluster"
        " will stay single-node for the whole run, or if this path is a shared mount under"
        " another name",
        run_dir=run_dir,
        nodes_alive=len(nodes) or "unknown",
        recognized_prefixes=list(SHARED_STORAGE_PREFIXES),
    )
    if multi_node:
        # Already multi-node: this is not a risk, it is a defect waiting for the
        # scheduler to place one task elsewhere.
        logger.error(message)
    else:
        logger.warning(message)
