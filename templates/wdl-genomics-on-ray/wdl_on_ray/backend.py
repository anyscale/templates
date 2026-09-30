"""A miniwdl container backend that runs each WDL task as a Ray task."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, suppress
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

SHARED_STORAGE_PREFIXES = (
    "/mnt/cluster_storage",
    "/mnt/shared_storage",
    "/mnt/user_storage",
    "/mnt/shared",
    "/mnt/nfs",
    "/mnt/efs",
    "/mnt/fsx",
)

_CANCEL_GRACE_SECONDS = 15.0

#: miniwdl's own "no limit"; not a big number, which could reach a WDL 1.2 task's command as cpu.
NO_LIMIT = -1

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
    # isinstance: ObjectLostError subclasses (ObjectReconstructionFailedError) mean node loss too.
    from ray import exceptions as ray_exceptions

    classes = tuple(
        cls
        for cls in (getattr(ray_exceptions, name, None) for name in _INTERRUPTION_ERRORS)
        if isinstance(cls, type)
    )
    return (classes and isinstance(exn, classes)) or type(exn).__name__ in _INTERRUPTION_ERRORS


_connected = False

#: Pickled by value into every task, so both must stay stdlib-only.
_WORKER_MODULES = (ray_job, runtimes)


def _pickle_worker_modules_by_value() -> None:
    # By value, so workers need only Ray: importing wdl_on_ray fails from an uploaded working_dir,
    # and a runtime_env py_modules clashes with a job's own py_modules.
    from ray.cloudpickle import register_pickle_by_value

    for module in _WORKER_MODULES:
        register_pickle_by_value(module)


def connect(ray_cfg: ray_config.RayConfig, logger: logging.Logger) -> None:
    """Connect to Ray. Idempotent."""
    global _connected
    import ray

    _pickle_worker_modules_by_value()

    if ray.is_initialized():
        _connected = True
        return

    kwargs: dict[str, Any] = {
        "namespace": ray_cfg.namespace,
        "ignore_reinit_error": True,
        # miniwdl owns the console; worker stdout would drown out the per-task logs.
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
            address=ray.get_runtime_context().gcs_address,
            nodes=len([n for n in ray.nodes() if n.get("Alive")]),
            cluster_cpus=int(ray.cluster_resources().get("CPU", 0)),
            cluster_gpus=int(ray.cluster_resources().get("GPU", 0)),
        )
    )


def find_cluster(ray_cfg: ray_config.RayConfig) -> str | None:
    """The address :func:`connect` would join, found without joining; None means a new local one."""
    ray = sys.modules.get("ray")  # not imported, then not initialized either
    if ray is not None and ray.is_initialized():
        return str(ray.get_runtime_context().gcs_address)
    for address in (ray_cfg.address, os.environ.get("RAY_ADDRESS", "")):
        if address and address != "auto":
            return None if address == "local" else address
    try:
        with open(_ray_address_file()) as src:
            return src.read().strip() or None
    except OSError:
        return None


def _ray_address_file() -> str:
    # Ray's temp-dir rule, as in ray._common.utils.get_default_system_temp_dir.
    if "RAY_TMPDIR" in os.environ:
        base = os.environ["RAY_TMPDIR"]
    elif sys.platform.startswith("linux") and "TMPDIR" in os.environ:
        base = os.environ["TMPDIR"]
    elif sys.platform.startswith(("darwin", "linux")):
        base = "/tmp"
    else:
        base = tempfile.gettempdir()
    return os.path.join(base, "ray", "ray_current_cluster")


def task_ceiling(nodes: list[dict[str, Any]], limit_source: str) -> tuple[float, float] | None:
    """``(cpu, memory)`` one task can have, from ``ray.nodes()``; None if no node can run one."""
    # Every task asks for at least one CPU, so a node without CPUs (a ``CPU: 0`` head) runs none.
    usable = [
        resources
        for resources in (n.get("Resources", {}) for n in nodes if n.get("Alive"))
        if resources.get("CPU", 0) > 0
    ]
    if not usable:
        return None
    combine = sum if limit_source == "cluster" else max
    return combine(r["CPU"] for r in usable), combine(r.get("memory", 0.0) for r in usable)


def limits_from(
    found: tuple[float, float] | None, ray_cfg: ray_config.RayConfig
) -> dict[str, int]:
    """miniwdl's resource limits from a measured ``(cpu, memory)``; ``[ray] max_*`` settings win."""
    cpu, mem = found or (0.0, 0.0)
    limits = {
        "cpu": max(1, int(cpu)) if cpu > 0 else NO_LIMIT,
        "mem_bytes": int(mem) if mem >= 1 else NO_LIMIT,
    }
    if ray_cfg.max_cpu > 0:
        limits["cpu"] = ray_cfg.max_cpu
    if ray_cfg.max_memory_bytes > 0:
        limits["mem_bytes"] = ray_cfg.max_memory_bytes
    return limits


class RayContainer(SubprocessBase):
    """Dispatch WDL task containers onto a Ray cluster."""

    _ray_cfg: ray_config.RayConfig
    _runtime: runtimes.ContainerRuntime
    _exe: list[str]
    _limits: dict[str, int] | None = None
    _limits_lock = threading.Lock()
    _sif_cache_dir: str | None = None
    _checked_shared_run_dir = False

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
        shown = {key: "none" if value == NO_LIMIT else value for key, value in limits.items()}

        logger.notice(  # type: ignore[attr-defined]
            _(
                "Ray container backend initialized",
                container_runtime=cls._runtime.name,
                exe=" ".join(cls._exe) or "(none)",
                task_cpu_limit=shown["cpu"],
                task_mem_bytes_limit=shown["mem_bytes"],
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

        logger.info(
            "container_runtime=ray requires the run directory to be visible at the same path"
            " inside each task image; a task that cannot see its inputs is the symptom when"
            " it is not"
        )

    @classmethod
    def _warn_unless_wheelhouse_usable(cls, logger: logging.Logger) -> None:
        wheel_dir = cls._ray_cfg.tool_wheel_dir
        # Empty means "resolve from an index" on purpose; a URL cannot be inspected.
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
        # Ray's pip/uv plugins clone the base venv, so a base without pip gives a clone without it.
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
        # auto probes the driver's node only; on a heterogeneous cluster name the runtime.
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
        """The ceiling miniwdl clamps each task's cpu/memory to: one node, not the cluster."""
        with cls._limits_lock:
            if cls._limits is not None:
                return cls._limits

            ray_cfg = getattr(cls, "_ray_cfg", None) or ray_config.load(cfg)
            found = cls._probe_limits(ray_cfg, logger)
            cls._limits = limits_from(found, ray_cfg)

            unset = [
                (field, setting)
                for field, setting, key in (
                    ("runtime.cpu", "--max-cpu", "cpu"),
                    ("runtime.memory", "[ray] max_memory_bytes", "mem_bytes"),
                )
                if cls._limits[key] == NO_LIMIT
            ]
            if found is None and unset:
                logger.warning(
                    _(
                        "no node that can run tasks is up yet, so there is no per-task ceiling:"
                        " a task asking for more than any worker has waits without an error."
                        " Set the worker shape to clamp",
                        unclamped=[field for field, _setting in unset],
                        set_to_clamp=[setting for _field, setting in unset],
                    )
                )
            return cls._limits

    @classmethod
    def _probe_limits(
        cls, ray_cfg: ray_config.RayConfig, logger: logging.Logger
    ) -> tuple[float, float] | None:
        if ray_cfg.limit_source == "local":
            import multiprocessing

            import psutil

            return multiprocessing.cpu_count(), psutil.virtual_memory().total

        import ray

        connect(ray_cfg, logger)
        return task_ceiling(ray.nodes(), ray_cfg.limit_source)

    def __init__(self, cfg: wdl_config.Loader, run_id: str, host_dir: str) -> None:
        super().__init__(cfg, run_id, host_dir)
        if not self._runtime.isolated:
            # No filesystem namespace, so container paths are host paths.
            self.container_dir = self.host_dir
        if not RayContainer._checked_shared_run_dir:
            RayContainer._checked_shared_run_dir = True
            warn_if_not_shared(host_dir, logging.getLogger("wdl-on-ray"))

    def process_runtime(self, logger: logging.Logger, runtime_eval: dict[str, Any]) -> None:
        """Also read Cromwell's ``gpuCount``/``gpuType``/``disks`` and the ``ray_*`` keys."""
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
        return self._runtime.name

    @property
    def cli_exe(self) -> list[str]:
        return list(self._exe)

    def reset(self, logger: logging.Logger) -> None:
        """Prepare a retry's working directory."""
        super().reset(logger)
        if self._runtime.isolated:
            return
        # The rendered command says .../work and there is no bind mount, so symlink it to this try.
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
        # native needs the rendered command: its environment comes from the executables it runs.
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

    def _image_ref(self) -> tuple[str, str]:
        source = self.runtime_values.get(
            "docker", self.cfg.get_dict("task_runtime", "defaults")["docker"]
        )
        return source, self._runtime.image_ref(source, cache_dir=self._sif_cache_dir)

    def _link_inputs(self, logger: logging.Logger) -> None:
        # Stands in for bind mounts; symlinks, as inputs run to tens of GB on shared storage.
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
        # Apptainer's in-memory session directory is small and easily overrun.
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
        shell = self.cfg.get("task_runtime", "command_shell")
        if self._runtime.isolated:
            return ["/bin/sh", "-c", f"{shell} ../command >> ../stdout.txt 2>> ../stderr.txt"]
        # No mounts, so the redirections name the real host paths (stdout2.txt on a retry too).
        return [
            "/bin/sh",
            "-c",
            f"{shell} {shlex.quote(os.path.join(self.host_dir, 'command'))}"
            f" >> {shlex.quote(self.host_stdout_txt())}"
            f" 2>> {shlex.quote(self.host_stderr_txt())}",
        ]

    def _write_command_file(self, command: str) -> str:
        # Exported in the script, as miniwdl does: no argv length limit, no --env-file quoting.
        path = os.path.join(self.host_dir, "command")
        with open(path, "w") as outfile:
            for key, value in self.runtime_values.get("env", {}).items():
                outfile.write(f"export {key}={shlex.quote(value)}\n")
            outfile.write(command)
        return path

    def _run(self, logger: logging.Logger, terminating: Callable[[], bool], command: str) -> int:
        import ray

        with ExitStack() as cleanup:
            request = self._ray_request()
            source, image_ref = self._image_ref()
            self._write_command_file(command)

            if self._runtime.isolated:
                argv = self._build_argv(logger, cleanup, image_ref, self._entry())
            else:
                for stream in (self.host_stdout_txt(), self.host_stderr_txt()):
                    if not os.path.exists(stream):
                        self.touch_mount_point(stream)
                self._link_inputs(logger)
                argv = self._build_argv(logger, cleanup, image_ref, self._entry())

            cli_log_filename = os.path.join(self.host_dir, f"{self.cli_name}.log.txt")
            placement_path = os.path.join(self.host_dir, "ray_placement.json")

            # _await takes this file's appearance as the start, and a retry reuses this directory,
            # so remove the previous attempt's copy before submitting.
            with suppress(FileNotFoundError):
                os.unlink(placement_path)

            # Created here: a task that never starts would leave no file, and Pygtail's
            # FileNotFoundError would bury the real error.
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
                # Not an error: the cluster image may supply them.
                logger.warning(
                    _(
                        "no tool wheel provides these commands; they must already be on the"
                        " workers or the task will fail with exit 127",
                        commands=list(resolved.unresolved),
                    )
                )

            if self._ray_cfg.dispatch == "inprocess":
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
            # Inside `cleanup`: it owns the Apptainer scratch directories the container mounts.
            exit_code = self._await(logger, terminating, ref, cli_log_filename, placement_path)
        return exit_code

    def _run_inprocess(self, logger: logging.Logger, job: ray_job.ContainerJob) -> int:
        # For a caller that already scheduled the workflow graph, so this process is the Ray task.
        with ExitStack() as cleanup:
            poll_stderr = cleanup.enter_context(self.poll_stderr_context(logger))
            cleanup.enter_context(self.task_running_context())
            result = ray_job.execute(job)
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
                    # Only now does the task hold resources, so count it as running.
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
        import ray

        try:
            result = ray.get(ref)
            assert isinstance(result, ray_job.JobResult)
            return result
        except ray.exceptions.TaskCancelledError:
            raise Terminated() from None
        except ray.exceptions.RayTaskError as exn:
            # Our worker code raised: a real failure, which must not spend a preemptible try.
            cause = exn.cause if isinstance(getattr(exn, "cause", None), BaseException) else exn
            if isinstance(cause, ray_job.PullFailed) or "PullFailed" in str(exn):
                logger.error(_("image pull failed on Ray worker", error=str(cause)))
                raise DownloadFailed(self._image_ref()[0]) from None
            self.failure_info = {"ray_error": str(exn)}
            logger.error(_("Ray task raised", error=str(exn)))
            raise Error.RuntimeError(f"Ray task failed: {exn}") from None
        except Exception as exn:
            # Node loss, worker crash, lost objects: Interrupted spends runtime.preemptible.
            name = type(exn).__name__
            if _is_interruption(exn):
                logger.warning(_("Ray worker or node lost", error=name, detail=str(exn)))
                raise Interrupted(f"Ray {name}") from None
            raise


def register_backend(cfg: wdl_config.Loader) -> None:
    """Register the ``ray`` backend, which entry points miss on an uploaded working_dir."""
    from WDL.runtime import task_container

    # miniwdl only discovers plugins into an empty registry, so discover before adding ours.
    with task_container._backends_lock:
        if not task_container._backends:
            for name, plugin in wdl_config.load_plugins(cfg, "container_backend"):
                task_container._backends[name] = cast("type[task_container.TaskContainer]", plugin)
        task_container._backends["ray"] = RayContainer


def warn_if_not_shared(run_dir: str, logger: logging.Logger) -> None:
    """Warn when the run directory may not be visible from every node."""
    import ray

    if any(os.path.abspath(run_dir).startswith(p) for p in SHARED_STORAGE_PREFIXES):
        return

    # Node count sets severity only: an autoscaling cluster starts with one node.
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
        logger.error(message)
    else:
        logger.warning(message)
