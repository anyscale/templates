"""The worker side of the dispatch: run one container invocation on a Ray node.

Everything in this module executes inside a Ray task, so it is deliberately
policy-free and dependency-light. The driver (:mod:`wdl_on_ray.backend`) resolves
every decision (which image, which mounts, which flags) into a
:class:`ContainerJob` of plain strings; this module only pulls the image if
needed, spawns the process, and reports the exit code.

This module is serialized **by value** into every Ray task (see
:func:`wdl_on_ray.backend._pickle_worker_modules_by_value`), which is what frees the
workers from needing this package installed. Two constraints follow, and any change
here must preserve both: keep the module-level imports to the standard library plus
:mod:`wdl_on_ray.runtimes`, and import Ray lazily inside the one function that
needs it.

Two further details:

* Image pulls are serialized per node. A scatter of 40 tasks over the same
  image would otherwise start 40 concurrent pulls on each node it lands on. A
  ``flock``-based lock in a node-local directory collapses that to one, and an
  in-process cache short-circuits it entirely for the (common) case of a reused
  Ray worker process.
* Standard streams go to the shared filesystem, not through Ray. miniwdl
  tails ``stderr.txt`` from the driver to produce live task logs, so the
  container writes there directly and the driver's tailing keeps working
  unchanged. This is also why the run directory has to be on storage visible
  from every node.
"""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field, replace

from wdl_on_ray import runtimes

#: Images this worker process has already made local. Ray reuses worker
#: processes between tasks, so this saves even the lock acquisition.
_PULLED: set[str] = set()


@dataclass(frozen=True)
class ContainerJob:
    """A fully resolved container invocation, ready to ship to a Ray worker."""

    runtime_name: str
    run_argv: list[str]
    cwd: str
    cli_log_path: str
    exe: list[str] = field(default_factory=list)
    image_ref: str | None = None
    image_source: str | None = None
    pull_argv: list[str] | None = None
    image_present_argv: list[str] | None = None
    chown_argv: list[str] | None = None
    pull_lock_dir: str = "/tmp/wdl-on-ray/pull-locks"
    pull_timeout: int = 3600
    num_gpus: float = 0.0
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JobResult:
    """Outcome of one container invocation, plus placement info for logging."""

    exit_code: int
    node_id: str = ""
    node_ip: str = ""
    seconds_pulling: float = 0.0
    seconds_running: float = 0.0
    pulled: bool = False
    chown_error: str | None = None


class PullFailed(RuntimeError):
    """Raised when the image could not be made available on this node."""


def _run_quiet(argv: list[str], env: dict[str, str], timeout: int | None = None) -> int:
    return subprocess.run(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, **env},
        timeout=timeout,
        check=False,
    ).returncode


def _lock_path(lock_dir: str, image_ref: str) -> str:
    digest = hashlib.sha256(image_ref.encode()).hexdigest()[:32]
    return os.path.join(lock_dir, f"{digest}.lock")


def ensure_image(job: ContainerJob) -> tuple[bool, float]:
    """Make ``job.image_ref`` available on this node.

    Returns ``(pulled, seconds_spent)``; ``pulled`` is False when the image was
    already present or the runtime needs no pull at all.
    """
    if job.pull_argv is None or job.image_ref is None:
        return False, 0.0
    if job.image_ref in _PULLED:
        return False, 0.0

    import fcntl  # POSIX-only, and only needed on the worker

    started = time.monotonic()
    os.makedirs(job.pull_lock_dir, exist_ok=True)
    with open(_lock_path(job.pull_lock_dir, job.image_ref), "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if job.image_present_argv and _run_quiet(job.image_present_argv, job.env) == 0:
                _PULLED.add(job.image_ref)
                return False, time.monotonic() - started
            described = job.image_source or job.image_ref
            try:
                code = _run_quiet(job.pull_argv, job.env, timeout=job.pull_timeout)
            except subprocess.TimeoutExpired:
                raise PullFailed(
                    f"timed out after {job.pull_timeout}s pulling {described}"
                ) from None
            if code != 0:
                raise PullFailed(
                    f"`{' '.join(job.pull_argv)}` exited {code} while pulling {described}"
                )
            _PULLED.add(job.image_ref)
            return True, time.monotonic() - started
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    """Tear down the container CLI *and* whatever it spawned.

    The CLI is started in its own session, so signalling the process group
    reaches the container runtime's children too; without that, cancelling a
    workflow can leave orphaned containers holding the node's CPUs.
    """
    for sig, grace in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            proc.send_signal(sig)
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def execute_and_record(job: ContainerJob, placement_path: str) -> JobResult:
    """Ray task body: record where this landed, then run the container.

    ``placement_path`` is how the driver learns the task stopped queueing and
    actually started; Ray offers no callback for that, and the distinction is
    what separates "my cluster is too small" from "my task is slow".

    Lives here, and not in :mod:`wdl_on_ray.backend`, so that the code shipped
    to a Ray worker pulls in only this module and :mod:`wdl_on_ray.runtimes` --
    both pure-stdlib. That is what lets the backend serialize them *by value* and
    keep the workers free of any dependency on this package being installed.
    """
    import json

    import ray

    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    node_ip = ray.util.get_node_ip_address()
    with open(placement_path, "w") as out:
        json.dump({"node_id": node_id, "node_ip": node_ip, "pid": os.getpid()}, out)
    return replace(execute(job), node_id=node_id, node_ip=node_ip)


def execute(job: ContainerJob) -> JobResult:
    """Run one container invocation to completion on this node."""
    pulled, seconds_pulling = ensure_image(job)

    argv = job.run_argv
    if job.num_gpus:
        argv = runtimes.expand_gpu_args(
            argv,
            runtimes.get(job.runtime_name),
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            job.num_gpus,
        )

    started = time.monotonic()
    os.makedirs(os.path.dirname(job.cli_log_path), exist_ok=True)
    with open(job.cli_log_path, "ab") as cli_log:
        proc = subprocess.Popen(
            argv,
            stdout=cli_log,
            stderr=subprocess.STDOUT,
            cwd=job.cwd,
            env={**os.environ, **job.env},
            start_new_session=True,
        )
        try:
            exit_code = proc.wait()
        except BaseException:
            # Covers ray.cancel() (KeyboardInterrupt in the task) as well as
            # worker teardown.
            _terminate(proc)
            raise

    chown_error = None
    if job.chown_argv is not None:
        code = _run_quiet(job.chown_argv, job.env, timeout=600)
        if code != 0:
            chown_error = f"`{' '.join(job.chown_argv)}` exited {code}"

    return JobResult(
        exit_code=exit_code,
        seconds_pulling=seconds_pulling,
        seconds_running=time.monotonic() - started,
        pulled=pulled,
        chown_error=chown_error,
    )


def probe_image(run_dir: str, marker: str) -> dict[str, object]:
    """Runs *inside* a candidate task image. Stdlib plus Ray only.

    Answers the two questions that decide whether ``container_runtime = ray`` can work
    on a given cluster: does the nested container agree with the driver about Ray and
    Python versions, and can it read and write the shared run directory at the same
    absolute path the driver uses? A "no" to the second is fatal to the mode, because
    miniwdl passes files between tasks by path, so a task that cannot see the run
    directory cannot consume the previous task's outputs.

    Lives here rather than next to its CLI command so that ``register_pickle_by_value``
    covers it. Serialized by reference, the probe would need ``wdl_on_ray`` and miniwdl
    importable *inside the image being probed* -- which the small per-task images
    tools/BUILDING.md tells you to build deliberately do not have, so the probe would
    fail to deserialize and report the candidate image unusable for the wrong reason.
    """
    import platform
    import socket
    import tempfile

    import ray

    result: dict[str, object] = {
        "python": platform.python_version(),
        "ray": ray.__version__,
        "host": socket.gethostname(),
        "node_id": ray.get_runtime_context().get_node_id(),
        "run_dir_visible": os.path.isdir(run_dir),
        "run_dir_writable": False,
        "readback_ok": False,
        "error": "",
    }
    if not result["run_dir_visible"]:
        result["error"] = f"{run_dir} does not exist inside the task image"
        return result
    try:
        with tempfile.NamedTemporaryFile("w", dir=run_dir, prefix="probe-", delete=False) as fh:
            fh.write(marker)
            path = fh.name
        result["run_dir_writable"] = True
        with open(path) as fh:
            result["readback_ok"] = fh.read() == marker
        os.unlink(path)
    except OSError as exn:
        result["error"] = f"{type(exn).__name__}: {exn}"
    return result
