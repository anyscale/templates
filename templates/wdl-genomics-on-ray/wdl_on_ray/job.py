"""Worker side of the dispatch. Pickled by value: stdlib and runtimes only, Ray imported lazily."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field, replace

from wdl_on_ray import runtimes

#: Images this worker already made local. Ray reuses worker processes, so this skips the lock too.
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
    """Make ``job.image_ref`` local, one pull per node at a time. Returns ``(pulled, seconds)``."""
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
    # The CLI runs in its own session; signal the group so no orphaned container holds the CPUs.
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
    """Ray task body: record where it landed, which tells the driver it started, then run it."""
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
            # ray.cancel() (KeyboardInterrupt in the task) or worker teardown.
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
    """Runs inside a candidate task image: do versions match, and is the run directory shared?"""
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
