"""Worker side: run one Nextflow job script inside a Ray task.

Pickled by value onto workers, so module-level imports must stay in the standard library.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field, replace

# Written when the task starts: Ray has no start callback, and this splits queue from run time.
PLACEMENT_FILENAME = ".nf-ray.placement"

# Not .command.err, which belongs to the tool and is parsed by MultiQC.
LOG_FILENAME = ".nf-ray.log"


@dataclass(frozen=True)
class TaskSpec:
    """A fully resolved job-script invocation, ready to ship to a Ray worker."""

    task_id: int
    name: str
    work_dir: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    num_gpus: float = 0.0
    request: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    """Outcome of one job script, plus placement for the Gantt chart."""

    task_id: int
    exit_code: int
    node_id: str = ""
    node_ip: str = ""
    seconds_running: float = 0.0
    started_epoch: float = 0.0


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    # Signal the whole group: a surviving bwa-mem2 would hold cores on a node Ray thinks idle.
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


def execute_and_record(spec: TaskSpec) -> TaskResult:
    """Ray task body: record placement, then run the job script."""
    import ray  # noqa: PLC0415

    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    node_ip = ray.util.get_node_ip_address()
    started = time.time()

    placement = os.path.join(spec.work_dir, PLACEMENT_FILENAME)
    try:
        os.makedirs(spec.work_dir, exist_ok=True)
        with open(placement, "w") as out:
            json.dump(
                {
                    "task_id": spec.task_id,
                    "name": spec.name,
                    "node_id": node_id,
                    "node_ip": node_ip,
                    "pid": os.getpid(),
                    "started_epoch": started,
                    "request": spec.request,
                },
                out,
            )
    except OSError:
        # Observability only; a broken filesystem fails the task with a better message anyway.
        pass

    return replace(
        execute(spec),
        node_id=node_id,
        node_ip=node_ip,
        started_epoch=started,
    )


def execute(spec: TaskSpec) -> TaskResult:
    """Run one job script to completion on this node."""
    started = time.monotonic()
    log_path = os.path.join(spec.work_dir, LOG_FILENAME)

    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            spec.argv,
            # The task's own streams go to .command.out/.err; only bash's own errors land here.
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=spec.work_dir,
            env={**os.environ, **spec.env},
            start_new_session=True,
        )
        try:
            exit_code = proc.wait()
        except BaseException:
            # Covers ray.cancel() (raised into the task) and worker teardown.
            _terminate(proc)
            raise

    return TaskResult(
        task_id=spec.task_id,
        exit_code=exit_code,
        seconds_running=time.monotonic() - started,
    )


# Here, not in cli.py, so it pickles by value: the probed image has no nf_ray to import.
def probe(work_dir: str, marker: str) -> dict[str, object]:
    """Run inside a candidate task image: report its Ray/Python versions and work-dir access."""
    import platform  # noqa: PLC0415
    import socket  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import ray  # noqa: PLC0415

    result: dict[str, object] = {
        "python": platform.python_version(),
        "ray": ray.__version__,
        "host": socket.gethostname(),
        "node_id": ray.get_runtime_context().get_node_id(),
        "work_dir_visible": os.path.isdir(work_dir),
        "work_dir_writable": False,
        "readback_ok": False,
        "error": "",
    }
    if not result["work_dir_visible"]:
        result["error"] = f"{work_dir} does not exist inside the task image"
        return result
    try:
        with tempfile.NamedTemporaryFile("w", dir=work_dir, prefix="probe-", delete=False) as fh:
            fh.write(marker)
            path = fh.name
        result["work_dir_writable"] = True
        with open(path) as fh:
            result["readback_ok"] = fh.read() == marker
        os.unlink(path)
    except OSError as exn:
        result["error"] = f"{type(exn).__name__}: {exn}"
    return result
