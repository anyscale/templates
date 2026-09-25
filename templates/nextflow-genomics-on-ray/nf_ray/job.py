"""The worker side of the dispatch: run one Nextflow job script on a Ray node.

Everything here executes inside a Ray task, so it is deliberately policy-free.
The daemon resolves every decision into a :class:`TaskSpec` of plain strings;
this module records where it landed, runs the script, and reports the exit code.

This module is serialized **by value** into every Ray task (see
:func:`nf_ray.daemon._pickle_worker_modules_by_value`), which is what frees the
worker nodes from needing this package installed at all. One constraint follows,
and any change here must preserve it: **module-level imports stay in the standard
library**, and Ray is imported lazily inside the one function that needs it.

Two things this module deliberately does *not* do.

It does not build a container invocation. Nextflow already does that: when
``docker.enabled`` or ``apptainer.enabled`` is set, the container command is
written into ``.command.run`` before the executor ever sees it. On a node with a
container runtime, podman, Apptainer and Singularity would therefore stay
Nextflow's business; this template's image has none. Only Ray's ``image_uri`` is
ours, because that is a Ray-level concept Nextflow has no vocabulary for, and it
is handled on the driver in :mod:`nf_ray.envs`.

It does not write ``.exitcode``. Nextflow's wrapper writes that itself from an
``EXIT`` trap, and it is the authority on the task's own exit status. The daemon
writes it only when this function never got to return -- see :mod:`nf_ray.errors`.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field, replace

#: Written into the task's work directory the moment the task starts running,
#: which is how the driver learns the task stopped *queueing*. Ray offers no
#: started-callback, and the queue/run distinction is how to tell "my cluster is
#: too small" from "my task is slow". The daemon notices the file on its next
#: status poll, and the notebook's Gantt chart draws the queue time from that.
PLACEMENT_FILENAME = ".nf-ray.placement"

#: Where this module's own diagnostics go. Never the task's `.command.err`, which
#: belongs to the tool and gets parsed by MultiQC and by users.
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
    """The resource request, for the record written beside the task."""


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
    """Tear down the job script *and* whatever it spawned.

    The script is started in its own session, so signalling the process group
    reaches the tool's children too. Without that, cancelling a pipeline can
    leave a ``bwa-mem2`` holding 16 cores on a node Ray believes is idle, and the
    autoscaler will not reclaim the node while it looks busy.
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


def execute_and_record(spec: TaskSpec) -> TaskResult:
    """Ray task body: record placement, then run the job script.

    Lives here rather than in :mod:`nf_ray.daemon` so that the code shipped to a
    Ray worker pulls in only this module -- pure stdlib -- which is what lets the
    daemon serialize it *by value* and keep worker nodes free of any dependency
    on this package being installed.
    """
    import ray  # noqa: PLC0415 -- lazy on purpose; see the module docstring

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
        # A placement record is observability, not correctness. If the shared
        # filesystem is unhappy the task itself is about to fail with a much
        # better message than anything we could raise here.
        pass

    return replace(
        execute(spec),
        node_id=node_id,
        node_ip=node_ip,
        started_epoch=started,
    )


def execute(spec: TaskSpec) -> TaskResult:
    """Run one job script to completion on this node.

    ``CUDA_VISIBLE_DEVICES`` needs no handling: Ray sets it in the worker
    process, and the script inherits the environment.
    """
    started = time.monotonic()
    log_path = os.path.join(spec.work_dir, LOG_FILENAME)

    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            spec.argv,
            # The wrapper redirects the task's own streams to .command.out and
            # .command.err. What lands here is only what bash itself says --
            # a missing interpreter, a permissions problem.
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


def probe(work_dir: str, marker: str) -> dict[str, object]:
    """Runs *inside* a candidate task image. Stdlib plus Ray only.

    Answers the two questions that decide whether ``ext.image`` can work on a
    given cluster: does the image agree with the driver about Ray and Python
    versions, and can it read and write the pipeline's work directory at the same
    absolute path the driver uses? A "no" to the second is fatal, because
    Nextflow stages inputs by path -- a task that cannot see the work directory
    cannot consume the previous process's output.

    Lives here rather than beside its CLI command so that
    ``register_pickle_by_value`` covers it. Serialized by reference, the probe
    would need ``nf_ray`` importable *inside the image being probed*, and the
    small per-process images this feature exists to support do not have it -- so
    the probe would fail to deserialize and report the image unusable for
    entirely the wrong reason.
    """
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
