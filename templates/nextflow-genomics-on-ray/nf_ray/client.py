"""Stdlib-only client for the daemon's unix socket.

This module is on the hot path and is the reason the daemon exists in the shape it
does. Nextflow spawns one process per submit; on a pipeline with a few hundred
tasks, a client that imported Ray to talk to the cluster would spend roughly a
second per call doing nothing but importing. So nothing here imports Ray, or
anything else outside the standard library -- and :mod:`nf_ray.cli` is careful to
route the hot subcommands through this module without touching
:mod:`nf_ray.daemon`.

Keep it that way. The cheap way to break this template's throughput is a
convenience import at the top of this file.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import subprocess
import sys
import time

#: Long enough that a busy daemon reaping several hundred refs does not look dead,
#: short enough that a genuinely wedged daemon fails the run instead of hanging it.
DEFAULT_TIMEOUT = 120.0

#: How long to wait for a freshly started daemon to answer. Dominated by
#: `ray.init()` attaching to the cluster.
START_TIMEOUT = 180.0


class DaemonError(RuntimeError):
    """The daemon refused a request or could not be reached."""


def request(socket_path: str, payload: dict, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Send one request, return the decoded reply."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(socket_path)
            sock.sendall((json.dumps(payload) + "\n").encode())
            chunks = []
            while not (chunks and chunks[-1].endswith(b"\n")):
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except OSError as exn:
        raise DaemonError(f"could not reach the nf-ray daemon at {socket_path}: {exn}") from None

    raw = b"".join(chunks).strip()
    if not raw:
        raise DaemonError(f"the nf-ray daemon at {socket_path} closed without replying")
    try:
        reply = json.loads(raw)
    except json.JSONDecodeError:
        raise DaemonError(f"unparseable reply from the nf-ray daemon: {raw[:400]!r}") from None
    if not reply.get("ok"):
        raise DaemonError(str(reply.get("error", "unknown error")))
    return reply


def ping(socket_path: str, timeout: float = 5.0) -> bool:
    """Whether a daemon is answering."""
    if not os.path.exists(socket_path):
        return False
    try:
        request(socket_path, {"op": "ping"}, timeout=timeout)
    except DaemonError:
        return False
    return True


def ensure_daemon(socket_path: str, work_dir: str, timeout: float = START_TIMEOUT) -> None:
    """Start a daemon if none is answering, and wait until one is.

    Guarded by a lock file, because Nextflow submits concurrently: without it, a
    pipeline whose first wave is twenty tasks would race twenty daemons into
    existence, nineteen of which would fail to bind and take their tasks with
    them.

    A stale socket file left by a killed daemon is unlinked here rather than at
    startup, so the common case (nothing running, no socket) needs no cleanup and
    the uncommon one is handled where it is detectable.
    """
    if ping(socket_path):
        return

    lock_path = socket_path + ".lock"
    os.makedirs(os.path.dirname(socket_path) or ".", exist_ok=True)
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            # Another process may have started one while we waited for the lock.
            if ping(socket_path):
                return
            if os.path.exists(socket_path):
                with contextlib.suppress(OSError):
                    os.unlink(socket_path)
            _spawn(socket_path, work_dir)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if ping(socket_path):
                    return
                time.sleep(0.25)
            raise DaemonError(
                f"the nf-ray daemon did not come up within {timeout:.0f}s.\n"
                f"Its log is at {os.path.join(work_dir, '.nf-ray.daemon.log')}\n"
                "The usual cause is that Ray is not reachable: check `ray status`, and\n"
                "that NF_RAY_ADDRESS is 'auto' on a cluster or 'local' off one."
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _spawn(socket_path: str, work_dir: str) -> None:
    """Start the daemon detached, with its log on disk.

    ``start_new_session`` matters: without it the daemon joins Nextflow's process
    group and dies with the first ``nf-ray submit`` that spawned it, taking every
    task it owns.
    """
    log_path = os.path.join(work_dir, ".nf-ray.daemon.log")
    os.makedirs(work_dir, exist_ok=True)
    with open(log_path, "ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "nf_ray.cli", "daemon", "--socket", socket_path],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "NF_RAY_WORK_DIR": work_dir, "NF_RAY_SOCKET": socket_path},
        )


def submit(socket_path: str, script: str, work_dir: str) -> int:
    reply = request(socket_path, {"op": "submit", "script": script, "work_dir": work_dir})
    return int(reply["task_id"])


def status(socket_path: str) -> list[tuple[int, str]]:
    reply = request(socket_path, {"op": "status"})
    return [(int(row[0]), str(row[1])) for row in reply.get("rows", [])]


def kill(socket_path: str, ids: list[int]) -> None:
    request(socket_path, {"op": "kill", "ids": list(ids)})


def info(socket_path: str) -> dict:
    return dict(request(socket_path, {"op": "info"})["info"])


def shutdown(socket_path: str) -> None:
    with contextlib.suppress(DaemonError):
        request(socket_path, {"op": "shutdown"}, timeout=30.0)
