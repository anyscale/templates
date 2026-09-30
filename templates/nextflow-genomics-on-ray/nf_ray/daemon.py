"""The long-lived owner of every Ray task in a run: a Ray task dies with its submitter, and each
``nf-ray submit`` exits at once. Also writes ``.exitcode`` when Ray fails around a task.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field

from nf_ray import directives as directives_mod
from nf_ray import envs, errors, job, resources
from nf_ray.config import Config, shared_storage_warning

log = logging.getLogger("nf_ray.daemon")

PENDING = "PENDING"
RUNNING = "RUNNING"
DONE = "DONE"
ERROR = "ERROR"
CANCELLED = "CANCELLED"

_TERMINAL = frozenset({DONE, ERROR, CANCELLED})

# Nextflow must poll at least once to see a terminal state, or it reads the job as lost.
_REAPED_TTL = 600.0


# By value, so workers need no nf_ray install; py_modules could clash with the job's own.
def _pickle_worker_modules_by_value() -> None:
    import ray  # noqa: PLC0415

    ray.cloudpickle.register_pickle_by_value(job)


@dataclass
class Entry:
    """One submitted task, as the daemon tracks it."""

    task_id: int
    name: str
    work_dir: str
    ref: object = None
    state: str = PENDING
    submitted: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    exit_code: int | None = None
    node_id: str = ""
    node_ip: str = ""
    detail: str = ""
    reaped_at: float = 0.0


class Scheduler:
    """Owns the Ray connection and the task table."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._lock = threading.RLock()
        # The reaper thread and every status request reap; without this a task could finish twice.
        self._reap_lock = threading.Lock()
        self._entries: dict[int, Entry] = {}
        self._next_id = 1
        self._stop = threading.Event()
        self._seen_limits = resources.NodeLimits()
        self._connect()

    def _connect(self) -> None:
        import ray  # noqa: PLC0415

        address = self.config.address
        # address="auto" raises with no cluster running; "local" starts a Ray instance instead.
        kwargs: dict[str, object] = {"namespace": self.config.namespace}
        if address and address not in ("local", "auto"):
            kwargs["address"] = address
        elif address == "auto":
            kwargs["address"] = "auto"

        ray.init(**kwargs)  # type: ignore[arg-type]
        _pickle_worker_modules_by_value()
        log.info(
            "connected to Ray %s (python %s), namespace %r",
            ray.__version__,
            ".".join(str(v) for v in sys.version_info[:3]),
            self.config.namespace,
        )

    def limits(self) -> resources.NodeLimits:
        """The largest single node a task could land on; a declared ceiling beats observed."""
        cfg = self.config
        if cfg.max_node_cpus or cfg.max_node_memory_gb or cfg.max_node_gpus:
            return resources.NodeLimits(
                cpus=cfg.max_node_cpus,
                memory_bytes=int(cfg.max_node_memory_gb * (1 << 30)),
                gpus=cfg.max_node_gpus,
                source="NF_RAY_MAX_NODE_* (declared)",
            )

        import ray  # noqa: PLC0415

        best = resources.NodeLimits()
        for node in ray.nodes():
            if not node.get("Alive"):
                continue
            res = node.get("Resources", {}) or {}
            cpus = float(res.get("CPU", 0) or 0)
            memory = int(res.get("memory", 0) or 0)
            gpus = float(res.get("GPU", 0) or 0)
            # The head has CPU: 0 by template convention.
            if cpus <= 0:
                continue
            best = resources.NodeLimits(
                cpus=max(best.cpus, cpus),
                memory_bytes=max(best.memory_bytes, memory),
                gpus=max(best.gpus, gpus),
                source="largest live node",
            )

        # High-water mark: a group that scaled back down still proves a request fits.
        with self._lock:
            self._seen_limits = resources.NodeLimits(
                cpus=max(self._seen_limits.cpus, best.cpus),
                memory_bytes=max(self._seen_limits.memory_bytes, best.memory_bytes),
                gpus=max(self._seen_limits.gpus, best.gpus),
                source="largest node seen alive so far",
            )
            return self._seen_limits

    def submit(self, script: str, work_dir: str) -> int:
        """Submit one Nextflow job script as a Ray task; return its id."""
        import ray  # noqa: PLC0415

        script_path = script if os.path.isabs(script) else os.path.join(work_dir, script)
        parsed = directives_mod.parse_script(script_path)

        request = resources.build_request(
            parsed,
            extra_resources=self.config.extra_resources,
            default_accelerator_type=self.config.default_accelerator,
        )
        request = resources.check_schedulable(
            request,
            self.limits(),
            process=parsed.name,
            clamp=self.config.clamp_resources,
        )

        options: dict[str, object] = dict(request.options())
        options["max_retries"] = self.config.task_max_retries
        runtime_env = envs.build_runtime_env(parsed, self.config)
        if runtime_env:
            options["runtime_env"] = runtime_env
        if self.config.scheduling_strategy != "DEFAULT":
            options["scheduling_strategy"] = self.config.scheduling_strategy

        with self._lock:
            task_id = self._next_id
            self._next_id += 1

        spec = job.TaskSpec(
            task_id=task_id,
            name=parsed.name,
            work_dir=work_dir,
            argv=directives_mod.render_command(script_path),
            num_gpus=request.num_gpus,
            request=request.describe(),
        )

        remote = ray.remote(job.execute_and_record).options(**options)  # type: ignore[arg-type]
        ref = remote.remote(spec)

        entry = Entry(task_id=task_id, name=parsed.name, work_dir=work_dir, ref=ref)
        with self._lock:
            self._entries[task_id] = entry
        log.info("submitted task %s (%s) %s", task_id, parsed.name or "-", request.describe())
        return task_id

    def status(self) -> list[tuple[int, str]]:
        """The whole queue, in one call -- what Nextflow polls."""
        self._promote_started()
        with self._lock:
            return [(e.task_id, e.state) for e in self._entries.values()]

    def kill(self, ids: list[int]) -> None:
        import ray  # noqa: PLC0415

        with self._lock:
            targets = [self._entries.get(i) for i in ids]
        for entry in targets:
            if entry is None or entry.state in _TERMINAL or entry.ref is None:
                continue
            with contextlib.suppress(Exception):
                # force=False lets the job script's EXIT trap write .exitcode and clean scratch.
                ray.cancel(entry.ref, force=False)
            entry.detail = "cancelled by Nextflow"

    def info(self) -> dict[str, object]:
        import ray  # noqa: PLC0415

        limits = self.limits()
        with self._lock:
            counts: dict[str, int] = {}
            for entry in self._entries.values():
                counts[entry.state] = counts.get(entry.state, 0) + 1
        return {
            "ray_version": ray.__version__,
            "python_version": ".".join(str(v) for v in sys.version_info[:3]),
            "namespace": self.config.namespace,
            "work_dir": self.config.work_dir,
            "socket": self.config.socket_path,
            "counts": counts,
            "limits": {
                "cpus": limits.cpus,
                "memory_gb": round(limits.memory_bytes / (1 << 30), 1),
                "gpus": limits.gpus,
                "source": limits.source,
            },
            "cluster_resources": dict(ray.cluster_resources()),
            "nodes_alive": sum(1 for n in ray.nodes() if n.get("Alive")),
        }

    def _promote_started(self) -> None:
        with self._lock:
            pending = [e for e in self._entries.values() if e.state == PENDING]
        for entry in pending:
            marker = os.path.join(entry.work_dir, job.PLACEMENT_FILENAME)
            # Stat outside the lock (NFS); recheck under it, as the task may have finished.
            if os.path.exists(marker):
                with self._lock:
                    if entry.state == PENDING:
                        entry.state = RUNNING
                        entry.started = time.time()

    def reap_once(self) -> None:
        """Collect finished tasks and record their outcomes; safe from several threads."""
        import ray  # noqa: PLC0415

        with self._reap_lock:
            with self._lock:
                live = [
                    e
                    for e in self._entries.values()
                    if e.state not in _TERMINAL and e.ref is not None
                ]
            if live:
                refs = [e.ref for e in live]
                ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0, fetch_local=False)
                # ray.wait returns the refs it was given, so identity maps them back.
                by_ref = {id(e.ref): e for e in live}
                for ref in ready:
                    entry = by_ref.get(id(ref))
                    if entry is not None:
                        self._finish(entry, ref)

            self._expire()

    def _finish(self, entry: Entry, ref: object) -> None:
        import ray  # noqa: PLC0415

        entry.finished = time.time()
        try:
            result: job.TaskResult = ray.get(ref)
        except BaseException as exn:  # noqa: BLE001 -- Ray raises many types
            code = errors.exit_code_for(exn)
            with self._lock:
                entry.state = CANCELLED if code == errors.EXIT_CANCELLED else ERROR
                entry.exit_code = code
                entry.detail = errors.describe(exn)
            log.warning("task %s failed in Ray: %s", entry.task_id, entry.detail)
            self._write_exitcode_if_absent(entry.work_dir, code)
            self._log_beside_task(entry.work_dir, entry.detail)
        else:
            with self._lock:
                entry.state = DONE
                entry.exit_code = result.exit_code
                entry.node_id = result.node_id
                entry.node_ip = result.node_ip
                if not entry.started and result.started_epoch:
                    entry.started = result.started_epoch
            log.info(
                "task %s exited %s on %s after %.1fs",
                entry.task_id,
                result.exit_code,
                result.node_ip or "?",
                result.seconds_running,
            )
        entry.ref = None
        self._append_placement(entry)

    @staticmethod
    def _write_exitcode_if_absent(work_dir: str, code: int) -> None:
        # O_EXCL: a .exitcode the wrapper managed to write is the authority.
        path = os.path.join(work_dir, ".exitcode")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except OSError as exn:
            if exn.errno != errno.EEXIST:
                log.warning("could not write %s: %s", path, exn)
            return
        with os.fdopen(fd, "w") as out:
            out.write(f"{code}\n")

    @staticmethod
    def _log_beside_task(work_dir: str, message: str) -> None:
        path = os.path.join(work_dir, job.LOG_FILENAME)
        with contextlib.suppress(OSError), open(path, "a") as out:
            out.write(f"[nf-ray] {message}\n")

    def _append_placement(self, entry: Entry) -> None:
        # One writer on the head: many nodes appending to one NFS file interleave.
        path = self.config.placement_tsv
        if not path:
            return
        node_id, node_ip = entry.node_id, entry.node_ip
        if not node_id:
            # Ray failed before the task reported; use its placement file, if any.
            marker = os.path.join(entry.work_dir, job.PLACEMENT_FILENAME)
            with contextlib.suppress(OSError, ValueError):
                with open(marker) as handle:
                    payload = json.load(handle)
                node_id = str(payload.get("node_id", ""))
                node_ip = str(payload.get("node_ip", ""))
        try:
            new = not os.path.exists(path)
            with open(path, "a") as out:
                if new:
                    out.write(
                        "task_id\tname\tstate\texit_code\tnode_id\tnode_ip\t"
                        "submitted\tstarted\tfinished\twork_dir\n"
                    )
                out.write(
                    f"{entry.task_id}\t{entry.name}\t{entry.state}\t"
                    f"{'' if entry.exit_code is None else entry.exit_code}\t"
                    f"{node_id}\t{node_ip}\t{entry.submitted:.3f}\t"
                    f"{entry.started:.3f}\t{entry.finished:.3f}\t{entry.work_dir}\n"
                )
        except OSError as exn:
            log.warning("could not append placement record: %s", exn)

    def _expire(self) -> None:
        now = time.time()
        with self._lock:
            for entry in self._entries.values():
                if entry.state in _TERMINAL and not entry.reaped_at:
                    entry.reaped_at = now
            stale = [
                i
                for i, e in self._entries.items()
                if e.reaped_at and now - e.reaped_at > _REAPED_TTL
            ]
            for i in stale:
                del self._entries[i]

    def run_reaper(self) -> None:
        while not self._stop.wait(self.config.poll_interval):
            try:
                self.reap_once()
            except Exception:  # noqa: BLE001 -- a reaper that dies hangs the run
                log.exception("reaper iteration failed")

    def shutdown(self) -> None:
        """Cancel anything still running, then disconnect."""
        self._stop.set()
        with self._lock:
            live = [e for e in self._entries.values() if e.state not in _TERMINAL]
        if live:
            log.info("cancelling %d task(s) still in flight", len(live))
            self.kill([e.task_id for e in live])
        with contextlib.suppress(Exception):
            import ray  # noqa: PLC0415

            ray.shutdown()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:  # noqa: D102
        server: _Server = self.server  # type: ignore[assignment]
        server.touch()
        raw = self.rfile.readline()
        if not raw:
            return
        try:
            request = json.loads(raw)
            response = server.dispatch(request)
        except Exception as exn:  # noqa: BLE001 -- every failure must reach the client
            response = {"ok": False, "error": f"{type(exn).__name__}: {exn}"}
        self.wfile.write((json.dumps(response) + "\n").encode())


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, path: str, scheduler: Scheduler) -> None:
        super().__init__(path, _Handler)
        self.scheduler = scheduler
        self._last_contact = time.time()
        self._contact_lock = threading.Lock()

    def touch(self) -> None:
        with self._contact_lock:
            self._last_contact = time.time()

    def idle_seconds(self) -> float:
        with self._contact_lock:
            return time.time() - self._last_contact

    def dispatch(self, request: dict) -> dict:
        op = request.get("op")
        if op == "submit":
            task_id = self.scheduler.submit(request["script"], request["work_dir"])
            return {"ok": True, "task_id": task_id}
        if op == "status":
            self.scheduler.reap_once()
            return {"ok": True, "rows": self.scheduler.status()}
        if op == "kill":
            self.scheduler.kill([int(i) for i in request.get("ids", [])])
            return {"ok": True}
        if op == "info":
            return {"ok": True, "info": self.scheduler.info()}
        if op == "ping":
            return {"ok": True}
        if op == "shutdown":
            threading.Thread(target=self.shutdown, daemon=True).start()
            return {"ok": True}
        return {"ok": False, "error": f"unknown op {op!r}"}


def serve(config: Config) -> None:
    """Run the daemon in the foreground until idle, killed, or told to stop."""
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    warning = shared_storage_warning(config.work_dir)
    if warning:
        log.warning("%s", warning)

    with contextlib.suppress(FileNotFoundError):
        os.unlink(config.socket_path)
    os.makedirs(os.path.dirname(config.socket_path) or ".", exist_ok=True)

    scheduler = Scheduler(config)
    server = _Server(config.socket_path, scheduler)
    os.chmod(config.socket_path, 0o600)

    threading.Thread(target=scheduler.run_reaper, daemon=True).start()

    def watch_idle() -> None:
        while True:
            time.sleep(min(30.0, max(1.0, config.idle_timeout / 10)))
            if server.idle_seconds() > config.idle_timeout:
                log.warning(
                    "no client contact for %.0fs; shutting down so a killed run "
                    "cannot leave this cluster held open",
                    config.idle_timeout,
                )
                server.shutdown()
                return

    threading.Thread(target=watch_idle, daemon=True).start()

    log.info("listening on %s", config.socket_path)
    try:
        server.serve_forever()
    finally:
        scheduler.shutdown()
        server.server_close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(config.socket_path)
        log.info("stopped")


def is_running(socket_path: str) -> bool:
    """Whether a daemon is answering on *socket_path*."""
    from nf_ray import client  # noqa: PLC0415 -- avoids an import cycle

    return client.ping(socket_path)
