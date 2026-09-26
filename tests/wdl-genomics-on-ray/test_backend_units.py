#!/usr/bin/env python3
"""Offline checks of wdl_on_ray backend behaviour that a smoke run would not catch.

Stand-ins replace the Ray calls, so this needs neither a cluster nor data and runs in seconds:

* a retried task's start is its own attempt's, not the previous attempt's
  (``ray_placement.json`` survives in the task directory miniwdl retries in);
* the per-task ceiling comes from nodes that can run tasks, so a ``CPU: 0`` head neither sets
  it nor, alone, clamps every task to one CPU; and miniwdl's own clamp reads it as intended.

Each check drives the backend's real code, and miniwdl's; only Ray itself is replaced.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

#: The wdl_on_ray under test. rayapp, and so CI, flattens templates/<name>/ and tests/<name>/ into
#: one directory, which puts the package beside this file; a repo checkout keeps it two levels up.
#: Put first on sys.path either way, so the copy installed in the image cannot stand in for it.
_HERE = Path(__file__).resolve().parent
_CANDIDATES = (_HERE, _HERE.parents[1] / "templates" / "wdl-genomics-on-ray")
TEMPLATE = next((d for d in _CANDIDATES if (d / "wdl_on_ray" / "backend.py").is_file()), None)
if TEMPLATE is None:
    raise SystemExit(f"wdl_on_ray not found in {', '.join(map(str, _CANDIDATES))}")
sys.path.insert(0, str(TEMPLATE))

import ray  # noqa: E402
import WDL  # noqa: E402  (also installs miniwdl's NOTICE log level)
from WDL import Value  # noqa: E402
from WDL.runtime.config import Loader  # noqa: E402

from wdl_on_ray import backend, resources, runtimes  # noqa: E402
from wdl_on_ray import config as ray_config  # noqa: E402
from wdl_on_ray import job as ray_job  # noqa: E402

GiB = 2**30


class Records(logging.Handler):
    """Keeps every record, so a check can assert on the backend's structured log messages."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def named(self, message: str) -> list[dict[str, Any]]:
        """The fields of each structured message whose text is ``message``."""
        return [
            dict(r.msg.kwargs)
            for r in self.records
            if getattr(r.msg, "message", None) == message
        ]


def _logger(name: str) -> tuple[logging.Logger, Records]:
    logger = logging.getLogger(f"test_backend_units.{name}")
    logger.handlers[:] = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    records = Records()
    logger.addHandler(records)
    return logger, records


@contextmanager
def patched(target: Any, **attrs: Any) -> Iterator[None]:
    """Set attributes on ``target`` for the duration, then put the originals back."""
    missing = object()
    saved = {name: getattr(target, name, missing) for name in attrs}
    for name, value in attrs.items():
        setattr(target, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is missing:
                delattr(target, name)
            else:
                setattr(target, name, value)


@contextmanager
def environ(**values: str | None) -> Iterator[None]:
    """Set (or, with None, unset) environment variables for the duration."""
    saved = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------- stale placement


class FakeTask:
    """Stands in for one Ray task: queued for ``start_after`` polls of ``ray.wait``, then
    started (its worker writes the placement record, as ``job.execute_and_record`` does), then
    finished after ``done_after`` polls."""

    def __init__(self, start_after: int, done_after: int, node_id: str) -> None:
        self.start_after, self.done_after, self.node_id = start_after, done_after, node_id
        self.polls = 0
        self.placement_path = ""
        self.placement_existed_at_submit: bool | None = None

    def remote(self, fn: Callable[..., Any]) -> SimpleNamespace:
        """``ray.remote(fn)``, so that ``.options(**kw).remote(...)`` reaches :meth:`submit`."""
        assert fn is ray_job.execute_and_record, fn
        return SimpleNamespace(options=lambda **_: SimpleNamespace(remote=self.submit))

    def submit(self, job: ray_job.ContainerJob, placement_path: str) -> str:
        del job
        self.placement_path = placement_path
        self.placement_existed_at_submit = os.path.exists(placement_path)
        return "ref"

    def wait(self, refs: list[str], timeout: float) -> tuple[list[str], list[str]]:
        del timeout
        self.polls += 1
        if self.polls == self.start_after:
            with open(self.placement_path, "w") as out:
                json.dump({"node_id": self.node_id, "node_ip": "10.0.0.2", "pid": 2}, out)
        return (refs, []) if self.polls >= self.done_after else ([], refs)

    def get(self, ref: str) -> ray_job.JobResult:
        del ref
        return ray_job.JobResult(exit_code=0, node_id=self.node_id, node_ip="10.0.0.2")


def _container(run_dir: str, runtime_values: dict[str, Any] | None = None) -> backend.RayContainer:
    """A RayContainer as miniwdl would hold it, under ``--container-runtime none``, with
    ``runtime_values`` as miniwdl's ``process_runtime`` would have left them."""
    cfg = Loader(logging.getLogger("test_backend_units.cfg"))
    backend.RayContainer._ray_cfg = ray_config.load(cfg)
    backend.RayContainer._runtime = runtimes.get("none")
    backend.RayContainer._exe = []
    # Skips warn_if_not_shared, which would ask a real cluster how many nodes it has.
    backend.RayContainer._checked_shared_run_dir = True
    container = backend.RayContainer(cfg, "call-Shard", run_dir)
    container.runtime_values = dict(runtime_values or {})
    return container


def test_retry_does_not_inherit_placement() -> None:
    """Attempt 2 is queued until its own worker writes the record, and the record left behind
    is attempt 2's. Before the fix, attempt 1's record made the retry read as started at the
    first poll, on attempt 1's node."""
    logger, records = _logger("placement")
    with tempfile.TemporaryDirectory(prefix="wdl-units-") as tmp:
        run_dir = os.path.join(tmp, "call-Shard")
        container = _container(run_dir, {"cpu": 2, "memory_reservation": GiB, "docker": "x"})
        placement = Path(run_dir, "ray_placement.json")

        # Attempt 1 ran on node OLD and was interrupted; miniwdl resets into work2 and retries.
        placement.write_text(json.dumps({"node_id": "OLD", "node_ip": "10.0.0.1", "pid": 1}))
        container.reset(logger)
        assert container.try_counter == 2

        task = FakeTask(start_after=3, done_after=5, node_id="NEW")
        with patched(ray, remote=task.remote, wait=task.wait, get=task.get):
            exit_code = container._run(logger, lambda: False, "true")

        assert exit_code == 0
        assert task.placement_existed_at_submit is False, (
            "attempt 1's ray_placement.json was still there when attempt 2 was submitted"
        )
        started = records.named("task started on Ray worker")
        assert len(started) == 1, started
        assert started[0]["node_id"] == "NEW", f"reported the previous attempt's node: {started}"
        assert json.loads(placement.read_text())["node_id"] == "NEW"


# ------------------------------------------------------------------------------ task ceiling


def node(cpu: float, memory_gib: float, alive: bool = True) -> dict[str, Any]:
    """One entry of ``ray.nodes()``, reduced to what the backend reads. Like Ray, a node started
    with no CPUs has no ``CPU`` key at all."""
    resources: dict[str, float] = {"memory": memory_gib * GiB, "object_store_memory": GiB}
    if cpu:
        resources["CPU"] = float(cpu)
    return {"Alive": alive, "Resources": resources}


#: The template's head (`CPU: 0` in its compute configs), with more memory than the worker below,
#: so a ceiling that took memory from every node would visibly take the head's.
HEAD = node(0, 64)
WORKER = node(4, 8)


def limits_for(nodes: list[dict[str, Any]], **settings: str) -> tuple[dict[str, int], Records]:
    """detect_resource_limits against a cluster of ``nodes``, under ``[ray]`` ``settings``."""
    logger, records = _logger("limits")
    env = {f"MINIWDL__RAY__{key.upper()}": value for key, value in settings.items()}
    with environ(**env), patched(backend, connect=lambda *_: None), patched(
        ray, nodes=lambda: nodes
    ):
        cfg = Loader(logging.getLogger("test_backend_units.cfg"))
        backend.RayContainer._ray_cfg = ray_config.load(cfg)
        backend.RayContainer._limits = None
        return backend.RayContainer.detect_resource_limits(cfg, logger), records


def test_no_node_that_can_run_tasks_means_no_ceiling() -> None:
    """A job's entrypoint can start before its first worker joins. Before the fix, the CPU:0 head
    alone gave a ceiling of one CPU and the head's memory, and miniwdl clamped every task to it."""
    no_worker = [
        [HEAD],
        [],
        [HEAD, node(32, 128, alive=False)],  # a worker that has since gone
        [{"Alive": True, "Resources": {"CPU": 0.0, "memory": GiB}}],  # CPU reported, as 0
    ]
    for nodes in no_worker:
        limits, records = limits_for(nodes)
        assert limits == {"cpu": backend.NO_LIMIT, "mem_bytes": backend.NO_LIMIT}, (nodes, limits)
        [warning] = [r for r in records.records if r.levelno == logging.WARNING]
        assert set(warning.msg.kwargs["set_to_clamp"]) == {"--max-cpu", "[ray] max_memory_bytes"}


def test_head_without_cpus_does_not_set_the_ceiling() -> None:
    """With a worker up, the ceiling is the worker's: its CPUs and its memory, not the head's."""
    limits, records = limits_for([HEAD, WORKER, node(64, 256, alive=False)])
    assert limits == {"cpu": 4, "mem_bytes": 8 * GiB}, limits
    assert not [r for r in records.records if r.levelno == logging.WARNING]

    # `cluster` sums the nodes that can run tasks, and likewise leaves the head out.
    limits, _ = limits_for([HEAD, WORKER, node(8, 16)], limit_source="cluster")
    assert limits == {"cpu": 12, "mem_bytes": 24 * GiB}, limits
    limits, _ = limits_for([HEAD], limit_source="cluster")
    assert limits == {"cpu": backend.NO_LIMIT, "mem_bytes": backend.NO_LIMIT}, limits


def test_overrides_still_set_the_ceiling() -> None:
    """``[ray] max_cpu`` / ``max_memory_bytes`` win whether or not a worker is up, each on its
    own, and the warning names only what is still unclamped."""
    both = {"max_cpu": "16", "max_memory_bytes": str(64 * GiB)}
    for nodes in ([HEAD], [HEAD, WORKER]):
        limits, records = limits_for(nodes, **both)
        assert limits == {"cpu": 16, "mem_bytes": 64 * GiB}, (nodes, limits)
        assert not [r for r in records.records if r.levelno == logging.WARNING]

    limits, records = limits_for([HEAD], max_cpu="16")
    assert limits == {"cpu": 16, "mem_bytes": backend.NO_LIMIT}, limits
    [warning] = [r for r in records.records if r.levelno == logging.WARNING]
    assert warning.msg.kwargs["set_to_clamp"] == ["[ray] max_memory_bytes"], warning.msg.kwargs

    limits, _ = limits_for([HEAD, WORKER], max_memory_bytes=str(6 * GiB))
    assert limits == {"cpu": 4, "mem_bytes": 6 * GiB}, limits


def test_miniwdl_reads_no_ceiling_as_no_clamp() -> None:
    """miniwdl 1.15.0's own code, fed the reported limits. NO_LIMIT leaves Flye's 30 CPUs and
    32 GiB alone; a measured ceiling clamps them. And for a WDL 1.2 task that sets no cpu, the
    ``task.cpu`` miniwdl derives from the limit is the 1 CPU the backend reserves for it, where a
    large "no limit" number would reach the command as a thread count."""
    logger, records = _logger("clamp")
    flye = {"cpu": Value.Int(30), "memory": Value.String("32 GiB"), "docker": Value.String("x")}
    with tempfile.TemporaryDirectory(prefix="wdl-units-") as tmp:
        backend.RayContainer._limits = {"cpu": backend.NO_LIMIT, "mem_bytes": backend.NO_LIMIT}
        container = _container(os.path.join(tmp, "unclamped"))
        container.process_runtime(logger, dict(flye))
        assert container.runtime_values["cpu"] == 30, container.runtime_values
        assert container.runtime_values["memory_reservation"] == 32 * GiB
        assert not records.named("runtime.cpu adjusted to host limit")

        doc = WDL.parse_document("version 1.2\n\ntask t {\n  command <<<\n    true\n  >>>\n}\n")
        doc.typecheck()
        container = _container(os.path.join(tmp, "wdl12"))
        container.process_runtime(logger, {"docker": Value.String("x")})
        container.build_task_runtime_info_struct(logger, "t", doc.tasks[0])
        assert container.task_runtime_info_struct is not None
        task_cpu = container.task_runtime_info_struct.value["cpu"].value
        reserved = resources.build_request(container.runtime_values).num_cpus
        assert task_cpu == reserved == 1.0, (task_cpu, reserved)

        backend.RayContainer._limits = {"cpu": 4, "mem_bytes": 8 * GiB}
        container = _container(os.path.join(tmp, "clamped"))
        container.process_runtime(logger, dict(flye))
        assert container.runtime_values["cpu"] == 4, container.runtime_values
        assert container.runtime_values["memory_reservation"] == 8 * GiB
        assert records.named("runtime.cpu adjusted to host limit")


# ------------------------------------------------------------------------------------ runner

TESTS = [
    test_retry_does_not_inherit_placement,
    test_no_node_that_can_run_tasks_means_no_ceiling,
    test_head_without_cpus_does_not_set_the_ceiling,
    test_overrides_still_set_the_ceiling,
    test_miniwdl_reads_no_ceiling_as_no_clamp,
]


def main() -> int:
    print(f"wdl_on_ray under test: {Path(backend.__file__).parent}")
    failed = 0
    for test in TESTS:
        # Every check starts from an unconfigured backend: detect_resource_limits memoizes.
        backend.RayContainer._limits = None
        try:
            test()
        except Exception:  # noqa: BLE001 - reported, and counted
            failed += 1
            print(f"FAIL  {test.__name__}\n{traceback.format_exc()}", file=sys.stderr)
        else:
            print(f"ok    {test.__name__}")
    if failed:
        print(f"\n{failed} of {len(TESTS)} backend checks failed", file=sys.stderr)
        return 1
    print(f"\nall {len(TESTS)} backend checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
