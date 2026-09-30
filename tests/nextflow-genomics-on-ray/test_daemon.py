#!/usr/bin/env python3
"""Offline tests for nf_ray.daemon's task table, against a stand-in ray module."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import threading
import traceback
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE = os.path.abspath(
    os.path.join(_HERE, "..", "..", "templates", "nextflow-genomics-on-ray")
)
for candidate in (os.getcwd(), _TEMPLATE):
    if os.path.isdir(os.path.join(candidate, "nf_ray")):
        sys.path.insert(0, candidate)
        break


class _Ref:
    def __init__(self, task_id: int) -> None:
        self.task_id = task_id


class _FakeRay(types.ModuleType):
    """Just the surface nf_ray.daemon touches."""

    __version__ = "2.58.0"

    def __init__(self) -> None:
        super().__init__("ray")
        self.cloudpickle = types.SimpleNamespace(register_pickle_by_value=lambda _m: None)
        self.wait_hook = None
        self.get_result = None

    def init(self, **_kwargs) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def cancel(self, _ref, force: bool = False) -> None:
        pass

    def wait(self, refs, num_returns, timeout, fetch_local):
        if self.wait_hook:
            self.wait_hook()
        return list(refs), []

    def get(self, ref):
        return self.get_result(ref)


ray = _FakeRay()
sys.modules["ray"] = ray

from nf_ray import daemon, job  # noqa: E402
from nf_ray.config import Config  # noqa: E402

_FAILURES: list[str] = []


def check(name: str):
    def wrap(fn):
        try:
            fn()
        except Exception:
            _FAILURES.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
        return fn

    return wrap


def _scheduler(tmp: str) -> daemon.Scheduler:
    ray.wait_hook = None
    ray.get_result = lambda ref: job.TaskResult(
        task_id=ref.task_id, exit_code=0, node_id="node-a", node_ip="10.0.0.1"
    )
    placement = os.path.join(tmp, "nf_ray_placement.tsv")
    return daemon.Scheduler(Config(work_dir=tmp, placement_tsv=placement, address="local"))


def _add(s: daemon.Scheduler, tmp: str, task_id: int) -> daemon.Entry:
    work = os.path.join(tmp, f"task{task_id}")
    os.makedirs(work, exist_ok=True)
    entry = daemon.Entry(task_id=task_id, name=f"t{task_id}", work_dir=work, ref=_Ref(task_id))
    s._entries[task_id] = entry
    return entry


def _placement_rows(tmp: str) -> list[list[str]]:
    with open(os.path.join(tmp, "nf_ray_placement.tsv")) as handle:
        return [line.rstrip("\n").split("\t") for line in handle][1:]


@check("reap: two concurrent reapers finish each task exactly once")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        s = _scheduler(tmp)
        for i in (1, 2, 3):
            _add(s, tmp, i)

        # Hold each reaper at ray.wait until a second arrives, as the reaper thread and a status
        # request can; with the reap lock the second never does, and the barrier times out.
        barrier = threading.Barrier(2)

        def hold() -> None:
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait(timeout=0.5)

        ray.wait_hook = hold
        threads = [threading.Thread(target=s.reap_once) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rows = _placement_rows(tmp)
        assert len(rows) == 3, f"{len(rows)} placement rows for 3 tasks: {rows}"
        assert sorted(int(r[0]) for r in rows) == [1, 2, 3]
        assert all(e.state == daemon.DONE for e in s._entries.values())


@check("reap: a status request after the reaper finds nothing left to finish")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        s = _scheduler(tmp)
        _add(s, tmp, 1)
        s.reap_once()
        s.reap_once()
        assert len(_placement_rows(tmp)) == 1


@check("promote: a task that finishes mid-scan is not moved back to RUNNING")
def _() -> None:
    # _promote_started stats markers on NFS outside the lock; a task can finish in that window.
    with tempfile.TemporaryDirectory() as tmp:
        s = _scheduler(tmp)
        entry = _add(s, tmp, 1)
        with open(os.path.join(entry.work_dir, job.PLACEMENT_FILENAME), "w") as out:
            out.write("{}")

        real_exists = os.path.exists

        def exists_then_finish(path: str) -> bool:
            if path.endswith(job.PLACEMENT_FILENAME) and entry.state == daemon.PENDING:
                s.reap_once()  # the task completes while the marker is being checked
            return real_exists(path)

        daemon.os.path.exists = exists_then_finish
        try:
            s._promote_started()
        finally:
            daemon.os.path.exists = real_exists

        assert entry.state == daemon.DONE, entry.state
        s.reap_once()  # and a later reap does not trip over it
        assert len(_placement_rows(tmp)) == 1


@check("failure: Ray's error becomes .exitcode, but never over the wrapper's own")
def _() -> None:
    class OutOfMemoryError(Exception):
        pass

    def boom(_ref):
        raise OutOfMemoryError("worker killed by the memory monitor")

    with tempfile.TemporaryDirectory() as tmp:
        s = _scheduler(tmp)
        ray.get_result = boom
        lost = _add(s, tmp, 1)
        finished = _add(s, tmp, 2)
        with open(os.path.join(finished.work_dir, ".exitcode"), "w") as out:
            out.write("0\n")

        s.reap_once()

        with open(os.path.join(lost.work_dir, ".exitcode")) as handle:
            assert handle.read().strip() == "137"
        with open(os.path.join(finished.work_dir, ".exitcode")) as handle:
            assert handle.read().strip() == "0", "the wrapper's exit code was overwritten"
        assert lost.state == daemon.ERROR and lost.exit_code == 137


if _FAILURES:
    print(f"\n{len(_FAILURES)} failing: {', '.join(_FAILURES)}", file=sys.stderr)
    sys.exit(1)
print("\nall daemon unit checks passed")
