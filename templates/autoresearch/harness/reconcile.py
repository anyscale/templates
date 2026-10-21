"""R6 — the terminal-row writer: replace each run's committed estimate with what it actually used.

`hooks.py post` commits an *estimate* at submit time (RUNNING heartbeat). The estimate has to
exist: it's the only way to say whether a job fits the budget before it runs. Once the job
ends, this module asks Anyscale what actually ran and writes the single terminal row the
registry's spend math trusts: exact GPU-hours, CPU-hours and dollars per instance type, from
the job's cluster event log and published prices (`usage.py`, `prices.py`). The remaining
budget is then the envelope minus those actuals.

Recoverable by design: job state and cluster events survive on Anyscale's side, so a
reconcile that runs late is exactly as correct as one that runs live — a standing Monitor
gives freshness, this gives truth. The terminal row waits until the cluster has finished
tearing down, so its last nodes are counted.

    python3 reconcile.py <AUTORESEARCH_BASE>              # sweep every campaign's open rows
    python3 reconcile.py <AUTORESEARCH_BASE> --campaign fintech_fm

If the cluster's events can't be read, the old basis is used and labeled as such:
wall-clock (created_at -> terminal updated_at) x the committed max GPU fleet, an upper bound.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import registry  # noqa: E402

_STATE_RE = re.compile(r"^state:\s*(\S+)", re.M)
_TS_RE = {
    "created_at": re.compile(r"^created_at:\s*(\S+ \S+)", re.M),
    "updated_at": re.compile(r"^updated_at:\s*(\S+ \S+)", re.M),
}


def fetch_status(job_id: str) -> str:
    """Fallback only (read-only). Note: recent CLIs print no timestamps here."""
    return subprocess.run(["anyscale", "job", "status", "--id", job_id, "--verbose"],
                          capture_output=True, text=True, timeout=120).stdout


def measure(job_id: str) -> dict | None:
    """The job's actual usage, or None if it can't be read."""
    import usage
    return usage.job_usage(job_id)


def parse_status(text: str) -> dict | None:
    """{state, hours} from `anyscale job status --verbose` output, or None if unparseable."""
    m = _STATE_RE.search(text)
    if not m:
        return None
    out = {"state": m.group(1)}
    stamps = {}
    for key, rx in _TS_RE.items():
        t = rx.search(text)
        if t:
            stamps[key] = datetime.fromisoformat(t.group(1).replace(" ", "T"))
    if len(stamps) == 2:
        out["hours"] = round(
            (stamps["updated_at"] - stamps["created_at"]).total_seconds() / 3600, 4)
    return out


def open_rows(base: str, campaign: str) -> list[dict]:
    """RUNNING heartbeats with no terminal row yet."""
    return [r for r in registry.terminal_runs(base, campaign) if r["status"] == "RUNNING"]


def _ids(row: dict) -> dict:
    return {k: row[k] for k in ("campaign", "run_id", "commit", "rung", "eval_pin")}


def reconcile_measured(base: str, row: dict, used: dict) -> bool:
    """Write the terminal row from measured usage. Idempotent via registry.append_run."""
    return registry.append_run(base, {**_ids(row), "status": used["state"],
                                      "cost": used["cost"], "cost_basis": used["cost_basis"],
                                      "cluster_id": used.get("cluster_id"),
                                      "estimate": row["cost"]})


def reconcile_row(base: str, row: dict, status: dict) -> bool:
    """Fallback terminal row when usage can't be measured. Idempotent via registry.append_run."""
    if status["state"] not in registry.TERMINAL_STATES:
        return False  # still running — leave the heartbeat as-is
    hours = status.get("hours")
    num_gpus = (row.get("fleet") or {}).get("num_gpus", 0)
    cost = dict(row["cost"])
    if hours is not None:
        cost["gpu_hours"] = round(num_gpus * hours, 4)
        cost["wall_clock_hours"] = hours
        basis = ("fallback: wall-clock x max fleet (upper bound; cluster events unavailable)")
    else:
        basis = "fallback: committed estimate kept (no cluster events, no timestamps)"
    cost.pop("a10g_equiv_hours", None)  # re-derived at write time
    return registry.append_run(base, {**_ids(row), "status": status["state"], "cost": cost,
                                      "cost_basis": basis})


def sweep(base: str, campaign: str | None = None, fetch=fetch_status, measure=measure) -> dict:
    """Reconcile every open row (one campaign or all). Returns {run_id: outcome}."""
    campaigns = [campaign] if campaign else list(registry.iter_campaigns(base))
    results: dict[str, str] = {}
    for c in campaigns:
        for row in open_rows(base, c):
            job_id = row.get("job_id") or row["run_id"]
            if not job_id.startswith("prodjob_"):
                results[row["run_id"]] = "skipped: no prodjob id on the row"
                continue
            try:
                used = measure(job_id) if measure else None
            except Exception as e:  # network/auth: fall back, never crash the sweep
                used, results[row["run_id"]] = None, f"measure failed ({type(e).__name__})"
            if used is not None:
                if used["state"] not in registry.TERMINAL_STATES:
                    results[row["run_id"]] = f"still {used['state']}"
                elif not used.get("cluster_terminated", True):
                    results[row["run_id"]] = "waiting for cluster teardown"
                elif reconcile_measured(base, row, used):
                    k = used["cost"]
                    results[row["run_id"]] = (f"terminal row written: {used['state']} "
                                              f"({k['gpu_hours']} GPU-h, {k.get('cpu_hours')} "
                                              f"CPU-h, ${k.get('usd')})")
                continue
            status = parse_status(fetch(job_id))
            if status is None:
                results[row["run_id"]] = "skipped: unparseable status"
            elif reconcile_row(base, row, status):
                results[row["run_id"]] = f"terminal row written (fallback): {status['state']} " \
                                         f"({status.get('hours', '?')}h wall-clock)"
            else:
                results[row["run_id"]] = f"still {status['state']}"
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    camp = sys.argv[sys.argv.index("--campaign") + 1] if "--campaign" in sys.argv else None
    for rid, outcome in sweep(sys.argv[1], camp).items():
        print(f"{rid}: {outcome}")
