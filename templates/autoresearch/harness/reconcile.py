"""R6 — the terminal-row writer: reconcile committed estimates against what actually ran.

`hooks.py post` commits an *estimate* at submit time (RUNNING heartbeat); this module closes
the loop by asking Anyscale what actually happened and writing the single terminal row the
registry's spend math trusts. Recoverable by design: job state and timestamps survive on
Anyscale's side (proven by back-filling the FM campaign weeks later from `anyscale job
status` alone), so a reconcile that runs late is exactly as correct as one that runs live —
a standing Monitor gives freshness, this gives truth.

    python3 reconcile.py <AUTORESEARCH_BASE>              # sweep every campaign's open rows
    python3 reconcile.py <AUTORESEARCH_BASE> --campaign fintech_fm

Actual cost basis: wall-clock (created_at -> terminal updated_at) x the committed max GPU
fleet — an upper bound (autoscaled stages cost less; the console is the invoice of record).
The basis travels with the row so nobody mistakes the bound for a measurement.
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
    """The one infra-touching line (read-only)."""
    return subprocess.run(["anyscale", "job", "status", "--id", job_id, "--verbose"],
                          capture_output=True, text=True, timeout=120).stdout


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


def reconcile_row(base: str, row: dict, status: dict) -> bool:
    """Write the terminal row for one run. Idempotent via registry.append_run."""
    if status["state"] not in registry.TERMINAL_STATES:
        return False  # still running — leave the heartbeat as-is
    hours = status.get("hours")
    num_gpus = (row.get("fleet") or {}).get("num_gpus", 0)
    cost = dict(row["cost"])
    if hours is not None:
        cost["gpu_hours"] = round(num_gpus * hours, 4)
        cost["wall_clock_hours"] = hours
    cost.pop("a10g_equiv_hours", None)  # re-derived at write time from the actual raw hours
    return registry.append_run(base, {
        **{k: row[k] for k in ("campaign", "run_id", "commit", "rung", "eval_pin")},
        "status": status["state"], "cost": cost,
        "cost_basis": "actual wall-clock x max fleet (upper bound: autoscaling scales down; "
                      "console is the invoice of record)",
    })


def sweep(base: str, campaign: str | None = None, fetch=fetch_status) -> dict:
    """Reconcile every open row (one campaign or all). Returns {run_id: outcome}."""
    campaigns = [campaign] if campaign else list(registry.iter_campaigns(base))
    results: dict[str, str] = {}
    for c in campaigns:
        for row in open_rows(base, c):
            job_id = row.get("job_id") or row["run_id"]
            if not job_id.startswith("prodjob_"):
                results[row["run_id"]] = "skipped: no prodjob id on the row"
                continue
            status = parse_status(fetch(job_id))
            if status is None:
                results[row["run_id"]] = "skipped: unparseable status"
            elif reconcile_row(base, row, status):
                results[row["run_id"]] = f"terminal row written: {status['state']} " \
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
