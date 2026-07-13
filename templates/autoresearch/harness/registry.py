"""R1 — results registry (reference implementation).

The keystone requirement from REQUIREMENTS.md: an append-only JSONL ledger that lets a
fresh agent context reconstruct a campaign's whole state — what ran, what it cost, what
won, what was killed and why — from disk alone.

Dependency-free (stdlib only) on purpose: this is the "couple hundred lines of Python"
the build-order rule says to write first, before R2. Everything here traces to a line in
REQUIREMENTS.md R1; the one place it goes *beyond* the current docs is `a10g_equiv_hours`
in the cost record, which implements the tier-weighting fix (critiques.md #14/#20) — see
`tier_weight()` and `campaign_spend()`.

    Not-yet-built by design: R2 (launcher) generates the records this module stores; R6
    (monitor) is the single writer of terminal rows. This module is just the store + the
    queries R3/R4/promote-kill read.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

# --- tier weighting (critiques.md #14) -------------------------------------------------
# BUDGET_POLICY.md's own conversion table, as code. An A100-hour is 3.5 A10G-hours; a
# budget stated in raw GPU-hours silently lets an H100 run cost 5.5x what its number says.
# The registry stores BOTH raw hours (physics) and A10G-equivalent hours (currency), so
# wave caps and envelopes are comparable across GPU tiers. A10G is the unit (== 1.0).
TIER_WEIGHT = {
    "T4": 0.5,
    "L4": 0.8,
    "A10G": 1.0,
    "L40S": 1.8,
    "A100": 3.5,
    "H100": 5.5,
}

TERMINAL_STATES = {"SUCCEEDED", "FAILED", "ERRORED", "TERMINATED", "OUT_OF_RETRIES"}
RUN_REQUIRED = ("campaign", "run_id", "commit", "rung", "cost", "eval_pin", "status")
DECISION_VALUES = {"promote", "kill", "hold"}


def tier_weight(gpu_type: str) -> float:
    """A10G-equivalent multiplier for a GPU tier. Unknown tiers raise — an un-priced GPU
    must not silently count as free budget."""
    try:
        return TIER_WEIGHT[gpu_type]
    except KeyError:
        raise ValueError(
            f"unknown gpu_type {gpu_type!r}; add it to TIER_WEIGHT (BUDGET_POLICY.md table)"
        )


def a10g_equiv_hours(cost: dict) -> float:
    """Raw GPU-hours reweighted to the A10G-equivalent currency."""
    return round(cost["gpu_hours"] * tier_weight(cost["gpu_type"]), 4)


def _path(base: str, campaign: str) -> str:
    return os.path.join(base, "registry", f"{campaign}.jsonl")


def _read_rows(base: str, campaign: str) -> list[dict]:
    path = _path(base, campaign)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(base: str, campaign: str, record: dict) -> None:
    path = _path(base, campaign)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


# --- writers ---------------------------------------------------------------------------
def append_run(base: str, record: dict) -> bool:
    """Append one run row. Returns True if written, False if it was a no-op duplicate.

    Idempotent on (run_id, status) (REQUIREMENTS R1): the run streams a RUNNING heartbeat;
    the monitor writes the single terminal row. A double-fire of either is a no-op, so
    campaign_spend() never double-counts and R4's envelope math stays honest even if a
    monitor retries. The cost record is enriched with a10g_equiv_hours at write time.
    """
    missing = [k for k in RUN_REQUIRED if k not in record]
    if missing:
        raise ValueError(f"run row missing required fields: {missing}")
    if record["status"] not in TERMINAL_STATES and record["status"] != "RUNNING":
        raise ValueError(f"unknown status {record['status']!r}")
    if record.get("decision") is not None and record["decision"] not in DECISION_VALUES:
        raise ValueError(f"decision must be one of {DECISION_VALUES} or absent")

    row = {"type": "run", **record}
    row["cost"] = {**record["cost"], "a10g_equiv_hours": a10g_equiv_hours(record["cost"])}

    key = (row["run_id"], row["status"])
    for existing in _read_rows(base, record["campaign"]):
        if existing.get("type") == "run" and (existing["run_id"], existing["status"]) == key:
            return False  # idempotent no-op
    _append(base, record["campaign"], row)
    return True


def append_decision(base: str, campaign: str, run_ids: list[str], decision: str,
                    reason: str, seed_plan_commit: str) -> None:
    """Record a promote/kill/hold as its own typed row (critiques.md #3).

    Decisions are judgments *about* runs, not facts *of* a run — so they get their own row
    type instead of hiding in a run's free-text `notes`. `reason` is mandatory: the
    resume-from-disk promise breaks hardest on *why* something was killed.
    """
    if decision not in DECISION_VALUES:
        raise ValueError(f"decision must be one of {DECISION_VALUES}")
    if not reason:
        raise ValueError("a decision must record why (reason is mandatory)")
    _append(base, campaign, {
        "type": "decision", "campaign": campaign, "run_ids": run_ids,
        "decision": decision, "reason": reason, "seed_plan_commit": seed_plan_commit,
    })


# --- queries ---------------------------------------------------------------------------
def terminal_runs(base: str, campaign: str) -> list[dict]:
    """One row per run — the terminal row if present, else the latest heartbeat."""
    latest: dict[str, dict] = {}
    for row in _read_rows(base, campaign):
        if row.get("type") != "run":
            continue
        rid = row["run_id"]
        if row["status"] in TERMINAL_STATES or rid not in latest:
            latest[rid] = row
    return list(latest.values())


def campaign_spend(base: str, campaign: str) -> dict:
    """Spent budget in raw and A10G-equivalent GPU-hours. R4 enforces envelopes against
    the a10g_equiv total, NOT raw hours (critiques.md #14) — that is the whole point of
    storing both."""
    runs = terminal_runs(base, campaign)
    return {
        "gpu_hours": round(sum(r["cost"]["gpu_hours"] for r in runs), 4),
        "a10g_equiv_hours": round(sum(r["cost"]["a10g_equiv_hours"] for r in runs), 4),
        "n_runs": len(runs),
    }


def check_eval_pin_consistency(base: str, campaign: str) -> list[str]:
    """Return the distinct eval_pins in a campaign. More than one means some rows are being
    compared across different frozen evals — a bug the registry must make *detectable*
    (REQUIREMENTS R1: 'rows are only comparable within one pin'). Caller decides if the
    split is intended (e.g. the FM campaign's 112-fraud vs 2724-fraud tables)."""
    return sorted({r["eval_pin"] for r in terminal_runs(base, campaign)})


def conflicting_terminal_runs(base: str, campaign: str) -> list[str]:
    """run_ids that have more than one *distinct* terminal status on record (e.g. a buggy
    monitor wrote both FAILED and SUCCEEDED). Idempotency dedupes exact (run_id,status)
    pairs but cannot catch a conflicting pair — so, like cross-pin comparison, the registry
    must make it *detectable* rather than silently letting the last-written row win."""
    seen: dict[str, set] = {}
    for row in _read_rows(base, campaign):
        if row.get("type") == "run" and row["status"] in TERMINAL_STATES:
            seen.setdefault(row["run_id"], set()).add(row["status"])
    return sorted(rid for rid, statuses in seen.items() if len(statuses) > 1)


def reconstruct(base: str, campaign: str) -> dict:
    """The R1 'Done when' acceptance check, as a function: rebuild a campaign's full state
    from the JSONL alone — what ran, what it cost, what won, what was killed and why."""
    runs = terminal_runs(base, campaign)
    decisions = [r for r in _read_rows(base, campaign) if r.get("type") == "decision"]
    killed = [d for d in decisions if d["decision"] == "kill"]
    return {
        "campaign": campaign,
        "spend": campaign_spend(base, campaign),
        "eval_pins": check_eval_pin_consistency(base, campaign),
        "conflicting_terminal_runs": conflicting_terminal_runs(base, campaign),
        "runs": {r["run_id"]: r["status"] for r in runs},
        "promoted": [d["run_ids"] for d in decisions if d["decision"] == "promote"],
        "killed": {tuple(d["run_ids"]): d["reason"] for d in killed},
    }


def iter_campaigns(base: str) -> Iterator[str]:
    reg = os.path.join(base, "registry")
    if not os.path.isdir(reg):
        return
    for name in sorted(os.listdir(reg)):
        if name.endswith(".jsonl"):
            yield name[:-6]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        print(json.dumps({k: reconstruct(sys.argv[2], k) for k in iter_campaigns(sys.argv[2])},
                         indent=2))
    else:
        print(__doc__)
