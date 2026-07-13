"""The job contract — what makes the engine work on ANY ML job.

The engine can hill-climb an arbitrary job only if the job speaks one small interface. That
interface is deliberately tiny so wrapping an existing training/eval script is a few lines:

  A conforming job MUST:
   1. accept a base config + zero-or-one flag delta (one experiment = one change),
   2. write a per-example **eval output** (the standard record below) to a known path,
   3. append its **registry row** (`registry.append_run`) with the decision metric + cost.

Given (2), every downstream tool — `metrics`, `erroranalysis`, the recipe, the controller —
works on the job without knowing anything about its domain. This module validates the contract
and enforces the confound firewall (one flag = one change), so a non-conforming job fails loud
at the edge instead of quietly poisoning the ledger.
"""

from __future__ import annotations

# A per-example eval record. Ranking tasks carry `score`; classification carries `y_pred`.
# `slices` (optional) is what error analysis segments on — the more, the better the diagnosis.
REQUIRED = ("id", "y_true")
ONE_OF = ("score", "y_pred")


def validate_eval_output(records) -> list:
    """Return a list of problems (empty = conforms). Checks the standard record shape so any
    job's eval output plugs into metrics/erroranalysis unchanged."""
    problems = []
    if not isinstance(records, list) or not records:
        return ["eval output must be a non-empty list of records"]
    ids = set()
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            problems.append(f"record {i}: not an object"); continue
        for k in REQUIRED:
            if k not in r:
                problems.append(f"record {i}: missing '{k}'")
        if not any(k in r for k in ONE_OF):
            problems.append(f"record {i}: needs one of {ONE_OF}")
        rid = r.get("id")
        if rid in ids:
            problems.append(f"record {i}: duplicate id {rid!r}")
        ids.add(rid)
        if "slices" in r and not isinstance(r["slices"], dict):
            problems.append(f"record {i}: 'slices' must be an object")
    return problems


def single_delta(base_flags: dict, exp_flags: dict) -> str:
    """The confound firewall as code (Iron Rule #8). Return the ONE flag that differs between
    a base config and an experiment. Raise if zero (nothing to test) or more than one (a
    confounded experiment whose result can't be attributed — the thing that breaks compounding).
    """
    keys = set(base_flags) | set(exp_flags)
    changed = [k for k in keys if base_flags.get(k) != exp_flags.get(k)]
    if len(changed) == 0:
        raise ValueError("no delta: experiment is identical to base — nothing to test")
    if len(changed) > 1:
        raise ValueError(f"confounded experiment: {len(changed)} flags differ ({sorted(changed)}) "
                         f"— one experiment = one change, or the result can't be attributed")
    return changed[0]


def describe_delta(base_flags: dict, exp_flags: dict) -> str:
    k = single_delta(base_flags, exp_flags)
    return f"{k}: {base_flags.get(k)!r} -> {exp_flags.get(k)!r}"
