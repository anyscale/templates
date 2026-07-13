"""R2 — experiment launcher (spec generation; the live submit is the PI boundary).

One command per experiment: an `experiment` dict (base config + flag deltas + rung + seeds +
compute) → a generated Anyscale job spec, with the budget check (R4) run first and every
money-saving default from `BUDGET_POLICY.md` baked in so the human never hand-edits a job YAML
(the FM campaign had ~15 hand-written ones, two of which hosted infra bugs).

**The one thing this module does NOT do is submit.** `submit()` stops at the boundary the PI
defended by hand — money + irreversibility — and returns the exact CLI it *would* run instead
of running it. Wiring `anyscale job submit` in is a deliberate, separately-approved step.

Baked-in defaults (so the agent never has to remember them):
- spot + `fallback_to_on_demand` on GPU workers; on-demand only for the head.
- `min_nodes: 0` scale-to-zero on worker groups; `resources: {CPU: 0}` fence on GPU groups.
- wall-clock timeout from the cost estimate (`budget.wall_clock_timeout_s`) — the runtime kill.
- run name encodes campaign · rung · flags · seeds so `git log`/TensorBoard are greppable
  (Iron Rule #5).
- `working_dir` must be committed+pushed git; the entrypoint threads the eval pin + registry
  path so the run writes its own R1 rows and moves prior artifacts aside (R8).
"""

from __future__ import annotations

import budget

# GPU tier -> AWS instance (BUDGET_POLICY.md conversion table)
GPU_INSTANCE = {
    "T4": "g4dn.xlarge", "L4": "g6.xlarge", "A10G": "g5.xlarge",
    "L40S": "g6e.xlarge", "A100": "p4de.24xlarge", "H100": "p5.48xlarge",
}


class BudgetError(RuntimeError):
    """Raised when a run is refused at submit time (over cap / over envelope)."""


def _flag_args(flags: dict) -> str:
    # one experiment = one flag delta, default OFF (Iron Rule #8): only emit truthy flags;
    # a flag set False/None is "off" and must be omitted, not passed as "--flag False".
    parts = []
    for k in sorted(flags):
        v = flags[k]
        if v is False or v is None:
            continue
        parts.append(f"--{k}" if v is True else f"--{k} {v}")
    return " ".join(parts)


def run_name(exp: dict) -> str:
    on = {k: v for k, v in exp.get("flags", {}).items() if v is not False and v is not None}
    flat = "_".join(k if on[k] is True else f"{k}{on[k]}" for k in sorted(on)) or "base"
    return f"{exp['campaign']}-{exp['rung']}-{flat}-s{len(exp.get('seeds', [0]))}"


def build_job_spec(exp: dict, base: str | None = None) -> dict:
    """Generate the Anyscale job spec for one experiment. If `base` (a registry root) is
    given, run the R4 budget preflight first: refuse an over-cap/over-envelope run outright,
    and mark a `full` run as approval-required rather than auto-submitting it."""
    for req in ("campaign", "wave", "rung", "gpu", "num_gpus", "hours", "entrypoint",
                "base_config", "working_dir", "image"):
        if req not in exp:
            raise ValueError(f"experiment missing '{req}'")
    if exp["gpu"] not in GPU_INSTANCE:
        raise ValueError(f"unknown gpu {exp['gpu']!r}")

    est = budget.estimate(exp["gpu"], exp["num_gpus"], exp["hours"])
    verdict = None
    if base is not None:
        verdict = budget.preflight(base, exp["campaign"], exp["wave"], exp["rung"], est)
        if not verdict["allowed"] and not verdict["escalate"]:
            raise BudgetError(f"refused: {verdict['reason']}")
    # A `full` run always needs PI sign-off — that's a policy fact about crossing into full,
    # independent of whether a registry base was supplied to check spend against.
    needs_approval = (exp["rung"] == "full") or bool(verdict and verdict["escalate"])

    name = run_name(exp)
    flags = _flag_args(exp.get("flags", {}))
    seeds = exp.get("seeds", [0])
    entry = (f"python {exp['entrypoint']} --config {exp['base_config']} {flags} "
             f"--seeds {','.join(map(str, seeds))} "
             f"--registry {exp.get('registry_root','$BASE/registry')} "
             f"--eval-pin {exp.get('eval_pin','$EVAL_PIN')} --rung {exp['rung']}").replace("  ", " ")

    spec = {
        "name": name,
        "entrypoint": entry.strip(),
        "working_dir": exp["working_dir"],           # must be committed + pushed
        "image": exp["image"],
        "max_retries": 3,
        "timeout_s": budget.wall_clock_timeout_s(exp["hours"]),
        "compute_config": {
            "head_node": {"instance_type": exp.get("head_instance", "m6i.2xlarge"),
                          "market": "ON_DEMAND"},   # stateful head: never spot
            "worker_nodes": [{
                "instance_type": GPU_INSTANCE[exp["gpu"]],
                "min_nodes": 0,                       # scale-to-zero
                "max_nodes": int(exp["num_gpus"]),
                "market": "SPOT",
                "fallback_to_on_demand": True,
                "resources": {"CPU": 0},              # fence: CPU stages can't scale GPUs up
            }],
        },
        "env_vars": exp.get("env_vars", {}),
        "estimate": est,
        "budget": verdict,
        "approval_required": needs_approval,
    }
    return spec


def submit(spec: dict, pi_approved: bool = False) -> str:
    """THE BOUNDARY. Does not call Anyscale. Returns the CLI that would submit the spec, or
    refuses if the run needs PI approval and doesn't have it. Wiring the real
    `anyscale job submit` here is a separately-approved step (money + irreversibility)."""
    if spec.get("approval_required") and not pi_approved:
        raise BudgetError(f"'{spec['name']}' crosses into a full run — needs PI sign-off "
                          f"(pass pi_approved=True once the PI has signed). Not submitted.")
    # NOTE: intentionally not executed. This is the escalation boundary.
    return (f"anyscale job submit --name {spec['name']} "
            f"--working-dir {spec['working_dir']} --image {spec['image']} "
            f"-- {spec['entrypoint']}")
