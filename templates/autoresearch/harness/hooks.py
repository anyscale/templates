"""Claude Code submit hooks — R4's enforcement, wired to physics.

`budget.py` admits its own gap: "the live `anyscale job submit` wiring is the last stubbed
piece, so today this refuses over-budget runs by discipline + the harness, not by physics."
This module is that wiring. Two entrypoints, driven by Claude Code hooks (stdin = the hook's
JSON payload):

    python3 hooks.py pre    # PreToolUse(Bash): budget-gate the submit BEFORE it runs
    python3 hooks.py post   # PostToolUse(Bash): commit the estimate to the registry (R1)

`pre` finds `anyscale job submit` in the command (including inside a quoted `workspace ssh`
payload), reads the run's declaration, estimates worst-case cost, and calls
`budget.preflight()`. Refusal = exit 2, reason on stderr — the harness blocks the tool call
and the agent reads why. `post` appends the RUNNING heartbeat row with the committed
estimate; `reconcile.py` (R6) later writes the terminal row with actuals.

The gate only arms when `AUTORESEARCH_BASE` (the registry root) is set — other work in the
repo is untouched. When armed it is strict: a submit with no declaration is refused, because
an undeclared run is exactly the kind the envelope math can't see.

A run declares itself with one comment line in its job YAML:

    # autoresearch: campaign=fintech_fm wave=1 rung=proxy est_hours=2.0 [eval_pin=sha256:...]

GPU shape comes from the YAML's compute_config (worst case: max_nodes x GPUs-per-node), so
the declaration can't understate the fleet; only the duration is trusted — and
`budget.wall_clock_timeout_s` exists precisely because durations lie.

Wiring (repo `.claude/settings.json`; see harness/README.md):

    {"hooks": {"PreToolUse":  [{"matcher": "Bash", "hooks": [{"type": "command",
        "command": "python3 templates/autoresearch/harness/hooks.py pre"}]}],
               "PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
        "command": "python3 templates/autoresearch/harness/hooks.py post"}]}]}}
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import budget  # noqa: E402
import registry  # noqa: E402

# instance family -> (gpu tier, GPUs per node by size suffix). Unknown instance = refusal:
# an un-priced GPU must not silently count as free budget (same stance as tier_weight).
_FAMILY = {"g4dn": "T4", "g5": "A10G", "g6e": "L40S", "g6": "L4", "p4d": "A100",
           "p4de": "A100", "p5": "H100"}
_MULTI_GPU_SIZE = {"12xlarge": 4, "24xlarge": 4, "48xlarge": 8, "metal": 8}
_CPU_FAMILIES = ("m5", "m6", "m7", "c5", "c6", "c7", "r5", "r6", "r7", "t3", "i3", "i4")

SUBMIT_RE = re.compile(r"anyscale\s+job\s+submit\b")
DECL_RE = re.compile(r"#\s*autoresearch:\s*(.+)")
JOB_ID_RE = re.compile(r"prodjob_[a-z0-9]+")
YAML_ARG_RE = re.compile(
    r"anyscale\s+job\s+submit\s+(?:-f\s+|--config-file\s+)?(\S+\.ya?ml)")


def gpu_shape(instance_type: str) -> tuple[str, int] | None:
    """(gpu tier, GPUs per node) for an instance type; None for CPU instances;
    raises for GPU-looking instances we can't price."""
    family = instance_type.split(".")[0]
    if any(family.startswith(c) for c in _CPU_FAMILIES):
        return None
    if family not in _FAMILY:
        raise ValueError(f"unknown instance family {family!r} — add it to hooks._FAMILY "
                         f"or use a priced GPU (BUDGET_POLICY.md table)")
    size = instance_type.split(".", 1)[1] if "." in instance_type else "xlarge"
    return _FAMILY[family], _MULTI_GPU_SIZE.get(size, 1)


def parse_compute(yaml_text: str) -> dict:
    """Worst-case GPU fleet from a job YAML's compute_config — targeted stdlib parse
    (instance_type / max_nodes / market_type lines), no yaml dependency. Multi-tier GPU
    configs are refused: one run, one GPU tier, or the cost record lies (registry stores a
    single gpu_type per row)."""
    groups, cur = [], {}
    for line in yaml_text.splitlines():
        s = line.strip()
        if s.startswith("- ") or s.startswith("head_node"):
            if cur.get("instance_type"):
                groups.append(cur)
            cur = {}
            s = s[2:].strip() if s.startswith("- ") else s
        for key in ("instance_type", "max_nodes", "market_type"):
            m = re.match(rf"{key}:\s*(\S+)", s)
            if m:
                cur[key] = m.group(1)
    if cur.get("instance_type"):
        groups.append(cur)

    tiers: dict[str, dict] = {}
    on_demand = False
    for g in groups:
        shape = gpu_shape(g["instance_type"])
        if shape is None:
            continue
        tier, per_node = shape
        n = int(g.get("max_nodes", 1)) * per_node
        t = tiers.setdefault(tier, {"num_gpus": 0, "instance_type": g["instance_type"]})
        t["num_gpus"] += n
        if g.get("market_type", "ON_DEMAND").upper() != "SPOT":
            on_demand = True
    if not tiers:
        return {"gpu_type": None, "num_gpus": 0, "spot": True}
    if len(tiers) > 1:
        raise ValueError(f"multiple GPU tiers in one job ({sorted(tiers)}) — split into "
                         f"separate runs so each cost record has one gpu_type")
    tier, t = next(iter(tiers.items()))
    return {"gpu_type": tier, "num_gpus": t["num_gpus"],
            "instance_type": t["instance_type"], "spot": not on_demand}


def parse_declaration(yaml_text: str) -> dict | None:
    """The `# autoresearch: k=v ...` header, or None."""
    m = DECL_RE.search(yaml_text)
    if not m:
        return None
    decl = dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)
    return decl or None


def find_yaml(command: str) -> str | None:
    m = YAML_ARG_RE.search(command)
    return m.group(1) if m else None


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _refuse(msg: str) -> int:
    print(f"[autoresearch budget gate] REFUSED: {msg}", file=sys.stderr)
    return 2


def pre(payload: dict, base: str) -> int:
    command = (payload.get("tool_input") or {}).get("command", "")
    if not SUBMIT_RE.search(command):
        return 0
    yaml_path = find_yaml(command)
    if not yaml_path or not os.path.exists(yaml_path):
        return _refuse("could not locate the job YAML in this command — submit a declared "
                       "YAML (`anyscale job submit <file.yaml>`) so the gate can price it")
    text = open(yaml_path).read()
    decl = parse_declaration(text)
    if decl is None:
        return _refuse(f"{yaml_path} has no `# autoresearch: campaign=... wave=... rung=... "
                       f"est_hours=...` header — an undeclared run can't be enforced")
    missing = [k for k in ("campaign", "wave", "rung", "est_hours") if k not in decl]
    if missing:
        return _refuse(f"declaration missing {missing}")
    try:
        fleet = parse_compute(text)
    except ValueError as e:
        return _refuse(str(e))
    hours = float(decl["est_hours"])
    if fleet["gpu_type"] is None:
        print(f"[autoresearch budget gate] CPU-only run '{yaml_path}' — allowed, "
              f"ledgered at 0 GPU-hours")
        return 0
    est = budget.estimate(fleet["gpu_type"], fleet["num_gpus"], hours)
    v = budget.preflight(base, decl["campaign"], int(decl["wave"]), decl["rung"], est)
    if v["allowed"]:
        print(f"[autoresearch budget gate] OK: {est['a10g_equiv_hours']} A10G-eq committed "
              f"(~${budget.to_usd(est['a10g_equiv_hours'], fleet['spot'])}), "
              f"{v['remaining']} of {v['envelope']} remaining — {v['reason']}")
        return 0
    return _refuse(f"{v['reason']} (est {v['est_eq']} A10G-eq, spent {v['spent']}, "
                   f"envelope {v['envelope']})")


def post(payload: dict, base: str) -> int:
    command = (payload.get("tool_input") or {}).get("command", "")
    if not SUBMIT_RE.search(command):
        return 0
    yaml_path = find_yaml(command)
    if not yaml_path or not os.path.exists(yaml_path):
        return 0  # pre() already refused undeclared submits; nothing to ledger
    text = open(yaml_path).read()
    decl = parse_declaration(text)
    if decl is None:
        return 0
    fleet = parse_compute(text)
    m = JOB_ID_RE.search(json.dumps(payload.get("tool_response", "")))
    name_m = re.search(r"^name:\s*(\S+)", text, re.M)
    run_id = m.group(0) if m else (name_m.group(1) if name_m else os.path.basename(yaml_path))
    hours = float(decl["est_hours"])
    gpu_type = fleet["gpu_type"] or "A10G"  # CPU-only rows: 0 hours, tier irrelevant
    est_eq = budget.estimate(gpu_type, fleet["num_gpus"], hours)["a10g_equiv_hours"]
    registry.append_run(base, {
        "campaign": decl["campaign"], "run_id": run_id, "commit": _git_commit(),
        "rung": decl["rung"], "eval_pin": decl.get("eval_pin", "unpinned"),
        "status": "RUNNING",
        "cost": {"gpu_hours": round(fleet["num_gpus"] * hours, 4), "gpu_type": gpu_type,
                 "usd_est": budget.to_usd(est_eq, fleet["spot"]), "spot": fleet["spot"]},
        "wave": int(decl["wave"]), "job_yaml": yaml_path,
        "fleet": fleet, "est_hours": hours,
        "cost_basis": "committed estimate (max_nodes x est_hours); reconcile.py writes actuals",
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"[autoresearch ledger] RUNNING row committed for {run_id} "
          f"({decl['campaign']}, {decl['rung']})")
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("pre", "post"):
        print(__doc__)
        return 1
    base = os.environ.get("AUTORESEARCH_BASE")
    if not base:
        return 0  # gate not armed — never interfere with non-campaign work
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if (payload.get("tool_name") or "") != "Bash":
        return 0
    return pre(payload, base) if mode == "pre" else post(payload, base)


if __name__ == "__main__":
    sys.exit(main())
