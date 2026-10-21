"""What a job actually used: GPU-hours, CPU-hours and cost, from its cluster's event log.

`reconcile.py` used to write wall-clock x the YAML's max fleet, an upper bound that can be off
several-fold (an 8-GPU-max config that ran on 2 GPUs). This module reads the job's own
cluster event log from Anyscale and rebuilds every node's lifetime instead:

- **Start:** each `Launched N instance(s)` event, which names the instance type, AZ and market.
- **End:** the first evidence the node is gone: an explicit `Terminating N <type>/<market>`
  event; a `Cluster resized to {.. GPU}` drop (the most recently launched GPU nodes are closed
  first); a new head launch (the cluster restarted); or `Cluster is terminated`.
- **Cross-check:** the autoscaler's own `Cluster upscaled/resized to {C CPU, G GPU}` totals are
  integrated separately. The event log occasionally skips a launch (seen on a spot
  replacement), so GPU-time the autoscaler reports beyond the launch ledger is counted too, as
  `unattributed`, priced at the job's most expensive GPU. Nothing reported is dropped.

Cost comes from `prices.py` (AWS's published on-demand list price, EC2 spot price history for
spot nodes), never from the model. All reads are read-only, and stdlib-only: the workspace's
`ANYSCALE_CLI_TOKEN`, or `~/.anyscale/credentials.json` on a laptop.

Known gaps: the event log is not a billing record, so where it says nothing about a node
leaving (an idle scale-down it didn't log), the node is counted until the next evidence. That
makes the result an upper bound that is usually exact. A job is assumed to use the one cluster
its state names (true for every PoC job checked, retries included). The Anyscale platform
charge is not included.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hooks  # noqa: E402
import prices  # noqa: E402
import registry  # noqa: E402

LAUNCH_RE = re.compile(r"\[autoscaler\] \[(?P<group>[^\]|]+)\|(?P<itype>[^\]]+)\] "
                       r"\[(?P<az>[^\]]+)\] \[(?P<market>spot|on-demand)\] "
                       r"Launched (?P<n>\d+) instances?\.")
TERMINATING_RE = re.compile(r"Terminating (?P<n>\d+) (?P<itype>[^/\s]+)/(?P<market>spot|on-demand)"
                            r" instances")
RESIZE_RE = re.compile(r"Cluster (?:upscaled|resized|downscaled) to \{(?P<cpu>\d+) CPU"
                       r"(?:, (?P<gpu>\d+) GPU)?\}")
TERMINATED_RE = re.compile(r"^Cluster is terminated")
EVENTS_PAGE = 1000  # the API's cap per call; it returns no paging token past it
# HA-job states -> the registry's terminal vocabulary.
STATE_MAP = {"SUCCESS": "SUCCEEDED", "BROKEN": "ERRORED"}


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# --- read-only API access ----------------------------------------------------------------
def _auth() -> tuple[str, str]:
    host = os.environ.get("ANYSCALE_HOST", "https://console.anyscale.com").rstrip("/")
    token = os.environ.get("ANYSCALE_CLI_TOKEN")
    if not token:
        cred = os.path.expanduser("~/.anyscale/credentials.json")
        with open(cred) as f:
            token = json.load(f)["cli_token"]
    return host, token


def _get(path: str) -> dict:
    host, token = _auth()
    req = urllib.request.Request(host + path, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def job_record(job_id: str) -> dict:
    """{state, cluster_id, created_at, state_at} for a production job."""
    r = _get(f"/api/v2/decorated_ha_jobs/{job_id}")["result"]
    st = r.get("state") or {}
    state = st.get("current_state") or "UNKNOWN"
    return {"state": STATE_MAP.get(state, state), "cluster_id": st.get("cluster_id"),
            "created_at": r.get("created_at"), "state_at": st.get("state_transitioned_at")}


def cluster_events(cluster_id: str) -> list[dict]:
    """Every event for one cluster, oldest first. Pages by start_time, since the API stops at
    1,000 events without a paging token."""
    out, seen, start = [], set(), None
    while True:
        q = f"/api/v2/sessions/{cluster_id}/cluster_events?count={EVENTS_PAGE}&ascending_order=true"
        if start:
            q += "&start_time=" + urllib.parse.quote(start)
        page = _get(q)["results"]
        fresh = [e for e in page if (e["created_at"], e["message"]) not in seen]
        for e in fresh:
            seen.add((e["created_at"], e["message"]))
        out += fresh
        if len(page) < EVENTS_PAGE or not fresh:
            return out
        start = page[-1]["created_at"]


# --- the node ledger (pure) ------------------------------------------------------------------
def gpus_per_node(itype: str, table: dict | None) -> int:
    row = (table or {}).get("instances", {}).get(itype)
    if row:
        return row["gpus"]
    shape = hooks.gpu_shape(itype)
    return shape[1] if shape else 0


def node_intervals(events: list[dict], table: dict | None = None,
                   end: datetime | None = None) -> tuple[list[dict], list[tuple]]:
    """(nodes, autoscaler_steps) from a cluster's events.

    nodes: [{itype, market, az, head, gpus, start, end, end_reason}]
    autoscaler_steps: [(time, cpu, gpu)], the totals the autoscaler reported.
    """
    evs = sorted(events, key=lambda e: e["created_at"])
    open_: list[dict] = []
    nodes: list[dict] = []
    steps: list[tuple] = []
    reported_gpu = 0
    gpus_closed_since_report = 0

    def close(node, t, reason):
        node["end"], node["end_reason"] = t, reason
        open_.remove(node)

    for e in evs:
        t, msg = _ts(e["created_at"]), e["message"]
        if m := LAUNCH_RE.search(msg):
            head = m["group"] == "head-node"
            if head and nodes:  # a second head means the cluster restarted
                for n in list(open_):
                    close(n, t, "cluster restarted")
                steps.append((t, 0, 0))
                reported_gpu = 0
            for _ in range(int(m["n"])):
                node = {"itype": m["itype"], "market": m["market"], "az": m["az"],
                        "head": head, "gpus": gpus_per_node(m["itype"], table),
                        "start": t, "end": None, "end_reason": None}
                nodes.append(node)
                open_.append(node)
        elif m := TERMINATING_RE.search(msg):
            matching = [n for n in open_ if not n["head"] and n["itype"] == m["itype"]
                        and n["market"] == m["market"]]
            for n in matching[:int(m["n"])]:  # oldest first: replacements retire old nodes
                gpus_closed_since_report += n["gpus"]
                close(n, t, "terminated (event)")
        elif m := RESIZE_RE.search(msg):
            cpu, gpu = int(m["cpu"]), int(m["gpu"] or 0)
            need = reported_gpu - gpu - gpus_closed_since_report
            while need > 0:
                gpu_nodes = [n for n in open_ if not n["head"] and n["gpus"] > 0]
                if not gpu_nodes:
                    break
                n = gpu_nodes[-1]  # most recently launched first
                need -= n["gpus"]
                close(n, t, "cluster resized down")
            reported_gpu, gpus_closed_since_report = gpu, 0
            steps.append((t, cpu, gpu))
        elif TERMINATED_RE.search(msg):
            for n in list(open_):
                close(n, t, "cluster terminated")
            steps.append((t, 0, 0))
            reported_gpu = 0

    stop = end or (_ts(evs[-1]["created_at"]) if evs else None)
    for n in list(open_):
        close(n, max(stop, n["start"]), "still running at last event")
    return nodes, steps


def _gpu_timeline(nodes: list[dict], steps: list[tuple]) -> float:
    """GPU-hours the autoscaler reported beyond what the launch ledger explains."""
    points = sorted({n["start"] for n in nodes} | {n["end"] for n in nodes} | {s[0] for s in steps})
    extra = 0.0
    for a, b in zip(points, points[1:]):
        ledger = sum(n["gpus"] for n in nodes if n["start"] <= a < n["end"])
        reported = next((s[2] for s in reversed(steps) if s[0] <= a), 0)
        extra += max(0, reported - ledger) * (b - a).total_seconds() / 3600
    return round(extra, 4)


def _integral(steps: list[tuple], idx: int) -> float:
    return round(sum(s[idx] * (n[0] - s[0]).total_seconds() / 3600
                     for s, n in zip(steps, steps[1:])), 4)


def summarize(nodes: list[dict], steps: list[tuple], table: dict | None,
              spot_price=None) -> dict:
    """The cost record for a terminal registry row.

    spot_price(itype, az, start, end) -> $/hr or None. Pass None to price spot at on-demand.
    """
    inst = (table or {}).get("instances", {})
    by_type: dict[str, dict] = {}
    by_tier: dict[str, float] = {}
    gpu_hours = cpu_hours = usd = 0.0
    unpriced, spot_at_on_demand, no_vcpus = set(), set(), set()
    for n in nodes:
        hours = (n["end"] - n["start"]).total_seconds() / 3600
        key = f"{n['itype']}/{n['market']}"
        row = by_type.setdefault(key, {"nodes": 0, "hours": 0.0, "usd": 0.0})
        row["nodes"] += 1
        row["hours"] += hours
        gpu_hours += hours * n["gpus"]
        if n["gpus"]:
            tier = hooks.gpu_shape(n["itype"])[0]
            by_tier[tier] = by_tier.get(tier, 0.0) + hours * n["gpus"]
        if n["itype"] in inst:
            cpu_hours += hours * inst[n["itype"]]["vcpus"]
        else:
            no_vcpus.add(n["itype"])
        rate = inst.get(n["itype"], {}).get("on_demand_usd_hr")
        if n["market"] == "spot" and rate is not None:
            sp = spot_price(n["itype"], n["az"], n["start"], n["end"]) if spot_price else None
            if sp is None:
                spot_at_on_demand.add(n["itype"])
            else:
                rate = sp
        if rate is None:
            unpriced.add(n["itype"])
            continue
        row["usd"] += hours * rate
        usd += hours * rate

    extra = _gpu_timeline(nodes, steps)
    if extra:
        gpu_types = [n["itype"] for n in nodes if n["gpus"]]
        priciest = max(gpu_types, key=lambda t: inst.get(t, {}).get("on_demand_usd_hr", 0) /
                       max(1, gpus_per_node(t, table)), default=None)
        if priciest:
            tier = hooks.gpu_shape(priciest)[0]
            by_tier[tier] = by_tier.get(tier, 0.0) + extra
            rate = inst.get(priciest, {}).get("on_demand_usd_hr")
            if rate is not None:
                usd += extra * rate / max(1, gpus_per_node(priciest, table))
        gpu_hours += extra

    tiers = sorted(by_tier)
    wall = ((max(n["end"] for n in nodes) - min(n["start"] for n in nodes)).total_seconds() / 3600
            if nodes else 0.0)
    cost = {
        "gpu_hours": round(gpu_hours, 4),
        "cpu_hours": round(cpu_hours, 4),
        "gpu_type": tiers[0] if len(tiers) == 1 else ("mixed" if tiers else "A10G"),
        "by_tier": {k: round(v, 4) for k, v in by_tier.items()},
        "usd": round(usd, 2),
        "spot": any(n["market"] == "spot" for n in nodes),
        "wall_clock_hours": round(wall, 4),
        "instances": {k: {"nodes": v["nodes"], "hours": round(v["hours"], 4),
                          "usd": round(v["usd"], 2)} for k, v in sorted(by_type.items())},
        "unattributed_gpu_hours": extra,
        "autoscaler_gpu_hours": _integral(steps, 2),
        "autoscaler_cpu_hours": _integral(steps, 1),
    }
    flags = []
    if unpriced:
        flags.append(f"unpriced (not in the price table): {sorted(unpriced)}")
    if spot_at_on_demand:
        flags.append(f"spot priced at on-demand (no spot history): {sorted(spot_at_on_demand)}")
    if no_vcpus:
        flags.append(f"CPU-hours missing for: {sorted(no_vcpus)}")
    if flags:
        cost["flags"] = flags
    return cost


def job_usage(job_id: str, region: str = "us-west-2", spot: bool = True) -> dict | None:
    """Live lookup: the job's state plus its actual cost record. None if the cluster is unknown."""
    rec = job_record(job_id)
    if not rec["cluster_id"]:
        return None
    table = prices.get(region)
    events = cluster_events(rec["cluster_id"])
    end = _ts(rec["state_at"]) if rec["state"] in registry.TERMINAL_STATES and rec["state_at"] \
        else None
    nodes, steps = node_intervals(events, table, end)
    torn_down = any(TERMINATED_RE.search(e["message"]) for e in events)

    def spot_price(itype, az, start, stop):
        return prices.time_weighted(prices.spot_history(itype, az, start, stop), start, stop)

    cost = summarize(nodes, steps, table, spot_price if spot else None)
    basis = ("actual: node lifetimes from the cluster event log; AWS on-demand list prices "
             f"({(table or {}).get('publication_date', 'no price table')}, "
             f"{(table or {}).get('basis', 'none')})"
             + ("; spot from EC2 spot price history" if cost["spot"] and spot else ""))
    return {"state": rec["state"], "cluster_id": rec["cluster_id"], "cluster_terminated": torn_down,
            "cost": cost, "cost_basis": basis}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(json.dumps(job_usage(sys.argv[1]), indent=1, default=str))
