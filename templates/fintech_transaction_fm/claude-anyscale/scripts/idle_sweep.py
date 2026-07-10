#!/usr/bin/env python3
"""
Idle-workspace autoscaler — DETECTION + DRY-RUN APPLY (read-only; nothing is modified).

Companion to ../recipes/slack-idle-workspace-sweep.md. For every RUNNING workspace in a
cloud it reads the inline compute config (`workspace_v2 get --json --verbose`), flags PINNED
GPU worker groups (GPU instance + min_nodes > 0) and GPU HEAD nodes, and prints the exact
`compute-config create` / `workspace_v2 update` commands that WOULD scale a pinned workspace
to zero.

Verified read-only against a real org (aws-public-us-west-2): correctly classified 5 running
workspaces, surfaced 2 real GPU-head nodes. The GPU-worker "WOULD scale to zero" branch is
exercised by unit logic but hadn't hit a real pinned case at the time of writing.

STILL STUBBED ON PURPOSE (needs a live box / explicit go):
  - the ray-status/nvidia-smi probe that confirms it's *actually idle right now*
    (this script flags on config + uptime, which is NOT the same as idle)
  - actually running the apply commands (they restart the workspace; keep gated)

Usage: python3 idle_sweep.py <cloud-name>
"""
import subprocess, sys, json, os
from datetime import datetime, timezone

CLOUD = sys.argv[1] if len(sys.argv) > 1 else "aws-public-us-west-2"
VERBOSE = os.environ.get("VERBOSE", "1").lower() not in ("0", "", "false", "no")


def sh(args):
    if VERBOSE:
        print("   $ " + " ".join(args))
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   ! command failed (exit {r.returncode}): {' '.join(args)}")
        if r.stderr.strip():
            print("   ! " + r.stderr.strip().replace("\n", "\n   ! "))
    elif VERBOSE:
        print(f"   > exit 0, {len(r.stdout)} bytes stdout")
    return r.stdout


def is_gpu(it):
    """AWS GPU/accelerator families: g*, p*, inf*, trn*."""
    fam = it.split(".")[0].lower()
    return bool(fam) and (fam[0] in ("g", "p") or fam.startswith("inf") or fam.startswith("trn"))


def uptime(ts):
    if not ts:
        return "?"
    started = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    h = (datetime.now(timezone.utc) - started).total_seconds() / 3600
    return f"{h/24:.0f}d" if h >= 48 else f"{h:.0f}h"


def main():
    all_ws = json.loads(sh(["anyscale", "workspace_v2", "list", "--json",
                            "--cloud", CLOUD]) or "[]")
    ws = json.loads(sh(["anyscale", "workspace_v2", "list", "--json",
                        "--state", "RUNNING", "--cloud", CLOUD]) or "[]")
    print(f"=== idle-sweep DRY-RUN · scanned {len(all_ws)} workspace(s) in {CLOUD} "
          f"· {len(ws)} RUNNING · nothing modified ===\n")

    would = flags = clean = 0
    for w in ws:
        d = json.loads(sh(["anyscale", "workspace_v2", "get", "--id", w["id"],
                           "--json", "--verbose"]) or "{}")
        cc = (d.get("config") or {}).get("compute_config") or {}
        head = (cc.get("head_node") or {}).get("instance_type", "?")
        workers = cc.get("worker_nodes") or []
        owner = w.get("creator_email", "?").split("+")[0]

        pinned_gpu = [x for x in workers if x.get("min_nodes", 0) and is_gpu(x.get("instance_type", ""))]
        gpu_head = is_gpu(head)

        hdr = f"• {w['name'][:38]:38} ({owner}, up {uptime(w.get('last_started_at'))})"
        if not pinned_gpu and not gpu_head:
            clean += 1
            print(hdr + f"\n   clean · head {head} · {len(workers)} worker group(s), no pinned GPU\n")
            continue
        print(hdr)
        print(f"   head {head} · workers: " +
              ", ".join(f"{x['name']}={x['instance_type']}/min{x.get('min_nodes')}" for x in workers))
        if gpu_head:
            flags += 1
            print(f"   FLAG  GPU HEAD ({head}) — a head can't scale to zero; recommend a CPU head")
        if pinned_gpu:
            would += 1
            groups = ", ".join(f"{x['name']} ({x['instance_type']}) {x['min_nodes']}->0" for x in pinned_gpu)
            print(f"   WOULD scale to zero: {groups}")
            print(f"         # (dry-run) the apply would be:")
            print(f"         anyscale compute-config create -f {w['name']}-autozero.yaml -n {w['name']}-autozero")
            print(f"         anyscale workspace_v2 update {w['id']} --compute-config {w['name']}-autozero")
            print(f"         # gated: confirm idle via `run_command ... 'ray status'` first; applying restarts the ws")
        print()

    print(f"=== summary: {would} would-scale-to-zero · {flags} GPU-head flag(s) · {clean} clean ===")


if __name__ == "__main__":
    main()
