#!/usr/bin/env python3
"""Sample cluster + Ray Train progress once per minute during nb 04 pretrain.

Writes a durable, appendable log to /mnt/cluster_storage so runtimes survive a
node loss. Detects trial start (first new checkpoint/result dir), captures the
latest reported step/epoch/loss, and stamps wall-clock deltas. Read-only.
"""
import csv
import glob
import json
import os
import subprocess
import time
from datetime import datetime, timezone

BASE = "/mnt/cluster_storage/transaction-fm"
RESULTS = os.path.join(BASE, "ray_results", "transaction_fm_pretrain")
LOG = os.path.join(BASE, "PRETRAIN_MONITOR.log")
INTERVAL = 60

# Only trials created after the monitor starts count as "this run".
START_WALL = time.time()


def ts():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def ray_status():
    try:
        out = subprocess.run(["ray", "status"], capture_output=True, text=True, timeout=20).stdout
    except Exception as e:
        return f"(ray status failed: {e})"
    gpu = next((l.strip() for l in out.splitlines() if "GPU" in l and "/" in l), "GPU ?")
    nodes = out.count("1xa10g") + out.count("A10G")
    pending = "PENDING-DEMAND" if "no resource demands" not in out else "no-demand"
    active = out.split("Active:")[1].split("Idle:")[0].strip().replace("\n", "; ") if "Active:" in out else "?"
    return f"gpu[{gpu}] pending[{pending}] active[{active}]"


def newest_trial():
    dirs = [d for d in glob.glob(os.path.join(RESULTS, "*")) if os.path.isdir(d)]
    dirs = [d for d in dirs if os.path.getmtime(d) >= START_WALL - 120]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def latest_progress(trial):
    """Return the last row of progress.csv / result.json if present."""
    if not trial:
        return "no-new-trial"
    for name in ("progress.csv",):
        p = os.path.join(trial, name)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    r = rows[-1]
                    keys = [k for k in ("epoch", "step", "training_iteration", "loss", "train_loss", "perplexity", "ppl") if k in r]
                    return "csv:" + " ".join(f"{k}={r[k]}" for k in keys) + f" (rows={len(rows)})"
            except Exception as e:
                return f"csv-read-err:{e}"
    rj = os.path.join(trial, "result.json")
    if os.path.exists(rj):
        try:
            last = None
            with open(rj) as f:
                for line in f:
                    if line.strip():
                        last = line
            if last:
                d = json.loads(last)
                keys = [k for k in ("epoch", "step", "training_iteration", "loss", "train_loss", "perplexity", "ppl", "time_total_s") if k in d]
                return "json:" + " ".join(f"{k}={d[k]}" for k in keys)
        except Exception as e:
            return f"json-read-err:{e}"
    return f"trial={os.path.basename(trial)} (no progress file yet)"


def main():
    with open(LOG, "a") as f:
        f.write(f"\n===== monitor start {ts()} (interval {INTERVAL}s) =====\n")
        f.flush()
        trial_seen = None
        trial_start = None
        while True:
            trial = newest_trial()
            if trial and trial != trial_seen:
                trial_seen = trial
                trial_start = time.time()
                f.write(f"{ts()}  TRIAL DETECTED: {os.path.basename(trial)}\n")
            elapsed = f"{(time.time()-trial_start)/60:.1f}m-into-trial" if trial_start else "pre-trial"
            line = f"{ts()}  [{elapsed}]  {ray_status()}  || {latest_progress(trial)}"
            f.write(line + "\n")
            f.flush()
            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
