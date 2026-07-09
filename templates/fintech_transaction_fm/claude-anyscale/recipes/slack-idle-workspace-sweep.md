# Recipe: Auto-scale idle workspaces down to zero (+ flag GPU head nodes)

**The ask:** *"If someone left a workspace pinned so its GPU workers won't scale down, and it's
been idle over an hour on both the head and the workers — just fix it. Flip the compute config to
scale-to-zero with the same GPU type, so their environment survives and the GPUs come back the
moment they run something. Don't wait for a Slack reply; the only idle cost left should be a CPU
head node. And separately, flag GPU head nodes, since a head can't scale to zero."*

**This is a posture shift.** The nudge-and-wait version (@-mention, wait for `keep`/`stop`) saved
nothing until a human replied. This one *acts*: it re-provisions the idle workspace to scale-to-zero
and notifies the owner **after**. Cost stops accruing immediately, no human in the loop — and it's
non-destructive, so nobody loses their setup.

**Skills/tools:** `anyscale compute-config get/create` + `workspace_v2 update` (the fix),
`workspace_v2 run_command` (idle probe), `workspace_v2 tags` (opt-out), a Slack webhook (post-hoc
notice), `/schedule` (cadence).

> **Why scale-to-zero, not terminate:** terminate kills the whole environment. Scale-to-zero keeps
> the workspace alive — same image, same mounts, same worker *definition* — just with GPU workers at
> `min_nodes: 0`. The owner returns to their exact setup, and the first job they launch autoscales a
> GPU worker back. The only thing that stops costing money is the GPU that was sitting there idle.

---

## Config knobs

```bash
REGION="us-west-2"
IDLE_MIN=60                       # idle this long on HEAD *and* workers before acting
DRY_RUN=true                      # default: report what it WOULD change; flip to false to enforce
EXCLUDE_TAG="no-autoscale-fix"    # workspaces carrying this tag are left alone
SLACK_CHANNEL="#anyscale-cost"
STATE=/mnt/user_storage/idle-sweep
```

## The logic

```
1. list RUNNING workspaces in REGION   (anyscale workspace_v2 list, joined to cloud→region)
   skip any carrying EXCLUDE_TAG        (anyscale workspace_v2 tags)

2. idle probe — HEAD *and* workers — via run_command:
     anyscale workspace_v2 run_command --id <id> -- \
       'ray status; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits'
     idle? ← ray status shows no GPUs in use + no pending demand, AND max(gpu%) < a few %
             on every node (head included)
     stamp idle_since in $STATE/idle_<id> (clear it the moment it's busy);
     require now − idle_since ≥ IDLE_MIN

3. read the compute config:  anyscale compute-config get <config-name-from-workspace>
     • GPU worker group with min_nodes > 0   → PINNED  (this is the waste)
     • head node is a GPU instance           → GPU HEAD (can't scale to zero)

4. remediate:
   (a) PINNED idle GPU workers:
        - new config = current config with those groups' min_nodes: 0
          (KEEP node type, max_nodes, market type — only min_nodes changes)
        - anyscale compute-config create -f new.yaml -n <name>-autozero
        - anyscale workspace_v2 update <id> --compute-config <name>-autozero
        - ⚠️ applying re-provisions the cluster → the (idle) workspace RESTARTS.
          /mnt/* persists; unsaved in-memory state (rare on an idle box) does not.
        - if DRY_RUN: report the diff ("min_nodes 1→0 on group `gpu`"), don't apply.
   (b) GPU HEAD idle:
        - a head can't scale to zero (it's always up while the workspace is RUNNING)
        - FLAG only — recommend a CPU head or terminating. Swapping the head type is a
          bigger, owner-owned decision; don't auto-do it.

5. notify the owner in Slack AFTER acting (a notice, not a gate):
     owner ← anyscale workspace_v2 list CREATED BY → slack users.lookupByEmail → <@U…>
```

## What lands in Slack (post-hoc)

```
🔧 idle-workspace autoscaler · region us-west-2 · idle > 60m on head + workers

@alice  ✅ scaled *team-scratch-gpu* to zero.
        • was: 1× g5.12xlarge worker pinned (min_nodes=1), idle 3h14m (~$18 burned)
        • now: same worker group at min_nodes=0 — a GPU node autoscales back when you run something
        • only the CPU head remains. /mnt/user_storage untouched. re-pin anytime,
          or tag `no-autoscale-fix` to opt this workspace out.

@bob    ⚠️ *vision-demo* has a GPU HEAD node (g5.xlarge), idle 5h — a head can't scale to zero.
        consider a CPU head or terminating. I didn't touch it (head swap is your call).
```

## Safety rails

- **DRY_RUN by default.** First runs only report the diff ("would set `min_nodes` 1→0 on group
  `gpu`"); you flip to enforce once the reports look right.
- **Opt-out tag.** Anyone can `anyscale workspace_v2 tags` a workspace `no-autoscale-fix` and it's
  skipped — for the person who genuinely needs a warm GPU worker pinned.
- **Non-destructive + reversible.** Only `min_nodes` changes; node types, max, image, mounts are
  untouched, and any owner can re-pin. Nothing is deleted or terminated.
- **The one real caveat:** applying a compute config **re-provisions the workspace — it restarts.**
  Fine for a box idle > 1h (persistent storage survives), but it's a restart, not a live edit —
  confirm the apply semantics in your org before flipping `DRY_RUN` off.
- **Head nodes are flagged, never auto-changed** — a head-type swap is disruptive and owner-owned.

## Where the idle signal comes from

The utilization behind all of this is real platform data — Ray's per-node metrics, aggregated by
Anyscale and shown on the cloud **Dashboard** + **Grafana**. Three ways to read it, pick by access:

1. **`anyscale aggregated-instance-usage download-csv --start-date … --end-date … [--cloud …]`** —
   the CLI export behind the dashboard's "unused GPU-hours by workload" panels; the cleanest source
   for the *cost* numbers. ⚠️ Needs org/billing permission (a normal token gets `Permission denied`).
2. **Grafana / Prometheus** — the "View in Grafana" backend; `ray_node_gpus_utilization` &c. with an
   API token, live + historical.
3. **`run_command … 'ray status; nvidia-smi'`** — reads live state off the cluster, no special
   permission. This is what step 2 uses for the instant "idle right now?" check; idle *duration*
   comes from your own `idle_since` sampling.

> **Anyscale already idle-terminates workspaces** (one self-terminated while I wrote these guides).
> This recipe is the gentler cousin: instead of killing a workspace whose owner disabled or lengthened
> the idle timeout, it just removes the *pinned GPU workers* and leaves the workspace usable.

## Turn it into a routine

`/schedule` it hourly during work hours. Keep `DRY_RUN=true` for a week, read the "would change"
notices, then enforce. Org-wide idle GPU spend collapses to "a handful of CPU head nodes."
