# Recipe: Poll-chat — idle-workspace cost sweep in Slack

**The ask:** *"Watch every workspace in a region. If someone's left one running with GPUs
allocated but barely using them, @-mention them in Slack with how long it's been idle and roughly
what it's cost — and let them reply `keep` / `stop` right there."*

**What Claude does:** on a schedule, lists the region's workspaces, checks each one's utilization
against a threshold *you* set, and for anything under it posts a Slack message that **@-mentions the
person who launched it** with idle time + estimated wasted spend + a console link. Because it's
*poll-chat* (see [`slack-job-buddy` pattern](../README.md#the-recipe-box)), it also reads the
thread so the owner can reply `keep` (snooze) or `stop` (terminate) and Claude acts on it.

**Skills/tools:** `anyscale` CLI (`workspace_v2 list`, `cloud list`) + Slack (webhook to post,
Slack API/MCP to read replies and map email→user-id) + `/schedule` to run it on a cadence.

> **Deliberate non-goal:** scaled-to-zero workspaces are **not** flagged. A "RUNNING" workspace
> with no worker nodes is ~free (that's the whole point of keeping one warm) — waste is *nodes
> allocated + low utilization*, not "a workspace exists."

---

## Config knobs (all yours to set)

```bash
REGION="us-west-2"          # clouds are region-scoped; sweep the workspaces whose cloud is here
MIN_UTIL_PCT=15             # flag anything under this GPU utilization…
IDLE_GRACE_MIN=120          # …sustained for at least this long (don't nag on a lunch break)
POLL_EVERY=300              # seconds between sweeps when run as a loop
SLACK_CHANNEL="#anyscale-cost"
STATE=/mnt/user_storage/idle-sweep   # small state dir: remembers when each ws first went idle
# node $/hr lookup lives in AGENTS.md (or a small table); used to turn idle-hours into $ wasted
```

## The logic

```
1. clouds ← anyscale cloud list                 # id → region
   workspaces ← anyscale workspace_v2 list        # name, id, STATE, cloud, CREATED BY (email)
   keep those whose cloud.region == REGION and STATE == RUNNING
   (TERMINATED = already gone; Anyscale idle-terminates on its own — see note)

2. for each RUNNING candidate — read utilization STRAIGHT FROM THE CLUSTER:
     probe ← anyscale workspace_v2 run_command --id <id> -- \
               'ray status 2>/dev/null; echo ---; \
                nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null'
     # ray status → nodes up + GPUs/CPUs USED vs total + any pending demand
     # nvidia-smi  → hardware GPU util %, per GPU
     busy? ← (ray status shows GPUs in use OR pending demand) OR max(gpu%) >= MIN_UTIL_PCT
     if busy: clear idle marker ($STATE/idle_<wsid>); skip

3. idle bookkeeping — this is your "idle time so far":
     idle_since ← $STATE/idle_<wsid>   # written the FIRST cycle we saw it idle (persists on /mnt/user_storage)
     idle_min   ← now − idle_since
     if idle_min < IDLE_GRACE_MIN: skip                      # don't nag on a lunch break
     node_type  ← from the workspace's compute config
     $wasted    ← unused-hours from aggregated-instance-usage (if you have access), else
                  (idle_min/60) * $per_hr(node_type) from your own sampling
     owner_id   ← slack users.lookupByEmail(CREATED BY)      # email → <@U…> for the @-mention

4. post to SLACK_CHANNEL: @owner + workspace + util% + idle time + $wasted + console link
                          + reply options (keep / stop / ignore)

5. (poll-chat) read the thread each cycle: `keep` → snooze 24h; `stop` → confirm, then
   anyscale workspace_v2 terminate; `ignore` → drop it for this sweep.
```

**Where the numbers come from — the platform has all of this.** Ray exports per-node CPU/GPU
utilization from every cluster; Anyscale aggregates it and renders the console **Dashboard**
(Summary / Instance utilization / Resource utilization) and **Grafana** ("View in Grafana"). The
"unused CPU/GPU-hours by workload" panels are allocated-minus-used hours, attributed per workload
with owner — i.e. this recipe, minus the Slack nudge. Three ways to get at it programmatically,
pick by access + freshness:

1. **`anyscale aggregated-instance-usage download-csv --start-date … --end-date … [--cloud …]`** —
   the CLI export of that usage data (zipped CSV, per instance, filterable by cloud/project). The
   cleanest "unused GPU-hours by workload" source for a scheduled cost sweep.
   ⚠️ Needs org/billing permission — a normal user token gets `Permission denied`. Check access first.
2. **Grafana / Prometheus** — the "View in Grafana" backend. Query `ray_node_gpus_utilization` &c.
   with a Grafana API token; good for live *and* historical, if you have the endpoint.
3. **`anyscale workspace_v2 run_command --id <id> -- 'ray status; nvidia-smi …'`** — reads the live
   state straight off the cluster, no special permission. Best for the instant "is it idle right
   now" check at nudge time. This is the fallback that always works, so it's what step 2 uses above.

If you can't get (1), the "idle time so far" comes from *your own sampling*: the poll agent stamps `idle_since`
the first cycle it sees a workspace idle and clears it the moment it's busy — so idle duration and
$ wasted are measured, not guessed.

> **Anyscale already idle-terminates workspaces.** One of the workspaces I listed for these guides
> auto-terminated mid-writeup — the platform's own idle timeout at work. So this sweep is the
> *second* line of defense: it catches workspaces whose idle-termination is disabled or set long,
> or that pin worker nodes up. Worth checking a workspace's idle-termination setting before nagging
> its owner.

## What it looks like in Slack

```
🧹 idle-workspace sweep · region us-west-2 · flagging < 15% GPU util sustained > 2h

@alice  your workspace *team-scratch-gpu* has been < 6% GPU util for 3h14m.
        • cluster: 1× g5.12xlarge (4× A10G), nodes up since 09:12
        • idle ≈ 3h14m  →  ≈ $18.30 wasted so far (≈ $5.67/hr)
        • 🔗 console.anyscale.com/workspaces/expwrk_6b9f…
        reply in-thread:  `keep` (snooze 24h) · `stop` (I'll terminate) · `ignore`

@bob    *vision-demo* — 0% GPU util 5h+, 1× g5.xlarge → ≈ $6.05 wasted.  same options.

2 flagged · ≈ $XX/day if left as-is. (scaled-to-zero workspaces are fine — not shown.)
```

…and the reply half:

```
alice  ▸  keep, I'm running an eval tonight
claude ▸  👍 snoozed *team-scratch-gpu* for 24h, @alice — won't nag again until tomorrow 09:00.

bob    ▸  stop
claude ▸  ⚠️ terminating *vision-demo* (expwrk_…) — confirm with 👍 (irreversible; unsaved
          in-memory state is lost, /mnt/user_storage persists).
bob    ▸  👍
claude ▸  ✅ terminated *vision-demo*. saved ≈ $6/hr.
```

## Safety stance

- **Terminate is gated.** A parsed `stop` never auto-fires — Claude echoes the exact workspace +
  command and waits for a 👍. Wrong-workspace termination is irreversible; the confirm is cheap.
- **Post headless-safe.** In a scheduled/cron run the claude.ai Slack connector may be absent — use
  a **Slack incoming webhook** (`curl -X POST -d '{...}' $SLACK_WEBHOOK_URL`) or a bot token to
  post; use the Slack API/MCP (`users.lookupByEmail`, read history) for the @-mention + replies.
- **Poll latency is fine here.** Idle waste accrues over hours; a 5-minute sweep is plenty. This is
  poll-chat, not a real-time bot.

## Turn it into a routine

`/schedule` it every 15–30 min during work hours (`0,15,30,45 9-18 * * 1-5`). Pair it with the
[what's-running](whats-running.md) sweep for a fuller picture, and keep the node $/hr table in your
[`AGENTS.md`](../AGENTS_EXAMPLE.md) so the cost math is always current.

> Live demo needs a Slack channel + a webhook (or the Slack connector on) and will @-mention real
> people — so I won't run it until you point me at a test channel and say go.
