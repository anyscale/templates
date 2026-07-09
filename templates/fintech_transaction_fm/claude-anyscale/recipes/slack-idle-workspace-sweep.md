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
# node $/hr lookup lives in AGENTS.md (or a small table); used to turn idle-hours into $ wasted
```

## The logic

```
1. clouds ← anyscale cloud list        # id → region
   workspaces ← anyscale workspace_v2 list   # name, id, STATE, cloud, CREATED BY (email), CREATED AT
   keep those whose cloud.region == REGION and STATE == RUNNING with worker nodes up
   (skip scaled-to-zero: no nodes → nothing to waste)

2. for each candidate workspace:
     util%      ← <YOUR METRICS SOURCE>          # ← the one adapt-point, see note below
     if util% >= MIN_UTIL_PCT: skip
     idle_min   ← how long util has been under threshold   (≥ IDLE_GRACE_MIN to flag)
     node_type  ← anyscale workspace_v2 status / its compute config
     $wasted    ← (idle_min/60) * $per_hr(node_type)
     owner_id   ← slack lookup by email(CREATED BY)         # email → <@U…> for the @-mention

3. post to SLACK_CHANNEL: @owner + workspace + util% + idle time + $wasted + console link
                          + reply options (keep / stop / ignore)

4. (poll-chat) read the thread each cycle: on `keep` → snooze 24h; on `stop` → confirm, then
   anyscale workspace_v2 terminate; on `ignore` → drop it for this sweep.
```

**The one adapt-point — where util% comes from.** It is *not* in `workspace_v2 list`. Feed it from
whatever you already have:
- SSH the workspace and read `ray status` / the Ray dashboard's GPU metrics, or
- query your Prometheus/Grafana GPU-utilization series for that cluster.

Everything else (owner, uptime, node type, cost math, the @-mention, the Slack round-trip) is
straight off the CLI + Slack. Swap in your util call and the recipe is complete.

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
