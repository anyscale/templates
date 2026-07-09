# Recipe: "What's running, and what's it costing me?" ✅

**The ask:** *"Do a sweep — what jobs, services, and workspaces do I have running right now, and
what should I be worried about cost-wise?"*

**What Claude does:** reads your inventory across all three workload types (read-only, costs
nothing), then flags the things that quietly cost money: **running services** (always-on spend),
**live job clusters**, and **long-idle workspaces**. Workspaces themselves are cheap when idle
(they scale to zero) — the ones that matter are services and any cluster that's actually up.

**Skills/tools:** plain `anyscale` CLI. No skill required. Good candidate to allow-list so it
never prompts.

---

## Run it

```bash
# Services — these are almost always live spend (an endpoint sitting up 24/7)
anyscale service list 2>/dev/null | grep -v -i "warning\|listing\|per-page\|max-items\|view your\|mode "

# Recent jobs — look for anything still RUNNING/STARTING (a live cluster) and any failures
anyscale job list 2>/dev/null | grep -v -i "warning\|listing"

# Workspaces across all clouds — mostly cheap when idle; note who owns what
anyscale workspace_v2 list 2>/dev/null | grep -v -i "warning\|listing\|per-page\|max-items\|view your\|• "
```

Then just ask Claude: *"anything here I should shut down?"*

## Real output (from my org, sanitized)

**Services — two live right now, both real spend:**

```
NAME                              ID                CURRENT STATE
pubmatic-recommendation-serving   service2_qhy3…    RUNNING      ← live endpoint
deploy-gpt-oss-120b               service2_cw3s…    RUNNING      ← 120B model, live endpoint
```

**Jobs — my campaign history; note the one that didn't finish:**

```
NAME                          ID                CURRENT STATE
fintech-fm-paired-bootstrap   prodjob_62ql…     SUCCESS
fintech-fm-restore-r2         prodjob_6ha9…     SUCCESS
fintech-fm-next-merchant-v3   prodjob_rp4b…     SUCCESS
fintech-fm-fulltest           prodjob_1l5i…     OUT_OF_RETRIES   ← worth a look (see night-watch)
```

**Workspaces — the cost lesson is here:**

```
NAME                                    STATE     CREATED
distributing-pytorch-…-gray-zoo         RUNNING   2026-07-09
object-detection-video-processing-…     RUNNING   2026-07-08   (mine)
ray-demo-v2                             RUNNING   2026-06-26
zg-tfm                                  RUNNING   2026-06-15
optimization-agent-dev                  RUNNING   2026-03-11   ← "running" since MARCH, and that's fine
```

The read I take from this: the **services** are what to scrutinize (an idle endpoint still bills);
the **workspaces** running since March are *not* a problem because they idle-scale to zero — which
is exactly why keeping one warm is free
([always-on-workspace](always-on-workspace.md)).

## Turn it into an automation

- **Cost watchdog on a schedule:** run this every morning via a scheduled cloud agent (`/schedule`)
  and have Claude post a one-line "N services up, M job clusters live, nothing unexpected" to Slack —
  or flag anything new.
- **Teardown helper:** *"list every service and job cluster that's up, tell me the last time each
  was used, and draft the terminate commands for the ones idle > 7 days"* (Claude drafts; you
  approve — terminating is irreversible, so it stays gated).
