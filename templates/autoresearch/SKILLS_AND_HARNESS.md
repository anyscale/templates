# Skills or harness? What campaign 2 says

_Written 2026-10-16, after the pathology zoom-agent PoC (repo `vlm-sft-rl`, Sep 24 – Oct 9, 2026)._

That PoC ran SFT and GRPO on Qwen3-VL with TRL and SkyRL: about 118 Anyscale jobs and ~670 GPU-hours, all launched by Claude Code from a CPU-only workspace. It's the "campaign #2" that `REQUIREMENTS.md` → *Guard against over-building from n=1* asks for: the second campaign is supposed to rewrite this spec. This doc records three things:

- what happened with the shipped Anyscale skills during it
- what the platform already offers as guardrails
- where the evidence lands on the open question in `research_director.md` §13: *does this ship as an Anyscale Agent Skill or as a heavier standalone product?*

## 1. Short answer

**Both, layered, and the layers do different jobs.**

- **Anything that must happen goes in the harness**: budget caps, the ledger, terminal-state reconciliation, known-error checks. Code and hooks run on every tool call, whatever the model decides. Skill text only runs if the model chooses to load it. In campaign 2, the model chose to load an Anyscale skill **0 times in 14 sessions**, even with all 14 listed in every session (§2). That's the strongest evidence yet for the harness's own principle that *rules become physics*.
- **A skill is the front door and the playbook**: when to use the loop, how to frame a campaign, how to read a failure, which harness script to call. It's how people and agents find the harness. It doesn't enforce anything.
- **Ship it as a plugin: skill + hooks + harness scripts.** `anyscale-skills` already ships this way (`hooks/hooks.json` next to the skills). That answers §13: use the skill delivery channel, but put the guarantees in hooks and code.
- **Reuse the Anyscale debug skills as building blocks, not as the loop.** `anyscale-platform-inspect` (read-only, never asks the user anything) and its errors cache are worth calling. `anyscale-platform-fix` is the wrong loop for experiments (§2.3).

## 2. The Anyscale skills during campaign 2

### 2.1 What happened

| | |
|---|---|
| Anyscale jobs | ~118 (57 succeeded, 27 `OUT_OF_RETRIES`, 34 terminated, mostly on purpose) |
| Failures read from a job log and fixed | 31, of which 20 were on Oct 6–8. 8 repeated an earlier one |
| Anyscale skills listed in each session | 14 |
| Anyscale skill invocations | **0** (the only skill ever called was `dataviz`, twice) |
| Stored reasoning blocks in the main debug session that mention an Anyscale skill | 0 of 225 |

Stale skills weren't the cause. The workspace copy of `anyscale-platform-fix` is byte-identical to `anyscale-skills` main, and all 14 skills have identical frontmatter. The 31-failure table is in `LZ/autoresearch-story/raw_notes/08_anyscale_diff_detailed.md` Part 9.

### 2.2 Why they never fired

1. **The trigger is phrased as a hand-off.** `/fix` says "use when *given* a directory, job ID, … or error message to debug". In an experiment loop the agent submits the job and finds the failure itself. Failure text reached it 311 times in its own command output and 45 times from watchers it had armed, and never as a user saying "debug this job".
2. **`inspect`'s description has nothing to match**: "Inspect workloads." Agents match on the words they're looking at, such as STARTING, OUT_OF_RETRIES, OOM or quota.
3. **The loop was fast and already worked.** In a typical case, a watcher reports `job: FAILED` at 18:05:55. By 18:06:42 the agent has grepped the log, changed one value with the reason in a comment, and resubmitted. There's no gap where a skill gets considered.
4. **The routing rules don't ship.** Inside `anyscale-debug-agent`, CLAUDE.md says "IMMEDIATELY invoke `/fix`". Customers get only a security gate. That repo's evals start with an explicit `/fix`, so nothing measures whether the skill gets auto-selected.

**Lesson for our own skill:** write the description in the words the agent will be looking at (job states, "experiment", "sweep", "beat the baseline"). Name the skill or script explicitly in any loop or cron prompt. Don't rely on the model choosing to load it.

### 2.3 Would they have helped? About an hour, and only if used selectively

Claude's own diagnosis took a minute or two. The real costs were failed jobs: 27 failed, with a median of 24 minutes from submit to failure and ~23 overlapping wall-clock hours in total.

| | Estimate |
|---|---|
| Quota vs. capacity check on the job stuck in STARTING (Sep 29) | ~20 min saved, plus no lasting footnote on a benchmark |
| Job events for the 4 jobs that never got a cluster (Oct 6) | ~15–30 min saved |
| Errors cache as shipped | 0 (none of the 31 were in it) |
| A cache written after each fix and checked before each submit | ~2 h of avoided failed jobs |
| `/fix` stop-and-ask steps while the user is awake | ~3 questions per failure, ~90 in total |
| `/fix` stop-and-ask steps overnight | At least one 8–9 h stall: a spot-preemption resume 7 min after the user went to sleep (Oct 7) |
| `/fix` rule "always reproduce in a workspace first" | Impossible on a CPU-only workspace. A local SkyRL smoke run killed the node. A GPU workspace on Oct 6 would have hit the same AWS capacity wall |
| `/fix` static check of the whole directory on each failure | Minutes each, times 31, for mostly one-number fixes |

**Used as written, `/fix` would have cost time. Used selectively (inspect, the quota and event lookups, the cache), it would have saved about an hour.** The CLI, the docs and the agent reading framework source were enough for all 31 failures. The hard fixes (SkyRL's eager image loading, NCCL across mixed nodes, vLLM memory headroom) came from logs and library source, not Anyscale-specific knowledge.

`/fix` was built for a person handing over one broken workload, a workspace that can reproduce it, and someone around to answer questions. An experiment loop has none of those. It does have at least one unattended caller already: this repo's `AGENTS.md` has the `template-updater` Cursor agent drive `/anyscale-platform-fix` through template CI with nobody watching. That argues for an explicit unattended switch in the skill, rather than each caller working around the stop-and-ask steps. The escalation boundary in `BUDGET_POLICY.md` (ask only about money, irreversibility and "is this real?") is the right one, and campaign 2 confirms it.

## 3. What the shipped skills don't cover

The shipped skills (`anyscale-platform-{fix,inspect,run}` and `anyscale-workload-llm-post-training`) contain no early-stop guidance at all, and nothing on job watchers or spend caps. What campaign 2 actually ran on:

| Need | Anyscale skills | This harness | Campaign 2 evidence |
|---|---|---|---|
| Watch every job; echo on state change; grep-filtered log on any terminal state | none | Monitor template (`CLAUDE_WITH_ANYSCALE.md` §2), R6, `reconcile.py` | 171 watchers, 662 notifications |
| Kill losing runs early | none | R3 canaries (in-process), ASHA in the director's MONITOR phase | Free-zoom GRPO stopped after 0% on every eval; several runs killed for cost or because they were superseded |
| Smoke before scale | post-training skill: "run a 1–2 step smoke test" | smoke rung, non-functional #4 | Smoke passes ran *inside the job*, since the workspace couldn't run them |
| Checkpoint and resume | partly (post-training templates) | `BUDGET_POLICY.md` defaults | Checkpoint every 10 steps, `resume_mode latest`, `max_retries 2` after a run died at step 10 with nothing saved |
| Spend caps | none | `budget.py`, `hooks.py` | None in the PoC. GPU-hours were reconstructed after the fact (544 on the morning of Oct 8, ~670 at the end) |
| Unattended operation | none: `/fix` stops to ask | escalation boundary in `BUDGET_POLICY.md` | `overnight.sh` plus self-written cron prompts (§5) |
| Known-error memory | errors cache: static, hand-curated, 7 entries, last updated Aug 11 | none | 8 of 31 failures were repeats |
| Several agents in one repo | none | none | Two Claude sessions shared the repo. One handed over to the other by message, and commits had to name their own files |

## 4. Guardrails the platform already has

None of the skills tell an agent to use these as guardrails. `inspect` reads quotas only when diagnosing a stuck job, and budgets appear only in the raw API spec it bundles (`references/openapi.json`). Everything below comes from that spec and the Anyscale SDK; none of it has been tested here.

| Primitive | What it does | How the harness should use it |
|---|---|---|
| **Resource quotas** (`/api/v2/resource_quotas/`) | Per cloud, project or user, caps the number of CPUs, GPUs, instances or accelerators of a type. A hard quota stops new nodes from being added; a soft one only notifies. | Give each campaign its own project with a hard GPU-count quota. This is the hard layer `hooks.py` says it lacks: SDK or ssh-wrapped submits slip past the hook but not past the quota. Creating quotas probably needs org admin rights. |
| **Instance usage budgets** (`/api/v2/instance_usage_budgets/`) | `budget_amount` in `DOLLARS` or `ANYSCALE_CREDITS`, `DAILY` or `MONTHLY`, per cloud or project, with a notification channel. The schema has no enforcement field, so it looks like an alert, not a stop. | A daily dollar alert per campaign project, as a check on `budget.py`'s own ledger. |
| **Usage reporting** (`/api/v2/aggregated_instance_usage/` by project, user or instance type) | Actual usage from the platform | Reconcile the registry's estimated cost against the platform's number. The console stays the invoice of record (`harness/README.md`). |
| **Job `timeout_s`, `max_retries`** | Hard wall-clock kill; automatic retries | `launcher.py` already bakes in the timeout (R4's runtime half). Retries plus checkpoints make spot preemption cheap. |
| **Job queues** (`job_queue_config`, `max_concurrency`) | Caps how many of a campaign's jobs run at once | A concurrency cap per campaign. Note that the queue's default is 1, with every driver on one shared head node. |
| **Compute config** (`max_nodes`, `min_nodes: 0`, idle termination, `market_type`) | Fleet shape | Already in `BUDGET_POLICY.md` defaults. See §5 for the spot amendment. |

## 5. Campaign 2 against the spec

The labels mean: **held** (campaign 2 confirms it), **amend** (change the text), **new** (not in the spec).

- **R1 registry (held).** There was no JSONL. The ledger was README tables, `collect_results.py` and job names, and cost was reconstructed afterwards. A fresh session resumed from a memory note and READMEs, which worked but took re-reading.
- **R2 launcher (held, plus one ask).** There were 45 hand-written job YAMLs, against ~15 in the fintech campaign. The PoC's best habit was writing *why* a value changed as a comment next to it (`gpu_memory_utilization: 0.7  # 0.85 leaves ~0.9 GB; the weight push needs 1 GB`). The generated spec should carry a `why` field per non-default knob, so the log of changes survives generation.
- **R3 canaries (amend).** In-process is right for NaN, leaks and collapse. But most of the PoC's early stops were judgement calls on eval curves, read by watchers from files the run wrote to shared storage. That isn't the log-stream scraping R3 warns about. Amend "not an external sniper" to "not a log-stream sniper": watchers that read run-written artifacts on shared storage are fine.
- **R4 budget (held, plus a layer).** No cap existed, and 34 jobs were terminated by hand, partly for cost. Add the platform layer from §4 underneath `hooks.py`.
- **R6 monitors (held, at 7× the scale).** 171 watchers, against ~25 in the fintech campaign. Two additions:
  - Watchers expire, so re-arming them is part of the overnight prompt.
  - A re-armed watcher re-read a previous run's stale log lines from shared storage and reported old errors as new. Key watcher output on run id plus timestamp.
- **R7 eval pin (amend: the pin covers more than the data).**
  - A prompt bug invalidated a set of runs: the tool instructions showed a concrete target ("3"), and every model copied it. Prompts belong in the pin.
  - Two vLLM versions disagreed by about 5 points on the same untrained agent (19.7% vs 14.7%). The serving engine and its version belong in the pin.
- **R8 artifacts (held).** Keeping only the last 2 checkpoints deleted the best ones overnight. The fix was keeping 10 plus a copy loop. Checkpoint retention is an artifact-safety default, not a training knob.
- **`BUDGET_POLICY.md` spot default (amend).** "Spot for stateless GPU workers; on-demand only for the head/learner" misses two things:
  - RL inference engines aren't stateless. SkyRL can't lose one mid-step.
  - `PREFER_SPOT` *evicts on-demand fallback nodes* once spot comes back. An engine node was swapped about 80 min in, and the run died at step 10 with nothing saved.

  Engines go on-demand, or the run checkpoints often enough to lose an engine cheaply.
- **Feasibility (new: capacity).** `feasibility.py` checks cost and statistical power, not whether the instance type can be had today. On Oct 6 there was no 4-GPU G capacity in any zone all day (about 90 EC2 attempts). The fix was single-GPU nodes, and that shape then caused an NCCL failure across mixed instance types. The launcher needs a fallback ladder of compute shapes, and each step down the ladder has a cost the spec should name.
- **R10 failure triage with write-back (new).** A short loop when a job fails mid-campaign:
  1. Classify by job state. No run means infrastructure: check events, quota and capacity.
  2. Check known errors.
  3. Make the smallest fix, with the reason in a comment.
  4. Resubmit with a smoke pass.
  5. Write the entry.

  Entries go to `$AUTORESEARCH_BASE/errors-cache/`, which lives on shared storage and survives skill syncs. Good ones go upstream to the Anyscale errors cache.

  The pre-submit half is where the time savings are. `hooks.py pre` already parses every job YAML, so it can also lint it against known-bad patterns, such as vLLM memory ≥ 0.85 on a 24 GB card, `PREFER_SPOT` on engine nodes, or SkyRL without `SKYRL_RAY_PG_TIMEOUT_IN_S`. That is the "static check" `/fix` runs after a failure, moved to before the submit, where it can prevent one.
- **R11 multi-agent coordination (new).** Two sessions, one repo, one shared data root. Minimum rules:
  - Each session commits only its own files (`git commit -- <files>`).
  - Hand-overs are written messages.
  - The registry's single-writer rule extends to "one session owns a run".
- **The sequencer exists in miniature (evidence for `research_director.md` §3).** On the night of Oct 7 the agent wrote `skyrl-sft-projector/overnight.sh`, a deterministic loop with idempotent state files. When each GRPO run reached its last step, it exported the adapter and submitted evals, without a person. Its rules:
  - resubmit a failed run once, resuming from checkpoint
  - continue the run to step 200 if step 100 was within a point of its best
  - leave the NCCL-failed SFT "to a person"

  The night's results were all on disk in the morning. That's the outer state machine with gated interiors, built ad hoc in one evening.

## 6. Next steps, smallest first

1. **Pre-submit lint in `hooks.py`.** Check the parsed job YAML against a small list of known-bad patterns from §5 (R10). This extends an existing hook and is cheap.
2. **Errors-cache write-back.** Use the entry template from `anyscale-platform-inspect/references/errors-cache/cache-index.md`, so entries can go upstream unchanged.
3. **Platform guardrails per campaign.** A dedicated project, a hard GPU quota, a daily dollar budget alert, and `timeout_s` on every job.
4. **Amend the spec text**: the R3 wording, the R7 pin contents, and the `BUDGET_POLICY.md` spot rule.
5. **Package it as a plugin**: an autoresearch skill (§2.2 lesson: describe it in the words the agent sees) plus `hooks.py` plus the harness. Call `anyscale-platform-inspect` for diagnosis. Don't route through `/fix`.

## 7. Related work outside this repo

- **Upstream proposals to the Anyscale debug skills.** Two stacked local branches in `anyscale/anyscale-debug-agent`, neither pushed:
  - `geoff/errors-cache-and-inspect-trigger`: 7 errors-cache entries from campaign 2 and a sharper `inspect` description. PR text in `LZ/anyscale-debug-agent_PR_errors_cache.md`.
  - `geoff/fix-unattended-loop`: a `/fix` description plus a quick loop for unattended runs, for discussion with the skill owners. Note in `LZ/anyscale-debug-agent_PR_fix_unattended.md`.
- **Shareable write-up:** https://claude.ai/artifact/CMjGcEBuJdhUD58YLWKiDY
- **Evidence:**
  - Session transcripts under `~/.claude/projects/-home-ray-LZ/`
  - `LZ/autoresearch-story/` (evidence pack, the 31-failure table, the platform timeline)
  - Job records in `LZ/autoresearch-story/scripts/anyscale_jobs_geoff.json`
  - The PoC repo `vlm-sft-rl` (job YAML comments carry each fix and its reason)
