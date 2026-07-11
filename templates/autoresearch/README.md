# Autoresearch — a benchmark-anchored ML hill-climbing program on Anyscale

A program for running many **reproduce-then-beat** ML research campaigns on Anyscale,
each one: `git clone` a published research repo → reproduce its number in a Rayified
pipeline → iterate to push past it. Distilled from the fintech transaction-FM campaign
(2026-07-06 → 07-11), which reproduced NVIDIA's TabFormer fraud benchmark and beat its
published fusion headline (AP 0.1755) embedding-alone (AP 0.23–0.27). This directory
turns that one hand-driven campaign into a repeatable, budgeted, multi-vertical program.

> **Status: DESIGN / SEED PLANNING.** No campaign here has been launched. This is the
> first push — requirements, a budget policy, and pre-registered seed plans for 9
> candidate campaigns. The harness itself (`REQUIREMENTS.md`) is not built yet; we
> iterate on it as we build. Nothing in `campaigns/` should be run until its plan is
> reviewed and a budget envelope is approved.

## Where this lives (intent)

This directory is an **internal research program incubating inside the templates repo** — it is
deliberately *not* a console template: no `BUILD.yaml` entry, no build/publish/CI. It sits next
to `fintech_transaction_fm/` because that's the campaign it's distilled from and it inherits the
repo's conventions. If it proves out, "autoresearch as a console template" (or its own repo) is a
plausible product; until then, treat it as lab notes, not a shipping artifact. Because the disk
is the source of truth, **paths must not bake in one person's laptop layout** — reference repos
clone under a configurable `$AUTORESEARCH_REFS` root (default `~/anyscale/`), not a hardcoded home
dir, and all durable artifacts live under `$BASE` on cluster storage.

## What this is (and what already exists)

The **methodology** is already written and battle-tested — do not reinvent it:

| Doc | What it gives you | Where |
|---|---|---|
| `AUTORESEARCH.md` | The rigorous loop: 10 Iron Rules, the DIAGNOSE→PROPOSE→REVIEW→RUN→SCORE state machine, the per-domain fidelity/proxy-axis table, the §6 tooling roadmap | `../fintech_transaction_fm/claude-anyscale/` |
| `CLAUDE_WITH_ANYSCALE.md` | The PI operating manual: overnight monitors, fresh-context adversarial subagents, disk-as-source-of-truth, the escalation boundary | same dir |
| `AGENTS_EXAMPLE.md` | The `AGENTS.md` starter every campaign copies to its repo root | same dir |

This directory adds the **program layer** on top:

| Doc | Question it answers |
|---|---|
| `REQUIREMENTS.md` | What must the autoresearch **harness** do? (the §6 roadmap, made buildable) |
| `BUDGET_POLICY.md` | How much money/GPU does each campaign get, who approves spend, how is it tracked? (formalizes what the PI did by hand) |
| `SEED_INDEX.md` | The ranked menu of candidate campaigns, by vertical, cost, and confidence |
| `campaigns/*.md` | One pre-registered seed plan per candidate run |
| `campaigns/_TEMPLATE.md` | The seed-plan schema — copy it to start a new campaign |

## The operating model (unchanged from the FM campaign)

- **The PI owns** the benchmark choice, campaign sequencing, every dollar of GPU spend,
  every irreversible action, and the "is this real?" sign-off.
- **The agent (Claude Code) is the research engineer:** writes configs/scripts, submits
  and monitors Anyscale jobs, chains runs, runs the fresh-context review agents, drafts
  docs — largely unattended, event-driven off durable job monitors.
- **The disk is the source of truth** — not any context window. Every result is a file
  on `/mnt/user_storage`; every decision is a commit or a registry row. That is what
  lets any session (human or agent) resume a campaign it didn't start.

## The one thing every campaign must do first

**Reproduce the reference's published number through YOUR Ray pipeline before touching a
model.** Two gates, in order (Iron Rule #1):
1. **Artifact gate** — run the reference's *shipped* eval on its *shipped* checkpoint,
   match its reported number. Proves you hold their protocol.
2. **Pipeline gate** — produce that same number through your Rayified harness. Proves
   your data/eval stack is theirs.

No number, no claim. A campaign that hasn't cleared both gates is not allowed to spend
budget on "beat it" experiments.

## How to start a new campaign

1. Copy `campaigns/_TEMPLATE.md` to `campaigns/NN-<slug>.md` and fill every field. The seed
   plan is a **pre-registration**: once the first paid run happens, the plan is frozen —
   changes land as new commits marked `AMENDED`, never silent rewrites, and every registry
   row records the seed-plan commit SHA it ran under (so you can't retroactively "predict"
   the result).
2. Get the budget envelope approved per `BUDGET_POLICY.md` (the PI signs off the total
   and the smoke→proxy→full promotion gates).
3. `git clone` the reference repo under `$AUTORESEARCH_REFS` (default `~/anyscale/`, where
   `generative-recommenders` and `transaction-foundation-model` already sit).
4. Copy `AGENTS_EXAMPLE.md` → the campaign's `AGENTS.md`; copy the nearest
   `optimization-guides/workloads/<archetype>/{main.py,cluster_config.yaml}` as the
   Rayification skeleton.
5. Run the loop in `AUTORESEARCH.md`. Log every run to the results registry
   (`REQUIREMENTS.md` R1) — no state in a context window.

## Design assumptions baked into these docs

Stated explicitly so they're easy to challenge as we iterate:

1. **GPU-hours are the budget currency, not dollars.** Dollar figures drift with cloud
   pricing and spot; GPU-hours are stable and convert via one table. See `BUDGET_POLICY.md`.
2. **Reference numbers come from the repo's eval code, not its paper/blog.** Seed plans
   name the published number as a *target to confirm from code*, never as ground truth
   (the FM campaign's reference contradicted its own blog).
3. **Campaigns are pre-registered.** The hypotheses, decision metric, eval pin, and
   budget are written down *before* the first paid run — the seed plan is the pre-registration.
4. **Cost estimates in seed plans are pre-calibration guesses.** They exist to size the
   envelope; the proxy-calibration step (`AUTORESEARCH.md` §3) tightens them before any
   full run. Treat them as ±50% (the guides' own guardrail on baseline estimates).
5. **The program spans Anyscale's 5 service verticals plus adjacent good-fit domains.**
   Vertical coverage, not just benchmark prestige, drives which campaigns ship first.
