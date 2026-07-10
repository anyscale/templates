# Working with Claude on Anyscale — my operating manual

*Geoff Counihan · distilled from the fintech transaction-FM campaign (2026-07-06 → 07-09),
where Claude Code + Anyscale ran a multi-day GPU research campaign that beat NVIDIA's
published TabFormer fraud benchmark. This is the **how-we-worked** companion to
[`AUTORESEARCH.md`](AUTORESEARCH.md) (the **what-we-did**
research methodology).*

---

## The blurb (for Slack / the team channel)

> I ran a ~4-day ML research campaign — reproduce NVIDIA's transaction-FM benchmark, then
> beat it — where Claude Code did the overwhelming majority of the mechanical work:
> ~680 shell commands, ~470 file edits, ~25 background job monitors, 13 fresh-context
> review subagents, all against real 8×A10G Anyscale jobs. It ran **unattended overnight**
> and I resumed it across four separate sessions. The trick isn't "let the AI cook" — it's
> a small set of disciplines that make an agent's work *survivable*: durable job monitors
> that wake it on completion, adversarial subagents that can't see my notes, a written
> ledger any fresh context can resume from, and a handful of hard rules I enforce ruthlessly
> (never delete artifacts, dump configs verbatim, smoke the entry point, reproduce before
> you claim). This doc is the operating manual. Steal it.

**One sentence:** *I act as PI — I own the money, the irreversible calls, and the "is this
real?" sign-off; Claude runs the experiment loop, and everything is engineered so a dead
context (mine or its) can pick the campaign back up from disk.*

---

## 1. The operating model

Two documents, two jobs:

| doc | question it answers |
|---|---|
| `AUTORESEARCH.md` | *What* is the rigorous loop for "reproduce a published result, then beat it"? |
| this doc | *How* do I drive Claude Code + Anyscale to actually execute that loop? |

The division of labor that emerged (and that I now set up deliberately):

- **I am the PI.** I pick the benchmark, sequence the campaign (parity *before* lift), decide
  what's blog vs. diagnostic, approve every dollar of GPU spend and every irreversible action,
  and I personally read the code before I believe a win.
- **Claude is the research engineer.** It writes configs and scripts, submits and monitors
  Anyscale jobs, chains runs, does literature lookups, drafts the docs, and brainstorms +
  ranks + runs cheap experiments — largely unattended.
- **The disk is the source of truth.** Not my context window, not Claude's. Every result is a
  file; every decision is a commit or a ledger row. That's what lets any session resume.

My default handoff register is: *"ultrathink and really think about the simplest way to do
this, then go for it."* Slow the model down to reason first, then hand it the wheel.

---

## 2. The overnight machine — how jobs actually run

This is the core mechanic and the thing most people get wrong. GPU jobs take hours. I am not
going to babysit them (*"im not oging to babysit"*), and Claude shouldn't block on them either.
The loop is **event-driven off durable monitors**, not timers.

### The ritual for every experiment

```
1. edit code behind an off-by-default flag   (one experiment = one flag = one commit)
2. git add && git commit && git push          (the job pulls code from git, so push first)
3. anyscale job submit -f job_<exp>.yaml       (config-file driven; the yaml is in the repo)
4. attach a PERSISTENT Monitor on the job id   (see below)
5. …Claude does other work / I go to lunch / I go to sleep…
6. Monitor hits a terminal state → pushes a notification → Claude wakes, reads the metrics,
   and either submits the chained next job or starts the next experiment.
```

### The Monitor template

Every one of the ~25 monitors was the same shape: a poll loop over `anyscale job status`
that **echoes only on state change**, and on any terminal state dumps a **grep-filtered tail
of the logs** with exactly the numbers or failure signatures that matter:

```bash
prev=""
while true; do
  s=$(anyscale job status --id prodjob_XXXX 2>/dev/null | awk '/^state:/{print $2}')
  [ "$s" != "$prev" ] && echo "state: $s" && prev=$s
  case "$s" in
    SUCCEEDED|FAILED|*ERRORED*|TERMINATED|OUT_OF_RETRIES)
      anyscale job logs --id prodjob_XXXX 2>/dev/null \
        | grep -E "ROC-AUC|AP|HR@10|vs baseline|Traceback|out of memory|Unschedulable|Capacity" \
        | tail -30
      break;;
  esac
  sleep 90   # cadence scaled to expected job length; timeout ~1h
done
```

The grep filter is hand-tuned per experiment — it's the difference between "wake me with the
answer" and "wake me with 4,000 lines of log." **Rule: the monitor must match *every* terminal
state. Silence is not success** — a job that dies unmatched leaves the loop hanging forever.

### Chaining without babysitting

I ask for chained jobs explicitly (*"can you give me that command so its chained? so the next
begins once the first finishes?"*). Two flavors:

- **Within a run:** join pipeline stages with `&&` inside one entrypoint so any stage failure
  aborts the rest — `setsid bash -c "01 && 02 && 04 && 05 && 06 && echo SMOKE_E2E_OK"`.
- **Across runs:** a **`persistent: true` Monitor** carries the baton. Its description records
  the dependency (*"R2 probe: state + 30 fits; G1 submits on its terminal state"*); when the
  predecessor goes terminal the notification re-invokes Claude, which then submits the
  dependent job. This is how "run the analysis and launch the next job before I wake up" works.

> Note: this loop is built on **persistent Monitors + their completion notifications**, not on
> `ScheduleWakeup` timers. The equivalent of "wait N minutes" is the `sleep` *inside* the
> monitor. Timers are the wrong tool when the real signal is "job X reached a terminal state."

### Long commands on the persistent workspace

For work on the always-on workspace (not an ephemeral job cluster), SSH sessions die and take
your process with them. The pattern (also in `AUTORESEARCH.md` §5):

```bash
anyscale workspace_v2 ssh --id expwrk_XXXX -- '
  cd <repo> && git pull &&
  setsid python scripts/<x>.py ... > /mnt/user_storage/<x>.log 2>&1 < /dev/null &
  echo "launched pid $!"'
```

Then poll the **results file** for a sentinel (`SMOKE_E2E_OK`) or an error signature, not the
process. Persist partial results per stage so a late crash doesn't destroy earlier work
(*"Persist results as artifacts; never scrape logs"* — job-log streaming truncates silently).

---

## 3. Subagents as fresh-context adversaries

The single highest-leverage move in the whole campaign: **an independent agent with fresh
context found the decisive design flaw I'd missed for days, and a second vetted an idea bundle
and contributed a better idea than the one it rejected.** Independence is load-bearing.

The key realization: these are **not custom agent types**. They're plain `general-purpose`
agents wearing a **role prompt**, and their independence comes from one explicit clause:

```
CRITICAL INDEPENDENCE RULE: the repo contains markdown notes (EXPERIMENT_LOG.md, comments,
commit messages) that encode the current owner's beliefs and diagnoses. Do NOT anchor on them.
Form your own conclusions from source code + data handling + literature. If you read those
notes at all, treat every claim in them as unverified.
```

Without that clause, a review agent just hands you back your own beliefs with citations.

The three role-prompts I reuse:

- **Adversarial teardown** — *"tear apart the design; assume every decision is potentially
  wrong; you have NO loyalty to the current design."* Deliverables end with: *"if you took over
  tomorrow with a budget of three GPU runs, exactly what three runs would you do?"*
- **Leak audit** — pointed at a too-good result: *"your ONLY job is to try to destroy it by
  finding leakage / protocol violations / eval bugs. Assume it is wrong until you cannot break
  it."* Returns a one-word verdict (`BROKEN` w/ file:line / `SUSPICIOUS` / `CLEAN`) **plus the
  control experiments required before publishing** — those controls became the shuffle-label
  and velocity-baseline jobs that confirmed the win.
- **Idea vetter** — ranks a backlog by cost / impact / confound and **vetoes bundles that would
  confound the run they're meant to inform** ("is bundling G1+G2+G3 into one run acceptable, or
  is there an ordering you'd insist on?").

Two more subagent patterns:
- **`Explore` for read-only fan-out** — e.g. four parallel Explore agents, one per Ray subsystem
  (Data / Train / Core / Serve), each reviewing the template against the optimization guides and
  reporting `guide-id → file:line → severity`. Cheap, parallel, no write access.
- **Skill-in-subagent** — delegate a platform question by telling the subagent to run a skill
  itself (*"use the anyscale-platform-ask workflow… to answer precisely: how do I stop the
  autoscaler scaling GPU groups for CPU-only demand?"*). Keeps the skill's verbose output out of
  my main context.

Requirements I put in every review prompt: **cite file:line or a URL**, **verify before
reporting** (read the actual lines, don't pattern-match), and **end with a ranked, decisive
verdict** — not a file dump.

---

## 4. State that survives a dead context

The campaign spanned four sessions and multiple overnight gaps. Nothing survives in a context
window; everything survives on disk in four layers:

1. **The live task list** (`TaskCreate`/`TaskUpdate`, ~75 calls) — the working queue for the
   current session. A later session literally resumed on *"…proceed to the results rewrite per
   task #14."*
2. **The ledger** (`EXPERIMENT_LOG.md`, `TEARDOWN.md`) — claims → required evidence → status →
   cost, **including what we got wrong and why**. This is what lets a fresh context (agent or
   human) resume without re-deriving the campaign. Publication claims map 1:1 to ledger rows.
3. **Handoff docs** (`CLEANUP.md`) — a written plan the *next* session executes. I open sessions
   with *"take the steps proposed in this from the last guy… lemme know if anything doesn't make
   sense."* Claude runs these via the `superpowers:executing-plans` skill, which forces a
   review-then-execute discipline and logs any deviation (it correctly skipped an irreversible
   deletion the plan had marked "safe" and escalated it to me instead).
4. **Memory files** (`~/.claude/.../memory/`) — durable cross-session rules and campaign state.
   The feedback rules in §7 live here so I never have to re-teach them.

Commit discipline is part of this: **one experiment = one flag = one commit**, with a message
that names the experiment, so `git log` *is* the attribution trail and backtracking = flipping
a flag. *"commit each added change and make sure to write the commit message properly so we can
remember which code runs for each experiment."*

---

## 5. Skills & superpowers — an honest accounting

What I actually used (not what's theoretically available):

- **`superpowers:executing-plans`** — used once, correctly, on the `CLEANUP.md` handoff. Its
  "critically review the plan against real repo state before executing, then log deviations"
  loop is exactly right for a written handoff. This is the superpower that earned its keep.
- **Git worktrees** — used for real: an isolated checkout to pin `xgboost==3.2.0` for env parity
  without disturbing the main tree, and a clean baseline worktree for the multi-agent guide
  review. The flag discipline (every change default-off) is what makes worktree branches merge
  without conflict, so N experiments can proceed in parallel.
- **`anyscale-platform-{ask,fix,run,inspect}`** — read and *delegated* (via subagents) rather
  than slash-invoked. `anyscale-platform-ask` answered the CPU:0 autoscaler question.
- **`template` skill** — the canonical entry point for template maintenance/publish work
  (not the research campaign itself).

What I did **not** lean on but would next time: I ran the *ideas* through fresh-context review
(§3) but never invoked `superpowers:brainstorming` or `systematic-debugging` as formal skills —
I hand-rolled the equivalents. For a teammate adopting this, the formal skills are a shortcut to
the same discipline. Don't over-index on the count; index on the *pattern* (fresh-context review,
plan-then-execute, one-flag-per-experiment).

---

## 6. Anyscale plumbing that bit us (so it won't bite you)

Condensed from `AUTORESEARCH.md` §5 and the memory files — the substrate gotchas that cost real
time:

- **`workspace_v2 ssh -n <name>` resolves inside your *default* cloud/project**, not the org. If
  the workspace lives in another cloud, name lookup fails even though `list` shows it. **Fix: SSH
  by `--id expwrk_...`** (bypasses scoping), or pass `--cloud`. Port-forward TensorBoard from the
  workspace with `--id ... -- -L 6006:localhost:6006`, and run the tunnel on your laptop.
- **GPU worker groups: advertise `resources: {CPU: 0}`** so CPU-heavy stages can't scale them up —
  but keep ≥1 CPU-capable group (the head is CPU:0 too, and data-plane tasks need CPU:1 somewhere).
- **The autoscaler scales on pending logical resource requests, not utilization.**
- **Throughput knobs are model changes.** A batch-size bump that halves optimizer steps silently
  degrades everything in update-count-bound regimes. Ablate infra changes like architecture ones.
- **Checkpoint paths must be absolute/URI**; per-epoch checkpoints + `FailureConfig` make spot
  instances safe.
- **`iter_torch_batches(dtypes=...)` must cover every column or be `None`** (a partial dict
  KeyErrors mid-train-loop).
- **`RunConfig.name` reuse auto-resumes** the latest checkpoint — a footgun that we *weaponized*
  into a half-price "warm-restart continuation" for is-it-undertrained diagnosis, but that
  silently trains 0 epochs if you didn't mean it. Unique run names per invocation otherwise.
- **macOS local runs segfault at the XGBoost stage** (torch+xgboost libomp clash in the Ray
  driver) — run that stage in a separate process locally; the cluster is unaffected.

---

## 7. The rules I enforce on Claude

These are the corrections I gave often enough that they became standing rules (now in memory).
Every one was paid for. The blunt ones are quoted faithfully because bluntness is what made them
stick:

1. **Never delete an artifact — move it aside** (`<name>_old_<stamp>`, one stamp per rerun, for
   *every* artifact kind: model, embeddings, tokenized, downstream). Moving on one filesystem is
   free; deleting cost me a $4/50-min GPU re-extraction. → *"WTF WHY DIDN'T YOU MOVE THE EMBS?
   ALWAYS FUCKING MOVE THE EMBS. JUST MOVE THE ENTIRE DIR PLEASE NEXT TIME."*
2. **Dump configs/outputs verbatim.** Whole config object, original key names, zero filtering,
   zero "friendly" renames. Renamed keys break grep between output and code; hand-picked fields
   drift from reality. → *"just put the entire fucking yamls into the dump. stop filtering
   anything. thats the simplest thing."*
3. **Smoke the entry point, not the unit.** Run the *actual* script end-to-end on tiny data
   before any paid run — failures live in the glue (writers, health checks, eval readers), not
   the tested core. → *"did we try running this e2e before running a massive job on it? like for
   1 epoch on a sample?"*
4. **Reproduce before you claim; be extra skeptical of your own win.** Reproduce the reference's
   number through *your* harness before touching a model, and prove a saved run used the exact
   input pipeline before trusting its numbers. → *"im extra skeptical… ultrathink about why I
   would be right or wrong to do this."*
5. **No speculative complexity.** If the pipeline should use the same data as the repro, use the
   same script — don't invent a second code path. Simplicity first; strip falsified flags
   entirely on the clean branch rather than leaving dead knobs.
6. **A number on few positives is a rumor.** Bootstrap CIs on every reported number; the count of
   *positives* drives the eval plan; paired resampling for ordering claims. Never blog a point
   estimate.
7. **ultrathink, then go.** Reason about the simplest correct approach first; then act
   autonomously. The think-first gate is where over-engineering gets caught before it's committed.

---

## 8. What I delegate, reserve, and expect escalated

| | |
|---|---|
| **Delegate fully to Claude** | configs & scripts; job submit/monitor/chain; TensorBoard & hyperparameter logging; literature lookups; drafting docs; brainstorming → ranking → running *cheap* experiments |
| **Reserve for myself** | benchmark choice & campaign sequencing; go/no-go on GPU spend ("8×A10G for how long?"); anything irreversible; the "is this real?" sign-off (I read the code myself); what ships vs. stays diagnostic; the local git half ("commit it and ill pull") |
| **Expect Claude to escalate** (and it did, via `AskUserQuestion`) | money/compute commitments; anything irreversible (it refused to delete cluster storage a plan marked "safe"); protocol decisions that must stay pinned (eval-set size, where shared artifacts live); "launch the next scale, or stop and show you results first?" |

The escalation boundary is not arbitrary: it's exactly **money, irreversibility, and results
that need my eyes** — which is the same boundary the rules in §7 protect.

---

## 9. Adopt this in an afternoon

A teammate starting a reproduce-then-beat campaign:

```
[ ] Write the ledger stub first (claims → evidence → status → cost) — even empty.
[ ] Reproduce the reference number through YOUR pipeline before any model work.
[ ] One experiment = one off-by-default flag = one commit with an experiment-named message.
[ ] Smoke the real entry point on tiny data before every paid run.
[ ] Submit jobs from committed+pushed code via `anyscale job submit -f <yaml>`.
[ ] Attach a persistent Monitor per job: echo on state-change, grep the metrics/failures on
    terminal state, match EVERY terminal state.
[ ] Chain dependent jobs off the predecessor's completion notification, not off hope.
[ ] Before trusting a design, dispatch a fresh-context review agent with the independence
    clause ("do NOT anchor on my notes; verify from code; assume it's wrong until you can't
    break it"). Require file:line/URL citations and a decisive verdict.
[ ] Never delete artifacts — move aside. Dump configs verbatim. Bootstrap-CI every number.
[ ] Leave a handoff doc so the next session (yours or a fresh agent's) resumes from disk.
```

*Fuller research rigor lives in `AUTORESEARCH.md`; the worked example and its war stories live
in `TEARDOWN.md` / `EXPERIMENT_LOG.md` / `BLOG_NOTES.md` on the `geoff/fm_recs_and_fraud`
branch.*
