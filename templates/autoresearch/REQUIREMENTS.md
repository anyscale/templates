# Autoresearch harness — requirements

What the autoresearch **harness** must do to turn the hand-driven fintech-FM campaign into
a repeatable product. The methodology is fixed (`AUTORESEARCH.md`); this is the software
that mechanizes it. Everything below traces to something a human did by hand in the FM
campaign — the citations point at the manual version we are replacing.

Priorities: **R1–R5 are the `AUTORESEARCH.md` §6 tooling roadmap**, in its stated order.
R6–R9 are cross-cutting requirements the campaign teardown surfaced. Nothing here is built
yet; this is the spec we iterate against.

## Vocabulary

- **campaign** — one reproduce-then-beat effort against one reference (a seed plan in `campaigns/`).
- **experiment** — one config delta within a campaign; one flag, one commit (Iron Rule #8).
- **run** — one execution of an experiment at one rung (smoke / proxy / full).
- **rung** — fidelity level: `smoke` (code runs), `proxy` (rank ideas), `full` (publishable).
- **decision metric** — the one number claims are judged on; everything else is a guard metric.

---

## R1 — Results registry (highest priority)

**Problem it solves:** in the FM campaign "that state lived in one agent's context window —
exactly what a fresh context can't resume" (`AUTORESEARCH.md` §6.1). The ledger was prose
(`EXPERIMENT_LOG.md`) maintained by hand; promote/kill decisions were not machine-readable.

**Requirement.** Every run appends exactly one row to an append-only JSONL registry:

```jsonc
{
  "campaign": "01-hstu-recsys",
  "run_id": "prodjob_...",              // the Anyscale job id, the durable handle
  "commit": "a1b2c3d",                   // git SHA the job ran from
  "flags": {"readout": "mlp", "seq_len": 1024},   // the experiment delta, verbatim
  "rung": "proxy",
  "seeds": [0, 1, 2],
  "decision_metric": {"name": "ndcg@10", "value": 0.211, "ci": [0.204, 0.218]},
  "guard_metrics": {"hr@10": 0.355, "train_loss": 1.83},
  "cost": {"gpu_hours": 2.4, "gpu_type": "A10G", "usd_est": 2.4, "spot": true},
  "wall_clock_s": 5820,
  "eval_pin": "sha256:...",              // hash of the pinned eval artifact this scored on
  "seed_plan_commit": "e4f5a6b",         // the seed-plan SHA this run was pre-registered under
  "status": "SUCCEEDED",
  "decision": "promote",                 // promote | kill | hold — machine-readable, NOT prose
  "notes": "beats baseline CI-separated at proxy"
}
```

- **Decision is a typed field, not prose.** `decision ∈ {promote, kill, hold}` is what
  promote/kill logic reads; `notes` is human colour only. A killed run must record *why* in
  a structured way (add `decision_reason` if free-text is unavoidable) — the "fresh agent
  resumes from disk alone" promise breaks hardest exactly on *why things were killed*.
- **Single writer, and it survives a hard crash.** The run streams a `RUNNING` heartbeat row
  (with partial metrics) as it goes; the **monitor (R6) writes the one terminal row** on any
  terminal state. A run that dies to spot preemption / OOM / node loss cannot file its own
  `FAILED` row — dead processes don't write death certificates. Writes are idempotent keyed on
  `(run_id, status)` so a double-fire is a no-op. R4's envelope math (`sum(gpu_hours)`) depends
  on there being exactly one terminal row per run — never scraped from logs (log streaming
  truncates silently, `AUTORESEARCH.md` §9.11).
- **Every number carries a bootstrap CI** where the metric has few positives; the count of
  *positives* (not rows) drives whether a point estimate is even allowed (`AUTORESEARCH.md`
  §9.1 — the campaign drafted two false narratives from point estimates before CIs existed).
- `eval_pin` ties every row to the exact frozen eval artifact, so rows are only comparable
  within one pin. Cross-pin comparison is a bug the registry must make detectable.
- Promote/kill (`AUTORESEARCH.md` §3) reads the registry programmatically, not prose.

**Done when:** a fresh agent context can reconstruct a campaign's full state — what ran,
what it cost, what won, what was killed and why — from the JSONL alone.

## R2 — Experiment launcher

**Problem it solves:** "Hand-written job specs hosted two of our infra bugs"
(`AUTORESEARCH.md` §6.2). The FM campaign had ~15 hand-written `job_*.yaml` files.

**Requirement.** An `experiment.yaml` (base config + flag deltas + rung + seeds) →
generated Anyscale job spec → `anyscale job submit` → auto-score against the pinned eval →
append to R1. One command per experiment; the human never hand-edits a job YAML.

- The generated job spec encodes the rung's compute (per `BUDGET_POLICY.md`) — smoke on
  CPU/1-GPU, proxy at reduced fidelity, full at the campaign's hero shape.
- Reuses the FM campaign's proven job structure: `working_dir` from committed+pushed git;
  `env_vars`; per-stage `resources: {CPU: 0}` fences on GPU groups; `min_nodes: 0`
  scale-to-zero; artifact `mv`-aside baked into the entrypoint (`AGENTS_EXAMPLE.md`).
- Encodes key knobs in the run name so `git log` and TensorBoard are greppable (Iron Rule #5).

## R3 — In-training canary aborts

**Problem it solves:** a doomed run should cost minutes, not its budget (`AUTORESEARCH.md`
§6.3, §7). The FM campaign's canaries were watched by hand.

**Requirement.** Canaries are **in-process**: the check runs inside the training/eval loop as
a callback, and on a fired signature the run exits nonzero with a canary-named message; the
monitor (R6) just records the reason in R1. This is a self-destruct button, not an external
sniper — an out-of-band watcher streaming in-training metrics would mean log-scraping (which
the methodology says truncates silently, §9.11) or a metrics side-channel nobody has budgeted.
The FM "month canary" was already effectively an in-process check. Each campaign declares its
canary signatures; each must expose *this experiment's* failure mode (`AUTORESEARCH.md` Iron
Rule #7):

- **Target leak** → ~perfect decision metric at step 1.
- **Reward hacking** (RL campaigns) → KL / entropy collapse.
- **Recsys / retrieval leakage** → absurd offline lift.
- Universal: NaN loss, gradient blow-up, embedding collapse (mean pairwise cosine → 1).
- The FM-specific "month canary" (a trivially-predictable field learned via a late phase
  transition) is the template: a positive canary can also *diagnose* (its absence exposed
  per-position undertraining at long context — `AUTORESEARCH.md` §9.7).

## R4 — Per-idea, per-rung cost caps enforced at submit time

**Problem it solves:** no budget ceiling existed; spend authority was an informal
human-in-the-loop ("the PI approves every dollar"). `AUTORESEARCH.md` §6.4 lists this as
missing tooling.

**Requirement — estimate at submit AND hard-cap at runtime.** A submit-time estimate alone is a
pinky promise: the docs' own estimates are ±50%, so a run that blows past its estimate would
spend unwatched. Two halves, both required:
1. **Submit-time gate:** the launcher (R2) estimates a run's GPU-hours and refuses runs that
   exceed the rung cap or the campaign's remaining envelope (`BUDGET_POLICY.md`).
2. **Runtime kill:** every generated job carries a **hard wall-clock `timeout`** derived from
   the cap (estimated hours × a safety margin). The job aborts itself at the ceiling regardless
   of state. Estimate + hard abort = an *enforced* cap; R3's canaries only catch *broken* runs,
   not merely *slow* ones — the timeout catches the slow ones.

Crossing a rung boundary (smoke→proxy→full) or the campaign envelope requires explicit PI
approval — the launcher escalates via the same boundary the PI defended by hand: money,
irreversibility, "is this real?" (`CLAUDE_WITH_ANYSCALE.md` §8).

## R5 — Proxy calibration harness

**Problem it solves:** "a proxy that can't [reproduce known rank order] is a random-number
generator with a GPU bill" (`AUTORESEARCH.md` §3, §6.5).

**Requirement.** One command: build the proxy rung, replay 2–3 variants whose full-scale
results are already in R1, report rank correlation and effect-direction agreement. A proxy
that fails calibration blocks promotion decisions based on it. Proxy results are reported as
lift-vs-proxy-baseline over ≥3 seeds with CIs, never as absolutes.

---

## R6 — Durable job monitors (replace the ~25 hand-written bash loops)

**Problem it solves:** the FM campaign hand-wrote ~25 poll loops over `anyscale job status`,
each with a hand-tuned grep filter (`CLAUDE_WITH_ANYSCALE.md` §2). Fragile rule: "the
monitor must match *every* terminal state — silence is not success."

**Requirement.** The launcher attaches a monitor per run that: echoes on state change; on
*any* terminal state pulls the decision + guard metrics and failure signatures; writes the
**single terminal R1 row** (per R1's single-writer rule — the run itself only writes
`RUNNING` heartbeats); and, for chained experiments, submits the dependent run off the predecessor's
completion event (not a timer, not hope). Must match every terminal state
(`SUCCEEDED|FAILED|ERRORED|TERMINATED|OUT_OF_RETRIES`).

## R7 — Pinned-eval & environment-parity enforcement

**Problem it solves:** a device/library-version flip moved a headline by 0.05 AP and
re-ranked conditions (`AUTORESEARCH.md` §9.2); the eval artifact had to be frozen by hand.

**Requirement.**
- The pinned eval artifact (rows/prompts/episodes/tiles + seeds + decoding/sampling params +
  metric implementation) is content-hashed; the hash is the `eval_pin` in R1.
- The eval environment (library versions, device) is pinned in the job image; a campaign
  reports its whole table under one pin before narrating any difference. "If a harness flip
  re-ranks your conditions, your differences were noise."
- **One protocol module** (Iron Rule #3): split/sampling/features/metrics live in exactly
  one importable place consumed by both the repro gate and the pipeline.

## R8 — Artifact safety (never delete — move aside)

**Problem it solves:** a deleted embeddings dir cost a $4 / 50-min GPU re-extraction; a
failed run moved winning artifacts aside and died before writing replacements
(`AUTORESEARCH.md` Iron Rule #4, `CLEANUP.md` step 0).

**Requirement.** Every rerun moves prior artifacts to `<name>_old_<stamp>` (one shared stamp
per rerun, across *every* artifact kind: model, embeddings, tokenized, downstream). Restore
scripts are idempotent and guarded (a double-submit is a no-op, not a disaster). Checkpoint
paths are absolute/URI. This is baked into the R2-generated entrypoint, not left to the agent.

## R9 — Fresh-context review agents (the highest-leverage move, made routine)

**Problem it solves:** "an independent agent with fresh context found the decisive design
flaw I'd missed for days" (`CLAUDE_WITH_ANYSCALE.md` §3). Independence is load-bearing and
was invoked ad hoc.

**Requirement.** The loop dispatches, at defined gates, general-purpose agents wearing role
prompts, each carrying the **independence clause** ("do NOT anchor on the owner's notes;
form conclusions from code + data + literature; assume it's wrong until you can't break it"):
- **Adversarial teardown** (before trusting a design) — code + literature only, banned from
  the owner's notes; ends with "if you had 3 GPU runs, which 3?"
- **Leak audit** (on any too-good result) — one-word verdict `BROKEN`/`SUSPICIOUS`/`CLEAN`
  with file:line, plus the control experiments required before publishing.
- **Idea vetter** (on a hypothesis backlog) — rank by cost/impact/confound; veto bundles
  that confound the question a run is meant to answer.

Every review requires file:line / URL citations, verify-before-report, and a decisive
ranked verdict — not a file dump.

---

## Non-functional requirements (the rigor gates, each paid for)

These are conditions on *every* campaign, enforced by the harness or the loop:

1. **Reproduce before you claim.** No "beat it" experiment spends budget until both the
   artifact gate and the pipeline gate are green in R1. No number, no claim.
2. **The disk is the source of truth.** Every result is a durable file; every decision is a
   commit or an R1 row. A dead context (human or agent) resumes from disk alone.
3. **One experiment = one flag = one commit**, defaulting OFF, with a flag-off byte-parity
   test and a commit message naming the experiment. Backtrack = flip the flag.
4. **Smoke the entry point, not the unit** — the real script e2e on tiny data before any
   paid run (`AUTORESEARCH.md` §2 verification ladder).
5. **Bootstrap-CI every reported number**; paired resampling for ordering claims; the count
   of positives drives the eval plan.
6. **A faithful third-party replication is a first-class control**, not duplicated work — the
   strongest ablation and the source of env-sensitivity forensics (`AUTORESEARCH.md` §9.10,
   `ZGARNER_INTEGRATION.md`). Coordinate parallel replications; don't dedupe them.
7. **Use-license gate.** If the intended use of a result is anything beyond private research,
   "may we legally use this?" is a pre-registration gate, not per-campaign memory. Every seed
   plan declares `commercial_use ∈ {yes, no, needs-legal}` covering the reference's *code AND
   weights AND every dataset license* against the intended use. A `no`/`needs-legal` blocks
   budget until the human signs the license risk — no spending on a result that can't be used for
   what it's for. (The seeds already surface the landmines by hand: Criteo CC-BY-NC is
   non-commercial (03); OpenVLA weights under Llama-2's >700M-MAU clause (06); ESM-3
   non-commercial (08, dodged); MovieLens research-only (01). This makes the check a gate, not
   a habit.)
8. **A cost/efficiency beat is a distinct claim type from a metric beat**, with its own rigor
   rule. When a campaign's headline is "cheaper/faster at held quality" (04 BEIR, 07 Chronos,
   09 Whisper), the reference usually published *no* cost number — so the baseline is
   self-built and can be made arbitrarily slow. The baseline MUST be *a competent single-GPU
   implementation given the same hardware budget, reasonably tuned* — never the reference's
   demo script — and the quality metric (nDCG@10 / WQL / WER) must be held equal within CI.
   Report it as an efficiency result on a fair baseline, never as "beating the published
   number" (there wasn't one). Anything less is racing a strawman you built to lose.

## Build order (proposed)

R1 (registry) and R2 (launcher) first — they retire the manual monitors (R6), manual
chaining, manual ledger, and manual cost tracking (R4) simultaneously. Then R7 (eval pin) +
R8 (artifact safety) as launcher-generated defaults. Then R3 (canaries), R5 (proxy
calibration), R9 (review agents) as the loop matures. R6 and R9 can be scripted cheaply
early (they already exist as patterns) even before the full launcher lands.

**Guard against over-building from n=1.** This spec generalizes from *one* campaign — some of
R1–R9 were load-bearing, some were just what happened to work that week, and we can't yet tell
which. So the forcing rule: **build only R1 + R2 as a couple hundred lines of Python, run the
cheapest Wave-1 campaign on just that, and let campaign #2 rewrite this document.** Don't
implement R3–R9 speculatively; let the second campaign's pain justify each one. The docs are a
hypothesis about the harness, not its spec-in-stone.
