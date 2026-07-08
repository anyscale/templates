# AUTORESEARCH — a framework for benchmark-anchored hill climbing

Distilled from the fintech_transaction_fm campaign (2026-07-07: ~15 cluster
jobs, 3 independent review agents, 6 caught-before-GPU bugs, 2 caught-after).
The goal: a blank-template loop an agent (or human) can run against ANY
"reproduce a benchmark, then beat it" research/blog idea, spending GPU
dollars only on hypotheses that survived cheaper filters.

## 0. The Iron Rules (every one of these was paid for)

1. **Reproduce the published benchmark through YOUR pipeline before any
   model work.** One protocol module (`src/nvidia_baseline.py` pattern) is
   the single source of truth for split/sampling/features/params; the repro
   script and the pipeline import the SAME code. Gate: your baseline number
   matches the published one. (We shipped a 0.76 "baseline" against their
   0.99 for a week before doing this.)
2. **Pin the evaluation rows as an artifact** (`benchmark.parquet`): seeded,
   stable-sorted, written once, read by every stage. Determinism receipts:
   re-runs must reproduce the gate number byte-for-byte (ours did, 5x).
3. **Never delete artifacts — move aside** with one shared timestamp per
   run (`$kind/<scale>_old_$STAMP`). A deleted embeddings dir cost a GPU
   re-extraction; a moved model checkpoint saved the whole Run-1 experiment.
4. **Dump configs verbatim** (whole train_loop_config + whole scale YAML)
   into the experiment tracker; run names carry the key knobs. No renamed
   keys, no hand-picked fields — they drift and break grep.
5. **Smoke the ENTRY POINT, not the unit.** Before any cluster job: run the
   actual script/main on tiny data end-to-end. Failures live in the glue
   (writes, health checks, metric readers), not the tested core. Tonight's
   score: entry-point smokes caught 4 bugs pre-GPU; the one skipped smoke
   cost a 40-min job that died AFTER its 4.8GiB of useful work.
6. **Make failures cheap inside jobs:** print/persist per-stage partial
   results the moment they exist; skip-guards (`[ -d out ] || run`) so
   reruns reuse completed stages; `max_retries: 0` + explicit resume beats
   blind retry-from-scratch.
7. **One experiment = one flag = one commit.** Every change lands as a
   config knob defaulting to OFF (byte-identical old behavior, asserted in
   a unit test), with a commit message naming the experiment (RUN-N tags).
   Backtracking = flip the flag; attribution = the ledger + git log.
8. **Keep a ledger** (TEARDOWN.md / EXPERIMENT_LOG.md pattern): claims →
   required evidence → status, including wrong results and their causes.
   The ledger is what lets a fresh context (agent or human) resume without
   re-deriving the campaign.

## 1. The loop (state machine)

```
        ┌──────────────────────────────────────────────────────┐
        ▼                                                      │
  DIAGNOSE (cheap, artifacts-only)                             │
   probes on existing embeddings/checkpoints; health metrics;  │
   per-field TB curves; read the reference system's CODE       │
        │                                                      │
        ▼                                                      │
  PROPOSE hypotheses (each: mechanism + expected effect +      │
   the single experiment that validates it)                    │
        ▼                                                      │
  REVIEW — independent agents, fresh context:                  │
   (a) adversarial teardown: code+lit only, forbidden from     │
       reading the owner's notes (they inherit beliefs)        │
   (b) idea vetter: rank cost/impact/confounds; veto bundles   │
       that confound the question being asked                  │
        ▼                                                      │
  IMPLEMENT behind flags + unit tests + entry-point smoke      │
        ▼                                                      │
  RUN at the lowest INFORMATIVE rung of the fidelity ladder    │
        ▼                                                      │
  SCORE against the pinned benchmark; append to ledger ────────┘
   promote / kill / ablate per the promotion rules (§3)
```

Agent roles that worked: the teardown agent found the decisive facts
(reference system's eval didn't match its own blog; ranked flaw list with
file:line evidence); the vetter agent prevented a confounded bundle and
contributed a better idea than the one it vetoed (merchant recency). The
independence rule is load-bearing — an agent that reads your notes returns
your own beliefs with citations.

## 2. Verification ladder (before ANY cluster job)

1. `pytest`-style unit of the changed transform (shapes, grads, flag-off
   byte-parity, known leak canaries).
2. **Local entry-point e2e** on synthetic/smoke data — the real script,
   including writes and health checks. CPU, minutes.
3. (Only when schema/scale risk is genuinely new) subset run on real
   artifacts via `--limit` / `--min-match` style knobs — as a job or on a
   workspace. Skip when rung 2 already covered the failure class; tonight
   the workspace detour caught zero bugs at four-ssh-roundtrip cost.
4. Full job, with monitors watching state + failure signatures + the
   experiment's canary metric (e.g., a leak shows as acc→1.0 at step 1).

## 3. The fidelity ladder — how to make hypothesis testing fast

Three rungs, each with a DIFFERENT epistemic job:

| rung | data | job | what it can decide |
|---|---|---|---|
| smoke | synthetic / 2k cards, CPU, minutes | does the code run | NOTHING about quality — flow only |
| **small (proxy benchmark)** | subsample of the real data | is the idea promising | rank ideas, kill losers |
| full | the pinned benchmark | is the claim true | publishable numbers |

**Building the proxy benchmark (the interpolation you asked about):**
- Same protocol CODE, smaller inputs: sample X% of cards (not rows — keep
  sequences intact), then run the identical split/balance/stratify logic →
  `benchmark_small.parquet`, pinned like the real one.
- **Rare-positive trap:** fraud at 0.1% means a naive 10% sample has ~10x
  the AP variance. Mitigations: (a) keep ALL cards that ever have a positive
  and sample only negative-only cards; (b) run every small-rung experiment
  with 3+ seeds and decide on confidence intervals, never point estimates;
  (c) report the proxy metric as lift-vs-proxy-baseline, not absolute.
- **Calibrate before trusting:** run the baseline + 2-3 ALREADY-MEASURED
  variants (you have them from full-scale history) at the proxy scale and
  check the rank order / effect direction transfers. A proxy that can't
  reproduce known full-scale orderings is a random-number generator with
  a GPU bill.
- **Promotion rule (successive halving):** at small, run cheap and wide
  (many ideas × few seeds). Promote to full only if the CI on the lift
  excludes zero and the mechanism story survived review. Kill silently is
  fine; the ledger records the kill and why.

**Where Ray Tune/ASHA fits:** two different search problems —
- *Discrete ideas* (new field, new objective, new readout): the ladder
  above; each idea is a flag; ASHA-across-fidelities is the promotion rule.
- *Continuous knobs within a committed idea* (lr, loss weights, dims,
  temperature): a Tune+ASHA sweep AT THE PROXY SCALE (each trial = a proxy
  pretrain, ~minutes), then a single full-scale confirmation of the winner.
  Never HPO at full scale first, and never HPO an idea that hasn't shown a
  proxy-scale pulse (we nearly ASHA'd a dead readout).

## 4. Parallelism

- **Git worktrees** for concurrent idea implementation: one worktree per
  experiment branch, each with its own flags/commits; merges are trivial
  because everything is flag-gated and defaults-off.
- **Artifact namespaces:** every experiment writes under its own suffix
  (`embeddings/<scale>_<exp>`, `downstream/<scale>_<exp>`); jobs that share
  a canonical artifact (the model dir, the tokenized dir) MUST serialize —
  encode the dependency by submitting the next job from the previous job's
  completion event, not by hoping.
- **Cluster-level parallelism is free:** each experiment is its own
  ephemeral job cluster; two retrains at different scales ran concurrently
  all day without interference (verified: disjoint dirs, no races).

## 5. Infra checklist (Anyscale/Ray specifics that bit us)

- GPU worker groups: advertise `resources: CPU: 0` so CPU stages can't
  scale them up (default ranking does this; `custom_group_order` REPLACES
  the default and silently loses it). But keep ≥1 CPU-capable group — the
  head advertises CPU:0 too, and Ray Data read/write tasks need CPU:1.
- Autoscaler scales on *pending logical requests*, not utilization —
  right-size `shuffle_partitions`/max_nodes or accept over-provisioning.
- Small-model pretraining is update-count-bound: batch-size bumps to
  "saturate GRAM" halve steps and silently degrade every metric. Ablate
  throughput changes like model changes.
- Ray Data job-level checkpointing (id_column) gives mid-dataset resume for
  batch inference; per-epoch checkpoints + FailureConfig give spot-safe
  training. Both need absolute/URI checkpoint paths.
- Long-running commands on workspaces die with the ssh session — setsid +
  nohup + log file + poll the RESULTS FILE (a pgrep of your own pattern
  matches the ssh command carrying it).
- iter_torch_batches dtypes dict must cover every column or none.

## 6. Tooling wishlist (what would make this a real product loop)

Have today: flags, ledger, monitors, review agents, entry-point smokes,
skip-guards, verbatim config dumps, per-experiment commits.

Missing, in priority order:
1. **A results registry**: every job appends {commit, scale, seeds, flags,
   metrics, cost, wall-clock} as one JSONL row; the loop's promote/kill
   decisions read it programmatically (today: grep + memory).
2. **An experiment launcher**: `experiment.yaml` (base config + flag deltas
   + rung + seeds) → generated job yaml → submit → auto-score → registry
   append. Removes the hand-written job-yaml step where two of tonight's
   infra bugs lived.
3. **In-training canary asserts** (auto-abort on leak signatures / NaN /
   collapse) so a bad run costs minutes, not its full budget.
4. **Cost guardrails**: per-idea, per-rung dollar caps enforced at submit.
5. Proxy-benchmark calibration harness (one command: build proxy, replay
   known variants, report rank correlation).

## 7. Blank template — starting a new campaign

```
[ ] Identify the published benchmark + its exact protocol (read the CODE,
    not the blog — ours differed from its own blog in the decisive detail)
[ ] Write the protocol module; reproduce the number through your pipeline;
    pin the rows; record the gate in the ledger
[ ] Feature audit: every raw field → your treatment vs reference treatment
    vs literature treatment, BEFORE training anything
[ ] Adversarial teardown agent (fresh context, code+lit only)
[ ] Build the proxy benchmark; calibrate it on known variants
[ ] Backlog of hypotheses, each with mechanism + single validating
    experiment + rung; vet with an independent agent
[ ] Loop: implement-behind-flag → verify ladder → lowest informative rung →
    score → ledger → promote/kill
[ ] Blog claims map 1:1 to ledger entries with evidence (BLOG_NOTES.md
    pattern: no number, no claim)
```
