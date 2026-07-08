# AUTORESEARCH — a framework for benchmark-anchored ML hill climbing

A domain-generic loop for "there is a published result/recipe; reproduce it,
then beat it" campaigns: an NVIDIA blueprint, an open post-training cookbook
(VibeThinker-style RL/SFT recipes), a VLA on a manipulation suite, a rec
system on public interaction logs, a foundation model in a new domain
(satellite imagery, pathology, audio, ...). Distilled from a live campaign
(fintech transaction FM vs NVIDIA's TabFormer blueprint — used below as the
worked example; see TEARDOWN.md / EXPERIMENT_LOG.md for its full history).

Vocabulary used throughout:
- **reference** — the published system: its number(s), its protocol, and its
  SHIPPED code/checkpoints (not its blog prose).
- **decision metric** — the one number claims are judged on (AP, pass@1,
  success rate, NDCG@k, mIoU...); everything else is a guard metric.
- **pinned eval artifact** — the frozen, versioned thing you evaluate on
  (eval rows, prompt set + decoding params, episode/task list, tile set).
- **rung** — a fidelity level on the cheap→expensive ladder.

## 0. The Iron Rules (domain-independent; each one was paid for)

1. **Reproduce the reference before any model work — from its CODE.**
   Two gates, in order: (a) *artifact gate*: run the reference's shipped
   eval on the reference's shipped checkpoint and match its reported number
   (proves you hold their protocol); (b) *pipeline gate*: produce that same
   number through YOUR harness (proves your data/eval stack is theirs).
   Read the shipped code, not the write-up — in our campaign the reference's
   decisive detail (its embeddings were single-transaction, not sequence)
   contradicted its own blog, and only the code knew.
2. **Pin the evaluation.** Freeze the eval artifact (rows/prompts/episodes),
   the seeds, the sampling/decoding parameters, and the metric
   implementation as versioned files. Determinism receipt: rerunning the
   gate must reproduce the number (byte-exact for deterministic domains;
   within a pre-declared CI for stochastic ones like RL rollouts or
   sampled LLM decoding — then N eval seeds is part of the pin).
3. **One protocol module.** Split/sampling/features/metrics live in exactly
   one importable place; the repro script AND the pipeline consume it. Two
   implementations of a protocol will diverge, silently.
4. **Never delete artifacts — move aside** (`<name>_old_<stamp>`), one
   shared stamp per rerun, checkpoints AND derived artifacts alike. A
   preserved checkpoint later became our most important experiment; a
   deleted embeddings dir cost a re-extraction.
5. **Dump configs verbatim** into the experiment tracker (whole config
   objects, original key names, zero filtering) and encode key knobs in run
   names. Hand-picked summaries drift; renamed keys break grep.
6. **Smoke the ENTRY POINT, not the unit.** Before any paid run: execute
   the real script/main end-to-end on tiny inputs. Failures live in the
   glue (writers, health checks, eval readers), not the tested core.
   Campaign score: entry-point smokes caught 4 bugs pre-GPU; the one
   skipped smoke killed a job AFTER its expensive stage had succeeded.
7. **Make failures cheap inside paid runs:** stream partial results out the
   moment they exist; idempotent stage skip-guards; explicit resume over
   blind retry; canary metrics that expose the failure mode of THIS
   experiment (a target leak reads as ~perfect accuracy at step 1; reward
   hacking reads as KL/entropy collapse; recsys leakage reads as absurd
   offline lift).
8. **One experiment = one flag = one commit.** Every change is a config
   knob defaulting to OFF, with a flag-off byte-parity test and a commit
   message naming the experiment. Backtrack = flip the flag; attribution =
   ledger + git log.
9. **Keep a ledger** (claims → required evidence → status → cost), wrong
   results and their root causes included. The ledger is what lets a fresh
   context — agent or human — resume the campaign without re-deriving it.
10. **Audit the inputs against the reference before training anything.**
    Field-by-field / channel-by-channel / token-by-token: your treatment vs
    the reference's vs the literature's. Our biggest flaw was input
    deletion (per-event geography reduced to a constant) that no amount of
    training could have fixed. For a post-training recipe this audit is
    data mixture + prompt template + reward spec; for imagery it is bands/
    resolution/normalization; for VLA it is observation/action spaces.

## 1. The loop (state machine)

```
        ┌──────────────────────────────────────────────────────┐
        ▼                                                      │
  DIAGNOSE — cheap, artifacts-only: probes on existing         │
   checkpoints/outputs, health metrics, training curves,       │
   the reference's code                                        │
        ▼                                                      │
  PROPOSE hypotheses — each: mechanism + expected effect on    │
   the decision metric + the single cheapest validating        │
   experiment + its rung                                       │
        ▼                                                      │
  REVIEW — independent agents with FRESH context:              │
   (a) adversarial teardown: code + literature only, banned    │
       from the owner's notes (else it returns your own        │
       beliefs with citations)                                 │
   (b) idea vetter: rank cost/impact/confounds; veto bundles   │
       that confound the question a run is meant to answer     │
        ▼                                                      │
  IMPLEMENT behind flags + unit tests + entry-point smoke      │
        ▼                                                      │
  RUN at the lowest INFORMATIVE rung                           │
        ▼                                                      │
  SCORE vs the pinned eval; append to the ledger ──────────────┘
   promote / kill / ablate per §3
```

In our campaign the teardown agent found the decisive fact the owner had
missed for days, and the vetter vetoed a confounded bundle while
contributing a better idea than the one it rejected. Independence is
load-bearing.

## 2. Verification ladder (before ANY paid run)

1. Unit tests of the changed transform: shapes, grads, flag-off parity,
   known leak canaries.
2. **Local entry-point e2e** on synthetic/miniature inputs — the real
   script, including writes, health checks, and the eval reader. Minutes.
3. Only when the risk is genuinely new at scale (schema change, new data
   modality): a `--limit`-style subset run on real artifacts. Skip when
   rung 2 already covered the failure class — extra environments add ssh
   round-trips, not safety.
4. The paid run, with monitors watching state + failure signatures + this
   experiment's canary metric. Monitors must match every terminal state;
   silence is not success.

## 3. The fidelity ladder — fast hypothesis testing

Three rungs with DIFFERENT epistemic jobs:

| rung | what it is | what it may decide |
|---|---|---|
| smoke | tiny/synthetic, CPU/1-GPU, minutes | does the code run — NOTHING about quality |
| **proxy** | reduced-fidelity real task | rank ideas; kill losers |
| full | the pinned benchmark | publishable claims |

**Choosing the fidelity axis is domain work.** Reduce the axis that scales
cost while preserving the mechanism your hypotheses act on:

| domain | proxy axes | rare-signal trap to protect |
|---|---|---|
| tabular/event FM | subsample ENTITIES (cards/users), keep sequences whole | keep ALL positive-bearing entities; subsample only negative-only ones |
| LLM post-training (VibeThinker-style) | smaller base model; subset of training mix; fewer RL/SFT steps; EVAL SUBSET of the benchmark suite | rank transfer across model scale is the weakest link — calibrate it explicitly before trusting; pin decoding params + N eval seeds (sampled evals are noisy) |
| VLA / robotics | sim before real; task/episode subset; shorter horizons | success rates on few episodes have huge CIs; fix episode seeds; per-task breakdown, not just the mean |
| rec systems | user subsample; shorter time window | temporal leakage across the window cut; popularity effects distort small samples — stratify by activity |
| imagery FM (sat/pathology) | geographic/site subsample; lower resolution; fewer channels | site/scanner/domain shift — hold out whole sites, never random tiles; class rarity per region |

**Proxy construction rules (universal):**
- Same protocol CODE as full, smaller inputs; pinned like the real one.
- **Calibrate before trusting:** replay 2–3 variants whose full-scale
  results you already have; the proxy must reproduce their rank order /
  effect directions. A proxy that can't is a random-number generator with
  a GPU bill.
- Decide on **confidence intervals over ≥3 seeds**, never point estimates;
  report proxy results as lift-vs-proxy-baseline, not absolutes.
- **Promotion rule (successive halving over ideas):** proxy-run wide and
  cheap; promote to full only when the lift CI excludes zero AND the
  mechanism story survived review. Record kills in the ledger.

**Where hyperparameter search fits:** two different problems —
- *Discrete ideas* (new input, new objective, new readout, new reward):
  the ladder above; each idea is a flag.
- *Continuous knobs within a committed idea* (lr, loss weights, KL coef,
  temperature): Ray Tune + ASHA AT THE PROXY RUNG (each trial = one proxy
  run), then one full-scale confirmation of the winner. Never HPO at full
  scale first; never HPO an idea without a proxy-scale pulse — we nearly
  swept XGBoost params over embeddings that a $2 probe then showed carried
  no signal. Probe before sweep.

## 4. Parallelism

- **Git worktrees**, one per experiment branch. The flag discipline makes
  merges trivial (all changes default-off) and lets N implementations
  proceed concurrently.
- **Artifact namespaces** per experiment (`<artifact>/<scale>_<exp>`).
  Runs sharing a canonical artifact (the model dir, tokenized data) MUST
  serialize — chain job submission off the predecessor's completion event,
  never off hope.
- Cluster-level parallelism is free: one ephemeral job cluster per
  experiment; concurrent experiments at different scales ran all day
  without interference once dirs were disjoint.

## 5. Substrate notes (Anyscale/Ray specifics that bit us)

- GPU worker groups: advertise `resources: CPU: 0` so CPU-heavy stages
  can't scale them up (`custom_group_order` REPLACES the default ranking
  that normally protects you) — but keep ≥1 CPU-capable worker group; the
  head advertises CPU:0 too, and data-plane tasks need CPU:1 somewhere.
- The autoscaler scales on pending logical requests, not utilization.
- Throughput knobs are model changes: a batch-size bump that halves
  optimizer steps silently degrades everything (update-count-bound
  regimes). Ablate infra changes like architecture changes.
- Fault tolerance stack: per-epoch checkpoints + FailureConfig (training),
  Ray Data job-level checkpointing keyed on a unique id column (batch
  inference) — both make spot instances safe; checkpoint paths must be
  absolute/URI.
- Long workspace commands die with the ssh session: setsid + nohup + log
  file, and poll the RESULTS FILE (pgrep of your own pattern matches the
  ssh command carrying it).
- `iter_torch_batches(dtypes=...)` must cover every column or be None.

## 6. Tooling roadmap (what upgrades this from discipline to product)

Have: flags, ledger, monitors, review agents, entry-point smokes,
skip-guards, verbatim config dumps, per-experiment commits.

Missing, in priority order:
1. **Results registry** — every run appends one JSONL row {commit, flags,
   rung, seeds, decision+guard metrics, cost, wall-clock}; promote/kill
   reads it programmatically. (Tonight that state lived in one agent's
   context window — exactly what a fresh context can't resume.)
2. **Experiment launcher** — `experiment.yaml` (base + flag deltas + rung +
   seeds) → generated job spec → submit → auto-score → registry append.
   Hand-written job specs hosted two of our infra bugs.
3. **In-training canary aborts** (leak signatures, NaN, collapse, KL blowup)
   so a doomed run costs minutes, not its budget.
4. **Per-idea, per-rung cost caps** enforced at submit time.
5. **Proxy calibration harness** — one command: build proxy, replay known
   variants, report rank correlation.

## 7. Blank template — starting a new campaign

```
[ ] Pick the reference: published number + protocol + SHIPPED code/ckpts
[ ] Artifact gate: reproduce their number with their artifacts
[ ] Pipeline gate: reproduce it through YOUR harness (one protocol module)
[ ] Pin the eval artifact, seeds, decoding/sampling params, metric impl
[ ] Input audit: every field/channel/token — yours vs theirs vs literature
[ ] Adversarial teardown agent (fresh context; code + lit only)
[ ] Build + CALIBRATE the proxy rung; declare the promotion rule and seed
    count up front
[ ] Hypothesis backlog: mechanism + single validating experiment + rung;
    vetted by an independent agent
[ ] Loop: flag → verify ladder → lowest informative rung → score → ledger
    → promote/kill
[ ] Publication claims map 1:1 to ledger entries (no number, no claim)
```

## Appendix: the worked example

The fintech campaign this framework is distilled from: gate = NVIDIA's
TabFormer fraud benchmark (0.9885/0.1238) reproduced through the pipeline at
0.9875/0.1421 five times byte-stable; input audit found per-event geography
deleted; teardown found the reference's embeddings were single-transaction;
probes exonerated the classifier and convicted the readout/objective; the
fixes shipped as flags (RUN-1 readout surgery, RUN-2 input parity, RUN-2b
vetted architecture additions) with a deferred confound (G1) parked behind
an off-by-default flag. Full history: TEARDOWN.md, EXPERIMENT_LOG.md,
BLOG_NOTES.md, and the RUN-tagged commit series.
