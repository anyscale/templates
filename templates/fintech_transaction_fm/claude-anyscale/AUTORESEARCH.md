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

## 8. Prior art & further reading (URLs verified 2026-07)

The gap this framework fills: autonomous research loops exist and close
end-to-end, but experimental RIGOR is their documented weak layer — 42% of
the AI Scientist's experiments failed on coding errors (Beel et al.,
https://arxiv.org/abs/2502.14297); only 6 of 19 of CodeScientist's
machine-generated "discoveries" survived expert review
(https://arxiv.org/abs/2503.22708). The Iron Rules are that missing layer.

**Autonomous loops & their failure modes** (→ Rules 2, 7; the review agents)
- The AI Scientist / v2 — Sakana, 2024/25. https://arxiv.org/abs/2408.06292,
  https://arxiv.org/abs/2504.08066 — the existence proof; v2's agentic tree
  search over experiments prefigures the fidelity-ladder search.
- Sakana "AI CUDA Engineer" incident, 2025 — https://sakana.ai/ai-cuda-engineer/
  + https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/
  — agent exploited its own eval (claimed ~3.1x fell to ~1.5x after
  decontamination): THE case for pinned evals audited adversarially.
- Recent Frontier Models Are Reward Hacking — METR, 2025.
  https://metr.org/blog/2025-06-05-recent-reward-hacking/ — frontier agents
  exploiting grader bugs knowingly; the threat model canaries must assume.
- AIDE — Weco, 2025. https://arxiv.org/abs/2502.13138 — tree search over ML
  code scored by automated evaluation (the scaffolding behind top MLE-bench
  results). Curie — UMich, 2025. https://arxiv.org/abs/2502.16069 — explicit
  "rigor modules"; closest prior art to a built-in rigor layer.
  AI Co-Scientist — Google, 2025. https://arxiv.org/abs/2502.18864 —
  tournament-Elo hypothesis promotion. Agent Laboratory — 2025.
  https://arxiv.org/abs/2501.04227 — human ideation + automated execution.
  AlphaEvolve — DeepMind, 2025. https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
  — the pinned evaluator as THE load-bearing component.
- Measuring it: MLE-bench (https://arxiv.org/abs/2410.07095), MLAgentBench
  (https://arxiv.org/abs/2310.03302), RE-Bench (https://arxiv.org/abs/2411.15114
  — agents beat experts at short budgets, lose at long ones: run many cheap
  parallel attempts), PaperBench (https://arxiv.org/abs/2504.01848 —
  replication graded by 8,316 rubric sub-tasks), MLGym
  (https://arxiv.org/abs/2502.14499).

**Why "reproduce the baseline first" is a rule, not a taste** (→ Rules 1, 10)
- Dacrema et al., RecSys 2019. https://arxiv.org/abs/1907.06902 — most
  published neural recsys lost to tuned simple baselines.
- Musgrave et al., A Metric Learning Reality Check, ECCV 2020.
  https://arxiv.org/abs/2003.08505 — four years of claimed gains ~vanish
  under a fair pinned protocol.
- Melis et al., 2017. https://arxiv.org/abs/1707.05589 — equal tuning budget
  makes plain LSTMs beat "superior" architectures; tuning is a confound.
- Kapoor & Narayanan, Patterns 2023. https://arxiv.org/abs/2207.07048 —
  leakage taxonomy across 294 papers / 17 fields; our fraud-burst join bug
  is their taxonomy in the wild. The input-audit stage encodes this.
- Pineau et al., JMLR 2021. https://arxiv.org/abs/2003.12206 — reproduction
  as gating institutional process (NeurIPS reproducibility program).

**Multi-fidelity & proxy calibration** (→ §3)
- Hyperband (https://arxiv.org/abs/1603.06560) and ASHA
  (https://arxiv.org/abs/1810.05934) — the successive-halving math; Ray Tune
  (https://arxiv.org/abs/1807.05118) — the substrate.
- muTransfer, Tensor Programs V. https://arxiv.org/abs/2203.03466 — the
  theoretical license for tune-small-transfer-large (13M-param tuning beat
  published BERT-large HPs).
- GPT-4 Technical Report. https://arxiv.org/abs/2303.08774 — loss and
  HumanEval predicted from 1,000-10,000x-cheaper runs: the industrial proof
  that calibrated cheap rungs forecast expensive outcomes.
- The cautionary NAS literature: NAS evaluation is frustratingly hard
  (https://arxiv.org/abs/1912.12522 — protocol tricks + seed variance
  manufacture fake gains); EcoNAS (https://arxiv.org/abs/2001.01233 — which
  reduced-fidelity settings preserve rank order); NAS-Bench-Suite-Zero
  (https://arxiv.org/abs/2210.03230 — proxy rank correlations are
  inconsistent across tasks, sometimes negative: calibrate per-domain,
  never assume).
- Schaeffer et al., NeurIPS 2023. https://arxiv.org/abs/2304.15004 —
  "emergence" as metric artifact; metric choice decides whether proxy-scale
  signal exists at all.

**Ledgers & engineering discipline** (→ Rules 5, 8, 9; §2)
- Karpathy, A Recipe for Training Neural Networks, 2019.
  http://karpathy.github.io/2019/04/25/recipe/ — the human protocol this
  loop mechanizes ("neural nets fail silently; build the eval skeleton
  first; change one thing at a time").
- Sculley et al., Hidden Technical Debt in ML Systems, NeurIPS 2015 — why
  pipeline hygiene belongs inside the research loop.
- OPT-175B Chronicles — Meta, 2022.
  https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles
  — the founding public experiment logbook (~2k incidents in 3 months, raw).
- The Smol Training Playbook — Hugging Face, 2025.
  https://huggingfacetb-smol-training-playbook.hf.space/ — codified
  "derisking": one variable per ablation, never adopt untested changes;
  ablations cost >half the final run's GPU-hours and are worth it.
- Marin — Stanford, 2025. http://marin.community/blog/2025/05/19/announcement/
  — experiments-as-code with unique run IDs and full lineage: a working
  design for the results registry in §6.

**Live reproduce-then-improve campaigns** (→ the post-training use case)
- TinyZero — Berkeley, 2025. https://github.com/Jiayi-Pan/TinyZero — R1-Zero's
  "aha moment" reproduced for <$30 on a 3B model: the archetype of a
  calibrated micro-scale proxy for an expensive result.
- Open-R1 — Hugging Face, 2025. https://huggingface.co/blog/open-r1 — an
  organized community campaign validating a published pipeline step-by-step
  before extending it.
- SimpleRL-Zoo — 2025. https://arxiv.org/abs/2503.18892 — replication across
  10 base models sorting which claimed phenomena generalize: replication
  breadth as a claim filter.

## 9. Day-2 addendum — the eval is an experiment too (paid for 2026-07-08)

Day 2 took the win through controls, a context-scaling act, an environment-
parity gate, and a second task. Every rule below was purchased with a
wrong-narrative near-miss or a failed job, same as section 0.

1. **A decision metric on few positives is a rumor.** The pinned 100k eval
   holds 112 frauds; single-draw AP moves ±0.05–0.08. We drafted TWO false
   context-scaling narratives from point estimates ("monotone", then "peaks
   at 1024") before CIs existed. Rule: bootstrap CI on every reported
   number; the count of POSITIVES (not rows) drives the eval plan.
2. **Harness sensitivity is a treatment effect — measure it.** Flipping
   XGBoost device+version moved one condition by 0.05 and re-ranked the
   conditions while leaving another untouched. Pin the eval environment to
   the reference's (their library versions, their device) and rerun the
   whole table under the pin BEFORE narrating differences. If a harness
   flip re-ranks your conditions, your differences were noise.
3. **If the benchmark cannot answer your question, upgrade the benchmark,
   not the claim.** The 100k test is a stratified random subsample — so
   evaluating the ENTIRE test period is the same protocol with sampling
   variance removed (2,724 frauds, ~5x tighter CIs). But AP is
   eval-set-specific (our full-period baseline scores 0.208 vs 0.140 on the
   draw): keep TWO tables — published-comparable and internal-tight — and
   never compare numbers across them.
4. **Ordering claims need PAIRED resampling.** Conditions scored on
   identical rows deserve per-draw differences (report P(A>B) and a diff
   CI), not overlapping marginal intervals. Pairing resolved a strict
   1024 > 512 > 2048 ordering where marginal CIs said "unresolvable".
5. **Tighten the eval BEFORE the discriminating experiment.** The 40-epoch
   2048 continuation would have landed inside the 100k eval's noise and
   answered nothing; sequencing the full-period eval first made it decisive.
6. **Warm-restart continuation = the half-price discriminator.** Deliberate
   run-name reuse (the auto-resume "gotcha", weaponized) + a raised-epochs
   config continues a run from its last checkpoint under a recomputed
   cosine. Fine for is-it-undertrained-or-saturated diagnosis; a disclosed
   protocol deviation, not a headline row.
7. **Watch canary FIELDS, not just losses.** Trivially-predictable fields
   (calendar fields = copy a visible neighbor) are learned via late, sharp
   phase transitions gated on positional/attention health — their absence
   at long context exposed per-position exposure halving (windows/epoch
   halve as context doubles). And pretext accuracy is not downstream value:
   the 1024 model beat 512 downstream with WORSE MLM accuracy.
8. **The readout thesis generalizes — sweep readouts before judging a
   representation.** Same frozen embedding on next-merchant: 0.077 (masked
   state) -> 0.184 (zero-shot InfoNCE table) -> 0.397 (linear) -> 0.535
   (MLP+context) — a 7x swing with zero pretraining changes. Any verdict on
   a representation obtained through one readout is a verdict on the readout.
9. **Beat the strongest cheap baseline you can build, not the weakest one
   quoted.** Static train-period top-10 memorization = 0.598; the honest
   causal full-history floor = 0.647 (and recency decay HURT — measure,
   don't assume). Literature check: frequency baselines routinely beat
   neural sequence models under fair evaluation, so clearing the honest
   floor via a prior-blend (0.658) plus a blind-slice decomposition (35% of
   events where the baseline is structurally 0.000, model recovers 0.16) is
   the durable form of the claim.
10. **A faithful replication by someone else is your best control.** The
    parallel branch that rebuilt the reference verbatim (vendored tokenizer,
    exact architecture, pinned versions) became (a) proof the reference
    reproduces, (b) the strongest possible ablation for our thesis ("their
    design, trained by us, 4x below ours"), and (c) the source of the
    env-sensitivity forensics. Coordinate such efforts; do not deduplicate
    them.
11. **Persist results as artifacts; never scrape logs.** Job-log streaming
    truncates silently. Every stage writes its JSON/parquet to durable
    storage and every consumer reads those. (The day we had to `cat` a
    results file through a throwaway cluster job was the smell.)
12. **Loader tolerance for removed flags:** on checkpoint load, drop unknown
    arch keys whose value is falsy (feature was off — weights identical),
    raise loudly if truthy. Paid for when post-cleanup code refused a
    pre-cleanup checkpoint mid-eval.
13. **Old rules, re-earned:** idempotent guarded restore scripts (an
    accidental double-submit was a no-op instead of a disaster); mv-aside
    on every rerun; micro-test fresh numeric code locally before any GPU
    job (caught a wrong test expectation and an HR-denominator bug at zero
    cost); and budget host RAM for eval paths that hold multiple copies of
    an N x d matrix (the no-PCA path OOM'd a 32GB host at 1.2M x 512).

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
