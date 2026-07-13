# Autoresearch budget & compute governance policy

In the fintech-FM campaign, spend authority was an informal human-in-the-loop: "the PI
approves every dollar of GPU spend" (`CLAUDE_WITH_ANYSCALE.md` §8), per-scale compute was
hardcoded in each `job_*.yaml`, and there was **no cost cap, no shared budget layer, and no
recorded cost ledger** — the dollar figures were placeholders pulled from the console after the
fact (`AUTORESEARCH.md` §6.4 lists submit-time cost caps as *missing* tooling). This document
formalizes cost discipline so a campaign can run largely unattended without spending past its
envelope.

## Principle: GPU-hours are the currency, not dollars

Dollar prices drift with cloud, region, spot market, and reservation. **GPU-hours are
stable**; convert to dollars through one table at the end. Every budget below is stated in
GPU-hours first. Anyscale's own cost guidance is ratio-based for exactly this reason.

**But a raw GPU-hour is not one currency — it's six.** An A100-hour costs 3.5× an A10G-hour;
an H100-hour, 5.5×. A cap stated in raw hours can be gamed by tier choice, and a wave label
assigned by raw hours is wrong by up to 5.5×. So the unit is the **A10G-equivalent hour**:
`raw GPU-hours × rel-cost` from the table below (A10G = 1.0). Every cap, envelope, and wave
threshold in this doc is denominated in A10G-equivalent hours. The results registry stores
both (`cost.gpu_hours` and `cost.a10g_equiv_hours`), and R4 enforces envelopes against the
equivalent, never the raw count (`harness/registry.py`, `REQUIREMENTS.md` R1/R4).

### Conversion table (representative AWS us-west-2 on-demand — confirm against console)

| GPU | Instance | $/hr/GPU (on-demand) | rel. cost (A10G = 1.0) | Notes |
|---|---|--:|--:|---|
| T4 16GB | g4dn.xlarge | ~$0.53 | 0.5 | CI / tiny smokes |
| L4 24GB | g6.xlarge | ~$0.80 | 0.8 | cost-optimal inference/embed |
| **A10G 24GB** | **g5.xlarge** | **~$1.01** | **1.0** | the FM campaign's workhorse |
| L40S 48GB | g6e.xlarge | ~$1.80 | 1.8 | 24GB+ models, FP8 |
| A100 80GB | p4de.24xlarge (÷8) | ~$3.50 | 3.5 | multi-GPU TP, large FMs |
| H100 80GB | p5.48xlarge (÷8) | ~$3–5.50 | 5.5 | frontier training only |

**Cost levers (apply as multipliers to the GPU-hour → $ conversion):**
- **Spot + checkpointing: ×0.3–0.5** (the FM campaign measured g5 spot ~$0.35/hr, ~65% off).
  Default ON for all training/batch runs; on-demand only for the stateful head/learner.
- FP8/BF16 mixed precision: ×0.5 memory, throughput win where supported.
- Batch inference vs online serving: 3–10× cheaper per unit (80–95% vs 30–70% GPU util).
- Reserved: ×0.6–0.7. Not relevant for burst research.

> **Estimates are ±50%.** Anyscale's baseline tables are flagged 20–50% optimistic vs prod.
> Seed-plan cost estimates exist to size envelopes; the proxy-calibration step
> (`REQUIREMENTS.md` R5) tightens them before any full run.

## The three rungs and their caps

Caps are **per run**, enforced at submit time (`REQUIREMENTS.md` R4). A run whose estimated
GPU-hours exceed its rung cap is refused until re-scoped or explicitly approved.

| Rung | Job | Cap (GPU-hr) | ~$ (on-demand) | Who can launch |
|---|---|--:|--:|---|
| **smoke** | code runs e2e on tiny/synthetic data, CPU or 1 GPU, minutes | **≤ 0.5** | < $1 | agent, freely |
| **proxy** | reduced-fidelity real task; rank ideas; kill losers | **≤ 10 / idea** (expect ~1–2) | < ~$15 | agent, within campaign envelope |
| **full** | the pinned benchmark; publishable numbers | per-campaign envelope | see waves | **PI approval to cross into** |

- The FM campaign's rung mapping, for reference: smoke = CPU/1-worker (~$0); proxy = `small`
  2-GPU or reduced-eval; full = `full`/`xl` at 4-GPU train + 8-GPU embed, ~2h, **~$4–6/run
  on spot** (its single hero run). Eval-only reruns (pinned/fulltest/probe/bootstrap) are
  decoupled and cost minutes-to-1h with **zero retraining** — budget them separately and
  cheaply.
- **Continuous knobs (lr, loss weights, KL coef, temperature) are tuned with Ray Tune +
  ASHA at the PROXY rung**, each trial = one proxy run, then one full-scale confirmation of
  the winner (`AUTORESEARCH.md` §3). ASHA kills ~50% of trials after epoch 1; 10–20 trials
  suffice for LLM-scale. Never HPO at full scale; never HPO an idea before a proxy pulse
  proved the idea carries signal (the campaign nearly swept XGBoost params over embeddings a
  $2 probe then showed were signal-free — probe before sweep).
- **The 10 GPU-hr proxy cap is a per-idea ceiling, not the expected spend** (FM proxies ran
  ~1–2 GPU-hr each). A Wave-1 envelope (≤60) holds a gate + ~6–10 proxy ideas at *expected*
  spend + a full run + the cheap control tail — it does *not* hold six ideas each at the 10-hr
  ceiling. Plan against expected proxy spend; size the envelope against the ceiling.

## Per-campaign envelopes (by wave)

Campaigns are grouped into waves by cost and reproduction confidence (see `SEED_INDEX.md`).
Each campaign gets a total GPU-hour envelope covering *all* its runs — gates, proxy sweeps,
the long tail of cheap control/eval jobs, and the hero full runs.

Envelopes are in **A10G-equivalent hours** (see the currency note above) — so a campaign
run mostly on A100/H100 hits its wave ceiling at far fewer *raw* hours than one run on A10G.

| Wave | Profile | Envelope (A10G-eq hr) | ~$ on-demand | ~$ spot | Approval |
|---|---|--:|--:|--:|---|
| **1** | cheap, high-repro-confidence, fast first result (small models / frozen-embedding probes / tiny data) | **≤ 60** | ~$60 | ~$25 | PI signs off envelope; agent runs the loop |
| **2** | mid (multi-GPU training, TB-scale data, or FM fine-tune) | **60–400** | $200–1,400 | $70–500 | PI signs off envelope + each full run |
| **3** | heavy (7B+ VLA/LLM fine-tune, multi-node, large sim-eval fleets) | **400–2,000** | $1,400–11,000 | $500–4,000 | PI signs off envelope, each full run, and any multi-node request |

The whole FM campaign — ~15 jobs, one hero pair, a long tail of minute-scale controls —
plausibly cost **$50–150 of GPU total**. Wave-1 campaigns should land in that band. Waves
exist so we ship a cheap, credible first result before committing to an OpenVLA-scale run.

> **Worked example (why the unit matters).** Campaign 02's 7B RL full run is ~120 **H100**
> hours. In raw hours that reads as Wave 2 (60–400). Weighted: 120 × 5.5 = **660 A10G-eq
> hours → Wave 3.** The registry computes this automatically and the launcher (R4) would
> refuse it under a Wave-2 envelope. A raw-hours policy would have waved the program's single
> most expensive run through the *lighter* approval ritual.

**Program-level cap:** the sum of active campaign envelopes cannot exceed the program's
standing GPU-hour allocation (set by the PI per quarter). New full campaigns queue until an
active one closes or the allocation is raised. This is the only place absolute money enters.

## The escalation boundary (what the PI must approve)

Exactly the boundary the FM PI defended by hand — **money, irreversibility, and results that
need a human's eyes** (`CLAUDE_WITH_ANYSCALE.md` §8). The harness escalates via
`AskUserQuestion`; it does not proceed on these without a yes:

- **Money/compute:** crossing a rung boundary into `full`; any run over its rung cap; exceeding
  the campaign envelope; any multi-node request.
- **Irreversibility:** deleting cluster storage or artifacts (default is move-aside, R8);
  overwriting a pinned eval artifact.
- **"Is this real?":** the sign-off that a reproduction gate is met or a "beat it" claim
  holds — the PI reads the code. No headline number ships without it.

Everything else — smoke runs, proxy sweeps within envelope, config/script edits, job
submit/monitor/chain, literature review, doc drafting, cheap control jobs — the agent does
autonomously.

## Cost accounting (tie-in to the registry)

- `cost` is a **mandatory column** in every results-registry row (`REQUIREMENTS.md` R1):
  `{gpu_hours, gpu_type, usd_est, spot}`. The FM campaign never recorded wall-clocks and had
  to reverse-engineer cost from the console after the fact — this makes cost a first-class,
  queryable output of every run.
- A campaign's spent-vs-envelope is `sum(cost.gpu_hours)` over its registry rows; the
  launcher reads it to enforce R4 caps.
- Reuse the campaign's shipped idle-remediation sweep (`claude-anyscale/scripts/idle_sweep.py`
  + the `idle-workspace-sweep` skill) as a standing guard against paying for idle GPUs
  (a pinned g5.12xlarge idle 3h14m = ~$18 burned, per the campaign's own war story).

## Defaults that save money without a decision

Baked into every R2-generated job spec so the agent never has to remember them:

- **Hard wall-clock `timeout`** on every job, derived from the run's GPU-hour cap × a safety
  margin — the enforced half of the cap (`REQUIREMENTS.md` R4). A run that overruns its estimate
  self-aborts instead of quietly spending the envelope.
- **Spot + `fallback_to_on_demand: true`** for stateless GPU workers; on-demand only for the
  head/learner. `FailureConfig` + per-epoch checkpoints make spot safe.
- **`min_nodes: 0` scale-to-zero** on every worker group; **`resources: {CPU: 0}`** on GPU
  groups so CPU-only stages can't scale GPUs up (keep ≥1 CPU-capable group).
- **Fractional GPUs** for small-model inference/embedding (pack N replicas per GPU; the
  model-to-GPU matrix: MiniLM 0.25, BGE/E5-large 0.5, ResNet-50 0.1, YOLOv8 0.25).
- **Low job priority** for overnight research so runs soak up idle fleet and yield to
  daytime work; checkpointing bounds preemption cost to minutes.
- Right-size the head node to the eval's peak host-RAM (the campaign needed a 128GB
  g5.8xlarge only for the eval path holding a 3.55M×512 matrix — not for training).
