# Blog notes — transaction FM vs NVIDIA's blueprint (data-point tracker)

Working thesis: **a FATA-style static/dynamic encoder (1 position per
transaction) matches or beats NVIDIA's decoder blueprint (12 tokens per
transaction) on their own benchmark, at an 8x cheaper sequence — and then
scales to context lengths their tokenization can't afford.**

Everything below is a claim we want to make, the exact data point that
proves or kills it, and where that number comes from. No number, no claim.

## Claim 0 — "We're on their data and protocol" (the gate) ✅ HAVE

| | Test ROC-AUC | Test AP |
|---|--:|--:|
| our pipeline, baseline (13 raw feats) | **0.9875** | **0.1421** |
| standalone CSV repro | 0.9873 | 0.1469 |
| NVIDIA blog | 0.9885 | 0.1238 |
| NVIDIA notebook 01 | 0.9914 | 0.1424 |

- Job `fintech-fm-nvidia-baseline` (2026-07-07), CPU-only, zero FM stages.
- Protocol pinned in `src/nvidia_baseline.py`; rows pinned in
  `benchmark.parquet` (seed 42, stable sort, written once).
- Blog beat: "before claiming an FM helps, reproduce the baseline — ours was
  off by 0.22 AUC until we did" (EXPERIMENT_LOG.md has the confession arc).

## Claim 1 — RESOLVED 2026-07-08: the FM WINS, decisively

Final overnight numbers (all on the pinned benchmark; full history in
TEARDOWN.md, seed CIs + adversarial audit + fair-head control done):

| | ROC | AP | vs baseline |
|---|--:|--:|--:|
| baseline (13 raw, their XGB) | 0.9875 | 0.1421 | — |
| NVIDIA combined (their headline) | 0.9925 | 0.1755 | +23.5% |
| ours, protocol-faithful (their PCA+XGB harness, embed-only) | 0.9914 | 0.1623 | +14.2% |
| **ours, embed-only, stable heads (no PCA)** | **0.997** | **0.23-0.26** | **+60-82%** |
| ours, embed-only MLP (report as range) | 0.994-0.998 | 0.33 +/- 0.10 | — |
| fair-head control: same torch heads on raw 13 | 0.47-0.61 | ~0.001 | (the lift is the embedding) |

The story: input parity + target-position readout turned a -98% embedding
into one that beats their published FUSION headline embedding-ONLY, at 1
position/txn vs their 12 tokens/txn. Mechanism (audited, measured): fraud
is bursty (90% of test frauds have a prior fraud within the same card's
last 512 txns vs 7.3% of normals) and the history readout legitimately
detects mid-burst cards from auth-time-legal features. Disclosures: 1,394
val-period (ZERO test) txns visible to pretraining via a cutoff mismatch
(fix queued); reco regressed (HR@10 0.077) — one-backbone-two-tasks needs
rework; remaining pre-publish controls: shuffled-label sanity + classical
burst-aggregates baseline (queued).

### Original framing (historical)

Status 2026-07-07 evening — the corrected (fixed-join, 100% row match) seq-512
numbers vs the deterministic 0.9875/0.1421 baseline:

| variant | embed-only ROC/AP | combined ROC/AP | combined AP lift |
|---|---|---|---|
| b128 model (UNDERTRAINED: loss 13.9 vs 10.3) | 0.7154 / 0.0023 | 0.9863 / 0.1016 | **-28.5%** |
| xl seq-1024 (converged; biased join, re-score pending) | 0.7209 / 0.0022 | 0.9862 / 0.0719* | +15.8%* vs crippled 0.0621 |
| navy seq-512 (converged, loss 10.28) x {last, mean} pooling | ⏳ running | ⏳ | the decisive test |

*not quotable until re-scored with the fixed join.

Embed-only ROC ~0.72 on BOTH models (incl. the converged xl) says the pooled
embedding barely ranks fraud even though user/card identity is in the FM.
Leading suspects: last-position pooling on a bidirectional encoder (mean
being tested), then objective mismatch (MLM field marginals vs their CLM
next-txn anomaly signal). If navy x mean is still negative -> pivot to
objective work before spending anything on HPO.

Two war stories captured on the way (both good blog material):
- Batch-size trap: bumping 64->128 to "saturate GRAM" halved optimizer steps
  (3,180->1,600) and degraded EVERY per-field metric; sqrt-lr scaling did not
  save it. Small-model pretraining is update-count-bound. TB caught it.
- Join-collision trap: (card_id, minute) join keys collide on same-minute
  bursts, which are disproportionately FRAUD; deduping dropped ~189 of the
  most informative train rows and halved baseline AP (0.0621 vs 0.1421).
  Fixed by keying on (card_id, ts, amount-in-cents) -> 100.00% match.

### Original data-point checklist (still what we need)

Their 4096-token context = ~315 txns (12 tokens/txn). Our 512 positions =
512 txns. Needed from `downstream/full/benchmark_metrics.json`:

- [ ] baseline / embeddings / combined: test ROC-AUC + AP (+ val, best_iteration)
- [ ] `combined_lift_ap_pct` — the headline vs their **+41.8%** AP
       (0.1238 → 0.1755; combined ROC-AUC 0.9925)
- [ ] `embeddings`-only vs baseline (theirs UNDERPERFORMS baseline —
       if ours does too that's expected; if ours doesn't, that's a bonus beat)
- [ ] `pca_explained_variance` (theirs: reported in their notebook run)
- [ ] pretrain curves from TB run `fm_full_seq512_b128x4_lr6e-4_*`:
       mlm_loss, per-field acc, InfoNCE ramp

Success = combined AP lift comfortably > 0 and in the neighborhood of theirs.
Honest confounds to state in the blog either way:
- their EMBED/COMBINED XGB params were Optuna-tuned for THEIR embeddings
  (we reuse them untuned — a conservative handicap for us)
- their pooling is last-token decoder state; ours is our encoder's pooling
- our FM is ~29M params vs their decoder (report their param count)

Fallback narrative if lift is small/negative: the efficiency story still
stands (Claim 0 + cost table), and we HPO the fusion XGB as follow-up work.

## Claim 2 — "Context scales where theirs can't" ⏳ RUNNING (xl seq-1024 job)

- [ ] xl `benchmark_metrics.json`: same trio; is combined AP > full's?
- [ ] xl baseline ≈ 0.987 again (independent re-verification of the gate on
       a re-derived benchmark — determinism receipt)
- [ ] cost per context length: 512 vs 1024 pretrain wall-clock + GPU-hours
       (job durations; cluster events)
- [ ] optional third point: xxl seq-2048 (config exists) if 1024 > 512
- Framing table: to see 1024 txns their tokenizer needs ~12k tokens,
  2048 → ~24k; O(S²) attention makes that the wall. We pay 1024/2048
  positions flat.

## Claim 3 — efficiency table (compute once numbers are in) ▢ TODO

| | NVIDIA blueprint | ours |
|---|---|---|
| tokens per txn | ~12 | 1 |
| context window | 4096 tok = ~315 txns | 512–2048 txns |
| attention cost @315 txns | (4096²) | (315²) → ~169x less |
| params | (their model card) | ~29M |
| pretrain GPU-hours | (blog/notebook if stated) | [ ] from jobs |
| embed throughput (rows/s/GPU) | — | [ ] from 04 logs |

## Ops/Anyscale angle (the platform half of the post) ▢ GATHER

- [ ] stage wall-clocks from job logs (01 benchmark, 02 tokenize, 03 pretrain,
      04 embed, 05 xgb, 06 reco) — full + xl
- [ ] autoscaling profile screenshot AFTER the CPU:0 fence (GPU nodes at 0
      until 03, exactly 4 up, then 8 for embed, down after) — pairs with the
      war story: custom_group_order silently replaced the default ranking
      that avoids GPU nodes for CPU work; fix = advertise CPU: 0 on the GPU
      group (+ Ray Train GPU workers request zero CPUs by default)
- [ ] $ cost of the full run (job cluster cost from console)
- [ ] one-command reproducibility: job_baseline.yaml / job_full.yaml / job_xl.yaml

## Claim 4 — infra economics (Anyscale vs their monolith) ▢ DRAFTED, verify numbers

Their facts (README + blog, 2026-07): the SHIPPED checkpoint (what their
published numbers ride on) was trained ~3,000 steps on 8x A100 (README L96);
the reader rig is 1x A100 80GB / H100 (L108) running everything in one NeMo
container (cuDF tokenize, XGBoost, train, embed — "every step runs on the
GPU"). The repo config is a 30-step DEMO (max_steps: 30, "~2 min") — readers
reproduce their inference, not their pretraining; the real 3,000-step run
arrives as a git-lfs artifact. 3,000 steps x batch 16 x 4096 tok ≈ ~200M
tokens into a ~29M-param model. Blog states no duration and no cost.
Reproducibility beat: our pipeline retrains from scratch every run, on
commodity GPUs, one `job submit`, with pinned benchmark rows + config dumps.

| | theirs | ours |
|---|---|---|
| topology | one GPU box occupied end-to-end | per-stage: CPU nodes for data, GPUs only for 03/04 |
| pretraining hardware | 8x A100 node (~$33-41/hr; how the shipped ckpt was made) | 4x g5.xlarge A10G ($1.01/hr each) |
| inference/downstream rig | 1x A100 80GB (~$5/hr) | 8x A10G for embed, CPU for XGBoost |
| GPUs during tokenize | the A100 (cuDF) | 0 (verified live w/ CPU:0 fence) |
| fault tolerance | torchrun in a notebook | per-epoch ckpt + FailureConfig + job retry -> spot-safe |
| spot discount | n/a in blueprint | ~65% off GPU (g5 spot ~$0.35/hr) |
| full-pipeline cost | ~$5-40+ (occupancy math, no published timings) | ~$6 on-demand, ~$3.50-4 spot (measured wall-clocks) |

One-liner: they scale UP (bigger GPU doing everything); we scale OUT and to
ZERO (each stage gets exactly the hardware it needs; idle costs nothing).
Unique extras: Ray Serve real-time path; `anyscale job submit` reproducibility.

- [ ] recompute our $ from the new jobs' actual wall-clocks + cluster cost in console
- [ ] verify current us-west-2 pricing at publish time
- [ ] optional demo: run one job on spot GPUs and show a preemption restore in logs

**DO NOT lead with training cost.** Their 3k-step run is ~200M tokens into
29M params ≈ 3.4e16 FLOPs — plausibly 5-30 MIN on 8x A100 (~$5-10). At demo
scale everyone's training is cheap; a reviewer kills that headline with
6*params*tokens math. The durable infra arguments instead:
1. EMBEDDING is the recurring production cost (re-score every txn, forever;
   scales with volume) — commodity A10Gs + scale-to-zero compound there.
2. TabFormer is 24M rows; a real issuer is 100-200x with recurring retrains —
   the heterogeneous/spot/right-sizing PATTERN is what you pick for that.
3. Accessibility (A100-80GB + quota vs any-cloud A10Gs) and reproducibility
   (their 30-step demo config vs our from-scratch one-command retrain) hold
   regardless of timing.

## Secondary material (use sparingly)

- Reco (06): frequency-dominated task — 95% repeat merchants; freq baseline
  HR@10 ~0.64 vs FM ~0.37 at 512. Honest "not every downstream task needs an
  FM" sidebar, or cut entirely.
- Pretraining behavior: day_of_week acc ≈ chance (1/7) — unlearnable field;
  error head near-trivial (0.98). Good "what the FM actually learns" color.
- TB hparams: full config dumped verbatim per run (Text tab) — reproducibility beat.

## Artifact locations

- metrics: `/mnt/user_storage/transaction-fm-v2/downstream/{full,xl}/benchmark_metrics.json`
- TB: `/mnt/user_storage/transaction-fm-v2/tensorboard/` (run names carry hparams)
- benchmark rows: `/mnt/user_storage/transaction-fm-v2/raw/{full,xl}/benchmark.parquet`
- history/mistakes: `EXPERIMENT_LOG.md`; protocol: `src/nvidia_baseline.py`
