# The Ray/Anyscale angle — verified decisions, claims → receipts, outline

Prep doc for the short "why this was easier/only possible on Ray + Anyscale"
post. Every claim below is checked against the code, the job YAMLs, or the
commit log — file:line and commit receipts inline so the post can't drift
from what we actually did.

## 1. The data-path decisions (verified, with one correction)

### Pretraining DOES stream Ray Data into Ray Train — the real decision is *what* streams

The belief "we didn't use Ray Data → Ray Train in pretraining" is not quite
right, and the truth is a better story:

- Tokenized pretrain windows are built by Ray Data (`map_groups` over 24.4M
  rows), **globally shuffled once, materialized into the object store**, and
  handed to `TorchTrainer` as `datasets={"train": pre}` with **no Parquet
  round-trip** (`run_pipeline.py:216-234`, `pretrain.py:308-312`).
- Workers consume it via `get_dataset_shard("train").iter_torch_batches(...)`
  (`pretrain.py:107,165`).
- The actual design decision: **run the expensive transform once, keep the
  per-epoch randomness cheap.** Tokenization + the all-to-all shuffle happen
  exactly once; per-epoch variation comes from (a) a seeded
  `local_shuffle_buffer_size` — order varies per epoch *without* a per-epoch
  global shuffle (`pretrain.py:170-175`), and (b) MLM masking drawn fresh per
  batch on the torch side (`mask_batch`), not baked into the dataset.
- So the honest line is: *"Ray Data does the once-per-run heavy lifting;
  Ray Train re-iterates the materialized shard every epoch; the random parts
  of the objective live where they're free."* Then `del pre` releases the
  object-store blocks before the embed stage (`run_pipeline.py:247`).
- Job mode (`run_pretrain.py`) deliberately breaks the fusion: stages hand
  off through Parquet on `/mnt/user_storage`, **so eval runs as a separate
  job against saved artifacts** — that separation is what made
  eval-only re-runs (pinned eval, fulltest, per-epoch probe) cheap.

### Where Ray Data was essential: the embedding stage (Geoff is right — and it's stronger than remembered)

The batch-inference stage that creates the XGBoost training set is the one
with **no clean public reference** (their pipeline does this on a single
GPU with RAPIDS-adjacent tooling; `embed.py:1-11` docstring):

- Heterogeneous streaming: CPU parquet read/tokenize → GPU forward pass in
  one lazy pipeline through the object store, **no intermediate disk**
  (`extract_embeddings(ds=...)` accepts a lazy dataset; `embed.py:88-90`).
- Model loads once per replica (`EmbeddingExtractor.__init__`), Ray Data
  manages replica lifecycle + GPU placement; **fractional GPUs** pack
  several replicas per A10G since the FM needs a few hundred MB
  (`embed.py:92-95`).
- Per-stage scaling: embed gets **8 GPUs** ("pure inference — free
  speedup") while pretrain stays at 4 "so training dynamics are fixed"
  (`job_full.yaml` header) — the monolith-vs-per-stage-scaling
  infra-economics claim from the old BLOG_NOTES (commit e06ed42f).
- The **fulltest eval is itself a Ray Data story**: 3.5M windows tokenized →
  embedded → scored as ONE job per scale on autoscaled 0→8 A10Gs
  (`fulltest_eval.py` header, `job_fulltest.yaml`), reusing the existing
  checkpoint — no retraining to upgrade the benchmark.
- Also Ray Data: the ~100k-merchant vocab (distributed frequency scan with
  reader-level column pruning, `run_pipeline.py:184-191`) and tokenization
  with right-sized shuffle (`shuffle_partitions` per scale; RayTurbo's
  numba-JIT hash shuffle — Anyscale-runtime specific — per README).

## 2. Claims → receipts

| claim | receipt |
|---|---|
| Parallel context-length experiments = 3 job submissions | `job_full/xl/xxl.yaml` — identical shape, one scale-config knob; 1024 measured 1h50m end-to-end; ~$15–25/run (B15 pending) |
| Whole experiment surface is declarative | 9 checked-in job YAMLs incl. `job_eval_pinned` (env-parity rerun), `job_xxl_continue` (warm restart), `job_paired_bootstrap`, `job_baseline` |
| Heterogeneous, cost-shaped clusters | GPU workers advertise `CPU: 0` so CPU-only tasks can't scale GPU nodes (`job_full.yaml`); `min_nodes: 0` scale-to-zero everywhere; fulltest head sized to the eval (g5.8xlarge, 128GB: "no-PCA eval holds 3.55M × 512 copies") |
| Fault tolerance actually exercised, not brochure-ware | `FailureConfig(max_failures=3)` in-place epoch restore "required for spot GPU nodes" (`pretrain.py:373-376`); every epoch checkpointed (`num_to_keep=None`) → enabled BOTH the per-epoch probe and the 20→40 warm-restart continuation (deliberate `run_name` reuse, commit 734785e6); mid-dataset embed resume (ddf91fd5) |
| Observability produced a *scientific finding* | The month-canary/undertraining result was read off TensorBoard `field_ce/*` per-step curves (`pretrain.py:214-221`); self-describing run names + the ENTIRE config dumped verbatim to the TB Text tab (`pretrain.py:125-143`, commits 85e4b15e/4ae70d9d); events sync to durable storage right after training so later-stage crashes can't lose curves (`run_pipeline.py:243-246`) |
| Env parity as a *scientific control* | Device/version flip moved the 1024 point by 0.05 → `job_eval_pinned.yaml` (xgboost==3.2.0, CUDA) made the eval a controlled experiment; pinned per-job envs are the mechanism |
| Same code laptop → cluster | CI smoke = 1 CPU worker, headline = 4–8 GPU workers; only `ScalingConfig` + scale YAML change (`pretrain.py:1-7`); sanity notebooks on the workspace read the same `/mnt/user_storage` artifacts the jobs wrote |
| Online path | Ray Serve: static-embedding cache + `@serve.batch` micro-batching + autoscaling 1→4 replicas (`serve.py:40-45,130`) |

## 3. The velocity story (from the commit log)

~161 fintech_fm commits over 10 days, **~115 of them in the final 48 hours**:

- **06-29 → 07-02**: model features land; then a 7-commit observability
  hardening burst in ~4h (TB logging, event sync, checkpoint retention) —
  the instrumentation that later paid for itself.
- **07-03/04** (18 + 18): baseline forensics — reproduce NVIDIA exactly
  before any FM claim.
- **07-06** (17): xl scale stood up in an afternoon (seq-1024 on 8×A10G was
  a config file + a job submission — commits 14:11 → 14:40).
- **07-07** (52): the campaign day — overnight R1/R2/G1 runs, TEARDOWN
  written the same night, probes as jobs.
- **07-08** (45): 1024 + 2048 acts, fulltest eval built *and run* on three
  scales, paired bootstrap, reco v3, blog reframed — in one day.

The mechanism behind the cadence, explicitly: (a) experiments are job
submissions against shared storage, so a new question = a new YAML, not new
infra; (b) decoupled stages mean eval-only questions never pay for
retraining; (c) every claim in the blog carries a prodjob id — the ledger
discipline in AUTORESEARCH.md. Frame: **"the unit of experimentation was a
job, so the unit of iteration was an hour."**

## 4. Proposed outline (short post)

1. **Hook**: 3 pretrained FMs (512/1024/2048 ctx), a full NVIDIA-protocol
   benchmark harness, a 2.44M-row eval upgrade, and a beaten published
   headline — ~48 hours of experiments, ~$15–25 per headline run.
2. **The pipeline** (B2 graphic, annotated with per-stage scale knobs +
   which Ray primitive owns each stage).
3. **The stage Ray was built for**: batch embedding extraction (CPU→GPU
   streaming, fractional GPUs, 8-GPU inference vs 4-GPU training) — and its
   encore, the fulltest eval as one autoscaled job.
4. **Training data path done right**: tokenize once → object store →
   `iter_torch_batches`; per-epoch randomness where it's free.
5. **Experiment velocity**: the §3 timeline + parallel seq-len jobs.
6. **Observability that found a result**: month canary from TB; full-config
   dumps; the pinned-env control.
7. **Fault tolerance we actually used**: epoch checkpoints → warm restart +
   per-epoch probe; spot-safe restore.
8. **Close**: template link, 3 commands. + 1–2 sentences: the whole
   campaign was driven through the Anyscale CLI with Claude (claude-anyscale/
   recipes in the template) — submit, tail, fix, resubmit without leaving
   the terminal.

## 5. Don't claim / candor budget

- **Don't claim "only possible on Ray."** Claim: the marginal cost of each
  additional experiment was a YAML + a job submission — and support with the
  timeline. "Could only happen" invites a someone-did-it-on-Slurm rebuttal;
  "48 hours, one person, receipts attached" doesn't.
- Their single-GPU RAPIDS tokenizer is fine at 24M rows — our claim is
  *horizontal scaling + streaming into training*, not that theirs breaks.
- Candor items that buy trust (all already public in README/log): the
  right-sizing we had to do (hash-shuffle aggregators, object-store
  fraction, driver OOM at holdout_keep=1.0, gpus_per_worker OOM at 0.5 on
  seq-512 embed), and the honest "warnings you'll see and why" README block.
- Pretraining framing must match §1 — do NOT say "we skipped Ray Data for
  training"; say we streamed a *materialized* dataset and kept the shuffle
  local.

## 6. Open items for this post

- [ ] B15 exact $ per run (shared with BLOG_ASSETS.md) — headline material here.
- [ ] B2 annotated architecture graphic (shared).
- [ ] Pull 2–3 dashboard/TB screenshots: field_ce/month canary, autoscaler
  0→8 ramp during fulltest, Serve replica scaling (nice-to-have).
- [ ] Confirm RayTurbo numba hash-shuffle detail is safe to publish as
  Anyscale-runtime-specific.
- [x] Distributed-XGB scale-out variant — DONE 2026-07-10 on the research
  branch (`scripts/distributed_xgb.py` + 3 job yamls). Verified: CUDA-dist
  AP 0.3139 INSIDE the single-node CI [0.2849, 0.3201]
  (prodjob_3by9j9fjbzmimt553tbwvzpx61), m5.4xlarge head. War stories in the
  draft: sharded-val early stopping (run 1, 0.2642) and the CPU/CUDA device
  effect reproduced (0.2806 vs 1-worker control 0.2598). Blog section
  updated with real code + receipts.
