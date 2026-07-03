# Performance & scale storylines — source material for the notebooks

These are the *real* performance/scale problems hit while building and running the
transaction-FM pipeline on the full IBM TabFormer benchmark (24.4M txns, seq_len 4096,
29M-param Llama decoder). Each is a candidate teaching moment — the whole point of the
series is to show **Ray addressing performance/scale issues**, so capture them as concrete
symptom → root cause → fix → *what Ray did* → where it belongs in the series.

Legend for "where": **04** pretrain · **05** embed · **06** downstream · **09** scaling-up
(the dedicated bottleneck notebook).

---

## 1. GPU capacity + the 4096-token pretrain won't fit a 16 GB card
**Symptom (2026-07-02 session):** A10G / L4 / L40S all `LaunchFailed` on capacity; only
T4 (g4dn, 16 GB) available. The seq-4096 Llama pretrain OOMs a T4, and even when it ran it
was ~2 h/epoch ≈ 16 h — never finished.
**Root cause:** at seq 4096 the causal-LM full-vocab logits (`batch × seq × vocab`) plus
attention activations blow past 16 GB.
**Fix (all live in code):**
- instance: `g4dn.xlarge` (T4) → `g5.xlarge` (A10G, 24 GB) — `job_config.yaml`
- `torch.autocast(bfloat16)` — `src/pretrain.py`
- `model.gradient_checkpointing_enable()` — `src/model.py:71`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — `src/pretrain.py:34`
- small per-worker batch (4), global batch 32 via 8-way DDP
**What Ray did:** Ray Train ran **DDP across 8 single-GPU nodes**. The model is ~29M params
so gradients are tiny — cross-node all-reduce is cheap and **FSDP is unnecessary**. Single-GPU
nodes bin-pack exactly (no stranded fractional GPUs) and dodge multi-GPU-instance capacity
shortages. **Where: 04 + 09.**

## 2. Ray Train worker-group startup timed out during autoscale
**Symptom (this session):** `WorkerGroupStartupTimeoutError: worker group startup timed out
after 60.0s waiting for 8 workers`, once, right at launch.
**Root cause:** only 1 A10G was warm; the autoscaler needed >60 s to bring up the other 7.
Ray Train's default `RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S=60` fired before they arrived.
**What Ray did:** the v2 controller's `FailurePolicy=RETRY` (max retries ∞) **auto-rescheduled**
and reached `RUNNING` on the retry once the nodes were warm — no manual intervention. Bump the
env var to pre-empt the noisy first failure. **Where: 09** (cold-start vs. warm-pool tradeoff;
fault-tolerant controller).

## 3. Embedding extraction OOM'd in fp32 at batch 256
**Symptom (this session):** `CUDA out of memory. Tried to allocate 16.00 GiB` in the SDPA
attention during `MapBatches(EmbeddingExtractor)`.
**Root cause:** `configs/full.yaml embed.batch_size` was **256** — a value tuned back when
sequences were 512 tokens (pre-rearchitecture). At seq 4096 in **fp32** the attention
activations are ~16× larger and blow the 24 GB budget. The embed forward wasn't using autocast.
**Fix:** `torch.autocast(bfloat16)` in the embed `__call__` (`src/embed.py`) + batch **256 → 64**.
**The deeper lesson (ties into #4b below):** the *naive* fix is "just lower the batch size,"
but that trades memory for GPU utilization. The *real* fix is **bf16**, which halves the
memory so you can keep the batch high **and** fit. batch 64 was conservative; with bf16 there's
likely headroom for 128. **Where: 05 + 09.**

## 4. GPU utilization — two separate stories

### 4a. Stranded GPUs: 24 provisioned, only 16 used (= 67%)
**Symptom (this session):** `ray status` shows `16.0/24.0 GPU` — the "67%" on the dashboard.
The embed stage runs **16 actors** (`embed.num_workers=16`) but **24 GPUs are up**, so **8 sit
idle** (billed, doing nothing). The cluster is a lumpy mix: 20× `1xa10g` singles + 1× `4xa10g`
(4-GPU) node.
**Root cause:** (a) the 8 GPU nodes from pretrain (stage 04) hadn't scaled down before embed
(stage 05) scaled up to 16; (b) the autoscaler satisfied part of the demand with a 4-GPU
instance, overshooting the 16 needed with a lumpy shape.
**Levers to explore:** shorter idle-node termination so stage-04 GPUs release before stage 05;
pin the worker group to `1xA10G` singles so it bin-packs exactly (no 4-GPU straggler); size the
actor pool to the provisioned cluster (or cap it). The template goal: **provisioned GPUs should
track each stage's actual parallelism**, and idle nodes should scale down *between* stages.
**Where: 09** (this is the headline "match resources to the workload" story).

### 4b. Per-GPU saturation of the 16 working GPUs (open — needs measurement)
**Hypothesis:** even the 16 busy GPUs may not be SM-saturated, because (i) batch 64 is small
for seq 4096, so each GPU may idle between batches waiting to be fed/collated, and (ii) the
CPU Parquet-read stage backpressures (see #5) — if reads can't keep 16 GPUs fed, they starve.
**Not yet measured** — needs `nvidia-smi`/dashboard SM-util per GPU, not just allocation.
Distinguish from 4a: 4a is "GPUs allocated but idle"; 4b is "GPUs allocated, working, but not
at 100% SM." **Action:** measure SM util on a future run; test batch 128 (bf16 headroom).
**Where: 09.** *(Flagged by Zach 2026-07-03 as a storyline to investigate, not fix now.)*

## 5. Heterogeneous CPU→GPU streaming pipeline (a Ray win, not a bug)
**Observation (this session):** during embed, `ReadFiles` shows
`[backpressured: tasks(DownstreamCapacity), outputs(DownstreamCapacity)]` and ~60 GiB of blocks
queued in the object store, while the 16 GPU actors stay fed.
**What Ray did:** Ray Data's **streaming executor** runs the CPU Parquet read and the GPU embed
map **in one topology** and **auto-backpressures** the fast CPU stage to match the slower GPU
consumer — no manual staging, no intermediate Parquet round-trip, bounded memory. This is the
positive framing: the GPU is the bottleneck *by design*, and Ray keeps it fed without the CPU
stage running away and OOMing the object store. **Where: 05 + 09** (GPU-starved-by-CPU bottleneck).

## 6. Embed is throughput-bound at seq 4096
**Observation (this session):** ~5.23M eval windows on 16× A10G ran ~53k rows/min → ~90 min
wall-clock. The seq-4096 forward is genuinely expensive.
**What Ray did / lever:** embarrassingly parallel — throughput scales ~linearly with GPU actor
count. Wall-clock is a straight "add more `1xA10G` replicas" dial (subject to #4a right-sizing).
**Where: 05 + 09** ("scale out to cut wall-clock" — but pair it with #4a so you don't strand GPUs).

## 7. Centralized scoring OOM'd the driver (prior session, already fixed)
**Symptom:** downstream eval pulled the 2.44M × 512 test split onto the 30 GB driver → OOM.
**Fix:** `evaluate()` scores via `map_batches` **on the cluster** and returns only thin
`(proba, label, weight)` columns (`src/downstream.py`, nb 06).
**Lesson:** never `.to_pandas()` / collect a full split to the driver — score distributed,
return summaries. **Where: 06 + 09.**

## 8. CPU-path actor pool tripped the autoscaler (prior session, already fixed)
**Symptom:** an embed actor with `num_gpus=0` and no `num_cpus` has *zero* min scheduling
footprint → the autoscaler's bundle-count estimate is infinite → assertion during scale-up.
**Fix:** `num_cpus=1` on the actor (`src/embed.py`) gives it a finite footprint.
**Where: 05 / 09** footnote.

---

## Quick tuning knobs referenced above
| stage | file | knob | this run | note |
|---|---|---|---|---|
| pretrain | `configs/full.yaml` | `pretrain.batch_size` | 4 | per-worker; ×8 workers = global 32 |
| pretrain | `src/pretrain.py` | bf16 autocast + grad-ckpt + expandable_segments | on | the trio that fits seq-4096 on 24 GB |
| embed | `configs/full.yaml` | `embed.batch_size` | 64 | was 256 (fp32 OOM); bf16 likely allows 128 |
| embed | `configs/full.yaml` | `embed.num_workers` | 16 | GPU actor count = wall-clock dial |
| all | `job_config.yaml` | worker instance | `g5.xlarge` (A10G) | was T4; 24 GB needed for seq 4096 |
