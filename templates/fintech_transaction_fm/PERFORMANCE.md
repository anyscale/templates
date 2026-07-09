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

### 4a. Stranded GPUs: a CPU-only stage dragged up GPU nodes for their vCPUs (confirmed 2026-07-04)
**Symptom (2026-07-04, nb 05 embed, interactive workspace):** `ray status` showed **`6.0/28.0 GPU`**
— 28 A10Gs provisioned (20× `1xa10g` + 2× `4xa10g`), still climbing toward ~36 (2 more `4xa10g`
launching), while only **6 GPU actors** were actually running. So ~22 GPUs sat billed and idle.
**Root cause (confirmed, not the earlier guess):** the resource requests told the story —
`{'CPU':1.0}: 584` alongside `{'CPU':1,'GPU':1}: 6`. The embed pipeline's first step,
`balanced_eval_sample`, is a **CPU-heavy Ray Data scan/sample over the 24.4M eval rows** that
requested ~**584 vCPUs**. The **workspace compute config had GPU nodes only** — no CPU-only group —
so the autoscaler satisfied that CPU demand the only way it could: by launching A10G nodes **for
their vCPUs**, stranding every one of those nodes' GPUs. (An earlier run's "stage-04 GPUs hadn't
scaled down + a lumpy 4-GPU node" was a minor contributor; the *dominant* cause is the missing CPU
node group. The transient nature matters too: the spike is during sampling and should relax toward
the ~16 the actor pool needs once embedding dominates — but you pay for the over-provision meanwhile.)
**Fix (decided 2026-07-04):** give the cluster a **CPU-only worker group** so CPU-only demand never
lands on GPU nodes. `job_config.yaml` (the *Job* path) already had this — a `cpu-workers`
(`m5.4xlarge`) group **plus** an `instance_ranking_strategy: custom_group_order [cpu-workers, gpu-1x]`.
The **interactive workspace** cluster did not, which is why notebook runs hit it and the Job wouldn't.
Add the same to the workspace compute config. Recommended CPU node: **`m5.8xlarge`** (32 vCPU / 128 GB —
general-purpose, 4 GB/vCPU headroom for the wide 4096-token arrays; *not* the skinny c-family),
`min_nodes: 0` (scales to zero, no idle cost) / `max_nodes: 16` (~512 vCPU, soaks the ~584 burst;
the remainder queues briefly). The `group_order` flag is the load-bearing half — without it,
price-based defaults still route CPU tasks onto the cheaper-per-node GPU instances.
**What Ray does once the CPU group exists:** the autoscaler routes the CPU sample stage to CPU nodes
and only brings up A10Gs for the actual GPU actors, so **provisioned GPUs track real parallelism**
instead of shadowing CPU demand.
**Where: 09** (the headline "match resources to the workload" story) — with the sharper sub-lesson
that **interactive-workspace and Job compute configs can diverge**, so a fix in one isn't a fix in
the other.

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

## 9. Elastic 8×A10G provisioning — the capacity story, resolved (2026-07-04)
**Symptom (good kind):** a clean full pretrain from an idle cluster. nb 04 requested 8 GPUs
(`{'GPU':1.0}*8`, PACK placement group); the autoscaler went from 1 warm A10G to 8 running in
**~2.5 min** (12:36 submit → 12:38:41 all 8 training), *including* recovering from one
`4xa10g-48cpu-192gb: LaunchFailed` by falling back to eight `1xA10G` nodes. Ran 8 epochs in
**1 h 59 m** (~14.9 min/epoch), final ppl 1.662, then **auto-terminated all 8 workers** back to
the head when the job ended.
**What Ray did:** the autoscaler matched a fleet to the workload on demand and released it after —
no pre-provisioning, no stranded GPUs, and it routed around a capacity failure without intervention.
This is the counterpart to §1/§2: same A10G capacity flakiness, now absorbed by elastic scheduling
+ the PACK group waiting for a complete worker set before `fit()`. **Where: 04 + 09** (the
"match resources to workload / scale up then back down" beat). Runtimes logged to
`/mnt/cluster_storage/transaction-fm/PRETRAIN_MONITOR.log` via `scripts/monitor_pretrain.py`.

## 10. Ray Data re-reads the pretrain set every epoch (candidate optimization)
**Symptom:** each epoch logs a fresh `Dataset train_2_N execution finished in ~440–490 s` — i.e.
~8 min/epoch to re-read + re-split the 66k pretrain sequences from Parquet, roughly half of the
~15 min/epoch wall time.
**Root cause (to confirm):** the training loop re-iterates the Ray Dataset per epoch, so the read
pipeline re-executes each time rather than reusing a materialized copy.
**Candidate fix:** `materialize()` the pretrain dataset once (it fits — 66k sequences) so epochs 2–N
read from the object store, *if* the read isn't already overlapping GPU compute via streaming
backpressure (§5). **Needs measurement** before claiming a win — this is an honest open item, not a
confirmed bottleneck. **Where: 09** (only if verified to be non-overlapped).

## 11. Killed-mid-run partial writes vs. the mtime cache guard (operational gotcha, 2026-07-04)
**What happened:** nb 05 was interrupted twice while streaming embeddings to Parquet. Ray Data
writes shards incrementally with no atomic commit, so each kill left a **partial** `embeddings/full`
(e.g. 114,987 rows, `val` split not yet reached, only 1,072 frauds — vs the complete ~3.4M / all
splits). The content-aware cache guard added this session (`src.paths.stale_or_missing`, mtime-based)
then treats that half-written directory as **"current"** — it's newer than the model — and the next
nb 05 run would **skip** and silently reuse the partial. So an interrupted stage needs its output dir
**manually cleared** before re-running.
**Root cause:** mtime tells you *when* a directory was last written, not *whether the write finished*.
Staleness-by-timestamp can't distinguish complete from partial.
**Fix options (not yet done):** write a completion marker (`_SUCCESS`) after the stage finishes and
have the guard require it; or record the expected row count and verify it. Until then: **if you stop
a stage mid-run, delete its output dir before re-running.** (This is the counterpart caveat to the
caching fix — the guard prevents *stale-upstream* reuse, but not *partial-self* reuse.)
**Where: dev-workflow note** (not a notebook teaching moment) — belongs in the run instructions /
CHANGES.md, not the reader-facing narrative.

## 12. Embed-stage stranding: a CPU-bound read drags up GPU nodes (2026-07-04, nv_downstream)
**Symptom:** running the NVIDIA-protocol downstream (`nv_downstream.py`, 8 GPU embed workers) the
cluster climbed to **19 A10G provisioned / 2 in use** while ~250–430 CPU were busy. The embed's
input read — decoding the sampled **seq-4096 token arrays** from tokenized eval — is CPU-heavy, and
the autoscaler satisfied that CPU demand by launching A10G nodes **for their vCPUs**, stranding ~17
GPUs. This is §4a again, now in the *embed* stage rather than the *sampling* stage: adding a CPU node
group helped but did not fully confine the read to CPU nodes — GPU nodes still got pulled for CPU.
**Why this matters (Zach's framing):** *this is exactly what Ray/Anyscale is supposed to right-size.*
A heterogeneous CPU-read → GPU-forward pipeline should schedule the read on CPU nodes and bring up GPU
nodes only for the actual GPU actors. That the default behavior over-provisions GPU nodes to serve a
CPU-bound read is a real gap to address in the template's scaling story — not hand-wave.
**Fixes (for the notebooks / a clean rerun):**
- **Don't re-embed what already exists.** `embeddings/full` already holds the train+test embeddings
  (single-txn, same model). nv_downstream re-embedded 1M train + 100K test + 100K val = ~1.2M windows,
  but only the **~100K val** was actually missing. Embedding the redundant 1.1M is most of the waste.
  Reuse existing embeddings; embed only the gap.
- **Confine the read to CPU nodes** — e.g. a placement/resource constraint so the seq-4096 decode can't
  land on GPU-node vCPUs, and/or cap the GPU worker group to exactly the actor count so idle GPU nodes
  can't be provisioned for CPU spillover.
- **Or embed at the single-transaction level up front** (max_ctx=14) so the read isn't decoding full
  4096-token arrays it doesn't need.
**Cost note:** on 2026-07-04 we let one such run finish rather than kill+restart (billed hourly, GPUs
already committed → same cost to completion). The lesson is to prevent the over-provision, not to
tolerate it. **Where: 09** (the headline "match resources to the workload" story — including where the
default autoscaling gets it wrong and what to constrain).

## 13. Downstream XGBoost is unstable on enriched-train / natural-test (2026-07-04)
**This is the real root of the weak raw/fusion — not the features, not the FM.** Diagnosis chain
(all measured, full 2.44M holdout unless noted):
- Our raw features are *faithful* (loader: `card_id=User*100+Card`, `merchant_id=Merchant Name`);
  swapping carried→source features gave the same 0.048, ruling features out.
- A/B on identical features: **fixed 400 trees, no early stop → raw 0.049; early stopping on a
  natural-rate val → raw 0.206 (best_iter=2).** So the original bug was: our downstream *removed*
  early stopping and trained a fixed pile of trees, which **overfits the fraud-enriched training
  sample and collapses on the natural-rate test.** (We removed early stopping earlier thinking the
  1–2 tree result was degenerate — it was correct.)
- But the fit is **pathologically unstable**: raw swung **0.169 → 0.088** between two runs differing
  only in early-stop patience (20 vs 30 rounds) → 5 vs 42 trees. Fusion bounced 0.072–0.147 across
  recipes. The optimum is only ~2–5 trees and the cliff is steep.

**Why unstable (root cause):** (a) training on a fraud-*enriched* balanced sample (2.5–10% fraud)
but evaluating at the *natural* 0.11% rate is a train/test distribution mismatch; (b) early stopping
on val **AUC does not track test AP** — the metric we actually report — so maximizing val-AUC (more
trees) overfits test-AP.

**Fixes that target the cause (NOT blind parameter search):**
1. **Early-stop on AP (`aucpr`), not AUC** — match the stopping objective to the reported metric.
2. **Cut the enrichment mismatch** — train nearer the natural rate using `scale_pos_weight` instead
   of aggressive resampling, so train and test distributions agree and the fit stops moving.
3. **Use the proper natural-rate temporal VAL period** for early stopping (not a test holdout).
4. **Or just fix a small tree count (~5)** — the optimum is tiny and stable there; sidesteps the
   early-stopping fragility entirely.

**Status:** raw reaches 0.17–0.21 (beats NVIDIA 0.124) once early-stopped; fm ~0.045 (beats their
0.012); best fusion so far 0.147 (below their 0.176) but on an unstable fit — a stabilized recipe
(fix #1/#2) is expected to lift it, since raw alone is already ~0.17–0.21 and the FM adds signal.
**Where: 06 downstream (this is a `src/downstream.py` change) + 09.**

## 14. Scale-to-zero vs. warm floors — elasticity is a per-pool policy knob (2026-07-09)

**Observation (Stage 1 of the Ray Data rebuild, measured live):** with `min_workers=0`
everywhere, the CPU pool sawtoothed 0 → 56 CPUs → 0 → 72 CPUs across three split runs in
one afternoon, each cold start costing ~2–3 min of waiting (matches the 8×A10G ~2.5 min
from §9). Node *mix* varied per run (3×8cpu+1×32cpu, then 2×32cpu+1×8cpu) — the autoscaler
fills the demand from whatever satisfies it fastest, so "how many nodes" is the wrong
metric; total CPUs is the right one.

**The trade, honestly:** scale-to-zero overhead = `startup_time / job_time`. For the 2-hour
pretrain that's ~2% (obviously right to scale to zero — and it's the *expensive* pool).
For the ~8-min split job it's ~30% (noticeable). For interactive development — many short
runs separated by thinking — it's the dominant cost of the afternoon.

**The resolution is asymmetric by pool cost:** give the cheap CPU pool a warm floor
(`min_workers=1` on the 32-CPU group ≈ ~a dollar-plus/hr) during dev or before a live demo
(pre-warming 5 min before presenting = a temporary floor); keep the GPU pool strictly
min=0 — idle A10Gs/A100s are exactly what elasticity exists to prevent. Set per node
group in the workspace/job compute config. NOT optimized in this repo on purpose —
kept as a discussion point.

**Demo/deck framing:** "elasticity is a policy you set per pool — floor for latency,
ceiling for cost — and you set it differently for a \$0.30/hr CPU node than an A100."
Contrast with the blueprint's fixed single node: it pays for its GPU whether working or
idle, and can't grow past it. **Where: nb01 scalability section (one line), nb09 (the
full discussion), deck segment 3.**

## Quick tuning knobs referenced above
| stage | file | knob | this run | note |
|---|---|---|---|---|
| pretrain | `configs/full.yaml` | `pretrain.batch_size` | 4 | per-worker; ×8 workers = global 32 |
| pretrain | `src/pretrain.py` | bf16 autocast + grad-ckpt + expandable_segments | on | the trio that fits seq-4096 on 24 GB |
| embed | `configs/full.yaml` | `embed.batch_size` | 64 | was 256 (fp32 OOM); bf16 likely allows 128 |
| embed | `configs/full.yaml` | `embed.num_workers` | 16 | GPU actor count = wall-clock dial |
| all | `job_config.yaml` | worker instance | `g5.xlarge` (A10G) | was T4; 24 GB needed for seq 4096 |
