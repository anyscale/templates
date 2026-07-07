# Transaction-FM experiment log (2026-07-06 → 07)

A record of the first round of runs comparing this transaction FM ("FATA"-style
encoder) against NVIDIA's transaction-foundation-model blueprint on IBM
TabFormer — **including what we got wrong**, so we don't repeat it. Most of the
numbers below are from a **flawed pipeline** (see "What we got wrong"); they are
kept for reference, not as valid results.

## Goal
1. Reproduce NVIDIA's baseline on TabFormer (prove same data/protocol).
2. Then show whether our FM (sequential encoder w/ MLM + InfoNCE, 1 position per
   txn vs their ~12-token decoder) adds value over that baseline.

## NVIDIA reference (their blog + `01_dataset_baseline.ipynb`)
Protocol: 24.4M txns, ~0.12% fraud, temporal 80/10/10 split, XGBoost on **13
raw features incl. `User` + `Card`** (ordinal-encoded), 1M balanced train, 100k
stratified holdout. AP is the headline metric (AUC saturates at 0.1% fraud).

| model | ROC-AUC | AP (PR-AUC) |
|---|--:|--:|
| baseline (raw features) | 0.9885 (blog) / 0.9914 (nb) | 0.1238 / 0.1424 |
| + FM embeddings (fusion) | 0.9925 | 0.1755 (**+41.8% AP**) |

## Our runs (all OLD/flawed pipeline unless noted)
Scales: `full`=seq 512, `xl`=seq 1024, `xxl`=seq 2048. ~29M-param model
(512d/8L/8H), 4–8× A10G on Anyscale. Decoupled stages: data → tokenize →
pretrain → embed → fraud → reco.

### Fraud (downstream XGBoost: raw vs fm vs fusion)
Our baseline was **crippled** (see below), so absolute numbers are far under
NVIDIA and NOT comparable. Iterations of the baseline as we debugged it:

| baseline variant | raw AUC | raw AP |
|---|--:|--:|
| 4 features (amount/hour/dow/mcc) | 0.768 | 0.0072 |
| 12 feat, frequency-encoded | 0.747 | 0.0056 |
| 12 feat, target-encoded | 0.756 | 0.0056 |
| 14 feat (+ user/card) target-enc | 0.766 | 0.0138 |
| **NVIDIA recipe (correct)** | **0.9873** | **0.1469** |

FM lift over the (crippled) baseline was ROC-positive but AP-flat/negative;
context length (512→1024→2048) did **not** move fraud materially.

### Reco (next-merchant HR@10; old pipeline, all seq lengths)
| seq | FM HR@10 | freq-baseline HR@10 | novel FM HR@10 |
|---|--:|--:|--:|
| 512 | 0.331 | 0.636 | 0.0024 |
| 1024 | 0.279 | 0.640 | 0.0009 |
| 2048 | 0.289 | 0.642 | 0.0011 |

**Reco is frequency-dominated**: 95% of targets are repeat merchants, which a
"recommend most-frequent merchants" baseline nails. The FM loses to it at every
context length; more context doesn't help. The FM only scores novel merchants
(freq = 0 there) but at ~random. **Conclusion: reco is not where FATA's value
lives — it's a task-structure fact, not a bug.**

### Pretraining behavior (xl, seq 1024, old)
- `mlm_loss` 18.5 → 12.65 over 20 epochs (epoch-1 bump = InfoNCE warm-up ramping
  in, then recovers). Healthy convergence.
- Per-field final acc: channel 0.85, error 0.98 (near-trivial), amount_bucket
  0.48, merchant_category 0.39, hour 0.35, mcc 0.28, merchant_bucket 0.12
  (20k-way, far above random), **day_of_week 0.14 ≈ chance (1/7) — unlearnable**.
- Embeddings healthy (mean pairwise cosine ~0.30 = not collapsed).

## What we got wrong (root causes)
1. **`_normalize` dropped the features that matter.** It folded `User`→`card_id`,
   dropped `Card`, `Merchant City`, raw `Merchant State`, `Zip`, and never even
   *read* `Merchant City`/`Zip`. TabFormer fraud is dominated by **User/Card
   identity** (fraud clusters by user; same users span the temporal split), so
   dropping them cratered the baseline (0.76 vs NVIDIA's 0.99). **This is the
   core "botch."**
2. **Wrong categorical encoding.** We target-encoded (leak-safe but weak) and, in
   the standalone repro, accidentally forced `Merchant Name`/`Zip` to strings →
   ordinal-encoded instead of numeric passthrough. That tanked **AP** (0.059 vs
   0.147) while barely denting AUC. NVIDIA's cuDF infers them numeric.
3. **FM sequence had no identity.** It carried only behavioral fields; NVIDIA's
   tokenizer includes `CUST`/`CARD`. Fixed by adding `user`/`card` as **static
   (per-card, broadcast) fields**.
4. **`01_generate_data` skips if raw exists** → stale raw silently blocked the
   new `_normalize`. Must use a fresh base-dir or force-regenerate.
5. **Infra/OOM footguns (all fixed):**
   - `holdout_keep: 1.0` (~5M eval windows) OOM'd the downstream driver → 0.1.
   - Embed `gpus_per_worker: 0.5` packed 2 actors/A10G → CUDA OOM → 1.0.
   - seq-2048 pretrain `batch 32` OOM on 24GB A10G → 16.
   - `job_config` missing `working_dir` → job ran with no code uploaded.
   - Fraud/reco eval trained on windowed eval targets (7.8% train fraud), a
     different regime than NVIDIA's 1M-balanced-all-transactions.

## What we fixed / validated
- **Reproduced NVIDIA's baseline: ROC-AUC 0.9873 / AP 0.1469** on the raw CSV
  (`scripts/baseline_repro.py`), matching their 0.9885/0.1238 (blog) and
  0.9914/0.1424 (notebook) → **same data + recipe confirmed.**
- **Same numbers through our `_normalize`** (`--via-normalize`) → **our pipeline
  reproduces the benchmark**, not just the CSV.
- Added `user`/`card` static fields to the FM tokenizer/model; TB writes to
  `<base>/tensorboard`.
- Full CPU smoke (incl. InfoNCE + reco) passes end-to-end on the corrected code.

## Status / next
- Fresh base `transaction-fm-v2` (old base kept for comparison).
- Real seq-512 run submitted on v2 with the corrected pipeline — the first run
  whose fraud numbers are actually comparable to NVIDIA.
- Open question (the real one): does the FM add AP lift **on top of the legit
  ~0.987/0.147 baseline**, the way NVIDIA's +41.8% AP claims?
