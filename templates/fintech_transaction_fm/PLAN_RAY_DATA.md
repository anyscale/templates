# Plan: distribute the data stages with Ray Data

**Directive (Zach, 2026-07-09):** the series must use Ray Data / Ray to scale across the whole
workload — CPU-based steps scaling independently of GPU-based steps. Wrapping NVIDIA's
single-GPU code in one Ray task per stage (the current state) is not that.

**Invariant (fidelity principle):** the modeling and its outputs do not change. Every stage
carries an acceptance test: distributed output == the single-GPU reference that produced the
current results table (raw 0.1238 / fm 0.0614 / fusion distribution). NVIDIA's vendored code
stays verbatim; anything we add sits beside it, never inside it.

## Facts the plan is built on (verified 2026-07-09)

- `nvsplit/nvcorpus/nvembed` are each ONE `@ray.remote(num_gpus=1)` task; nb02's exploration
  is head-node pandas. There is no Ray Data anywhere in the core path.
- cudf imports on CPU nodes but its operations **fail without a GPU** → today even `mini`
  needs a GPU worker per data stage, the notebooks' "mini = CPU-only" prose is wrong, and
  CPU-only CI cannot run the pipeline. A CPU tokenize path fixes mini + CI as a side effect.
- The tokenizer vocab is **structurally static** (fixed ranges/tables/hash buckets — vocab
  6251 regardless of data), so per-card sharding cannot change tokens.
- In the corpus build, the expensive step (per-line `tok.encode` loop) is already pure-Python
  CPU; only preprocess/transform (field derivation → per-field token strings) touches cudf.
- The two seeded, order-sensitive operations — the 100K stratified eval sample (nvsplit) and
  the 1M balanced train sample (nvembed) — must keep byte-identical row selections. Both act
  on small-enough data to run exactly as today, on one node, inside the distributed flow.
- Merchant hashing is cudf `hash_values` (murmur3). Reproducible on CPU via `mmh3`, but
  byte-identity on the real merchant strings is **the** risk item → verify first.

## Stages (each lands separately: code + notebook + identity check, full scale, then commit)

### Stage 0 — STATUS 2026-07-09: IN PROGRESS (resume notes below, written mid-build in case of session loss)

**Files being built:**
- `src/nvtokenize_cpu.py` — pandas mirror of `FinancialTokenizerPipeline.preprocess()` +
  `transform()` (vendored files untouched). Merchant hash via `mmh3` (murmur3_x86_32, seed 0,
  unsigned → matches cudf `Series.hash_values()` — TO BE VERIFIED, that's the point of Stage 0).
  `pip install mmh3` needed (add to requirements only if adopted).
- `scripts/verify_cpu_tokenizer.py` — two subcommands:
  - `dump`: ONE `@ray.remote(num_gpus=1)` task (run from template root so `working_dir` ships
    `src/`); reads `/mnt/cluster_storage/transaction-fm/nvsplit/full/train.parquet`; writes to
    `/mnt/cluster_storage/transaction-fm/stage0/`: `merchant_hash_gpu.parquet` (ALL distinct
    raw merchant names + cudf-cleaned string + cudf hash), `tokens_gpu.parquet` (first 200K
    rows tagged `__row_id__` BEFORE preprocess → preprocess → fit → transform; saves
    `__row_id__,user,card,time_full` + the 12 token columns), `vocab_gpu.json` (full
    token→id vocab from `FinancialTabularTokenizer`, must be 6251).
  - `compare`: local/CPU; pandas-reads the same train.parquet, `.head(200_000)` (same rows —
    parquet row order is deterministic), runs `nvtokenize_cpu`, checks: (1) cleaned merchant
    strings equal (catches cudf-vs-python regex `[^A-Z0-9\s\-]` divergence), (2) mmh3 == cudf
    hashes as uint32, (3) `__row_id__` sequence after the (user,card,time_full) sort equal —
    catches sort tie-break divergence (Time is minute-resolution → same-minute txns tie; CPU
    uses stable mergesort; if cudf differs ONLY in tie order, also compare after sorting both
    by (user,card,time_full,__row_id__) and report both results), (4) all 12 token columns
    equal row-by-row, (5) CPU-built `FinancialTabularTokenizer` vocab == `vocab_gpu.json`.

**Exact semantics already extracted from the vendored code (re-derive from these if lost):**
- `preprocess`: lowercase/underscore col names; amt = `$`-strip → float → `amt_val` = sum of
  ≥{10,50,100,500,1000,5000} (int32); merch_clean = upper + regex-remove `[^A-Z0-9\s\-]` →
  `merch_hash = hash_values()`; mcc `fillna(-1).astype(int)` → `mcc_int` + `mcc_str=str`;
  datetime from `Y-M(zfill2)-D(zfill2) Time(fillna "00:00")` fmt `%Y-%m-%d %H:%M` → hour/dow
  (dayofweek)/month; card int clip 0-9; chip_upper = upper(use_chip); zip = fillna("00000") →
  str → replace(".0","",regex=False) → first 3 chars zfill(3) → int; state = fillna("XX") upper
  strip, ""→"XX"; cust = user int clip 0-2999; `time_full` = the datetime; **global sort by
  (user, card, time_full)**; time_delta_s = per-(user,card) diff seconds fillna(0) clip≥0.
- `transform` output = one column per step keyed by step id, order: `amt_val, merch_hash,
  mcc_int, mcc_str, hour, dow, month, card, chip_upper, zip3, state_clean, cust`.
- Token formats: FixedVocab = `f"{prefix}_{v:0{pad}d}"` after int32 clip to [min,max] — pads:
  hour 2, month 2, zip3 3, all others 0 (AMT 0-6, DOW 0-6, CARD 0-9, CUST 0-2999);
  merch = `f"MERCH_{hash % 2000}"`; ranges (CAT) = numpy range masks over INDUSTRY_RANGES,
  default `CAT_GENERAL`; direct maps = `host.map(dict).fillna(default)` then `PREFIX_ + val`
  (MCC passthrough of KNOWN_MCCS strings, default "-1"; CHIP via CHIP_MAPPING, default "UNK";
  STATE passthrough ALL_STATES, default "XX").
- `FinancialTabularTokenizer` vocab build is data-free pure Python → `encode` already works on
  CPU. Specials: pad 0, bos 1, eos 2, sep 3, unk 4. vocab_size must equal 6251.

**Known risks being tested:** murmur3 exactness; sort-tie stability; cudf regex semantics;
zip float-string ".0" handling (replace removes ALL occurrences in both libs).

**To resume after a crash:** finish/re-create the two files per above, `python
scripts/verify_cpu_tokenizer.py dump` (waits for a GPU worker), then `... compare`. Paste the
compare report into this file, then proceed to Stage 1.

### Stage 0 — proof of the risky bits (no notebook changes)
Scratch script, GPU worker + CPU worker:
1. cudf `hash_values` vs CPU `mmh3` on all distinct `Merchant Name` values → must match 100%.
2. CPU (pandas) reimplementation of `FinancialTokenizerPipeline.preprocess+transform` on a
   ~100K-row sample vs the cudf output → token strings identical.
3. Assert vocab_size 6251 from the CPU path.
Deliverable: `src/nvtokenize_cpu.py` (new module beside the vendored package — vendored files
untouched) + a comparison report in this file. **If murmur3 identity fails, stop and rethink**
(fallback: tokenize-to-strings stays on GPU workers, sharded; everything else still proceeds).

### Stage 1 — nb02: split + exploration on Ray Data (CPU workers)
- `ray.data.read_csv` → `map_batches` (strip cols, derive date) on CPU workers.
- Cutoff dates: Ray Data `groupby("date").count()` (≈6K rows) → driver cumsum → same two dates.
- Train filter + write via Ray Data (order-preserving); val/test subsets (~2.4M rows) collected
  to one task that runs today's exact sklearn stratified sample (seed 42) → identical eval sets.
- Exploration cells: Ray Data aggregations instead of 19.5M-row head-node pandas.
- Identity: train rows 19,508,123; cutoff dates equal; val/test parquets byte-equal to reference.
- Prose: fix the stale "masked-feature modeling" sentence while in there; SCALE default → mini.

### Stage 2 — nb03: corpus build sharded per card (CPU workers)
- Ray Data over the train split → `groupby(user, card)` → `map_groups`: CPU tokenize-to-strings
  (`nvtokenize_cpu`) → chunk 315 → join `<bos>/<sep>/<eos>` → `tok.encode` (already CPU).
- Assemble `ids.npy`/`attn.npy` in deterministic (user, card, chunk) order.
- Identity: shape (64,335 × 4096), vocab 6251, and the sorted per-row hash multiset equals the
  reference corpus. (Row *order* may legally differ from the old build; pretrain shuffles —
  but sort deterministically anyway so re-runs are stable.)
- Measure for PERFORMANCE.md: wall-clock 1 GPU (old) vs N CPU workers (new), scaling curve.

### Stage 3 — nb05: streaming CPU→GPU embed (the heterogeneous showcase)
- Balanced train sample: exact seeded selection as today, on the driver (small index op).
- Ray Data: CPU workers tokenize per-transaction (`nvtokenize_cpu`) → `map_batches` with a GPU
  actor class (`num_gpus=1`, `concurrency=N`) doing only the forward pass → carry `__row_id__`
  → restore deterministic order → `embed_/lbl_/raw_` artifacts as today.
- Identity: labels/raw byte-equal; embeddings `allclose` vs reference (same per-row math — each
  row padded to max_length 128 independent of batching); nb06 full re-run reproduces the
  results table (raw 0.1238 exact).
- This stage is the literal "CPU steps scale independently of GPU steps": more CPU workers →
  GPUs never starve; more GPUs → linear embed throughput. Measure both for PERFORMANCE.md.

### Stage 4 — nb06 full re-run + results freeze
Re-run downstream at full on the Stage-3 artifacts; assert the nb01 table (raw exact, fm ~0.06,
fusion distribution consistent). Update FINDINGS/RESUME with the re-run.

### Stage 5 — truth-sync the frame
- nb01: Scalability section + architecture diagram + series table upgraded to the now-true
  claims ("[Ray Data] tokenize/embed across CPU workers, GPU actors for forward passes");
  subtitle back to "Ray Data + Ray Train + Ray Serve"; fix the currently-wrong "mini = CPU"
  wording (true after Stage 2).
- mini configs: verify whole series runs CPU-only end-to-end under papermill (this becomes the
  CI story for pointing tests at the series later).
- Purge remaining "smoke" phrasing (nb02/03/04/06, mini.yaml) as each notebook is touched.

## Decisions taken (flag if you disagree)

1. **CPU tokenize path is a NEW module, not an edit to vendored files** — keeps "their code
   verbatim" narratively clean; identity tests make the CPU path trustworthy.
2. **Seeded samples stay single-node** inside the distributed flow — exactness beats elegance
   at 100K/1M-row scale; the 19.5M-row work is what gets distributed.
3. **GPU cudf path remains available** (it IS NVIDIA's code); the notebooks default to the
   Ray Data path and mention the equivalence check.

## Cost / effort

Stage 0 ~hours (one GPU worker briefly). Stages 1–2 ~a day each including full-scale identity
runs. Stage 3 ~a day (embed re-run ~1.2M rows on a few A10Gs). Stage 4 one downstream run
(GPU, minutes–hour). Stage 5 prose. GPU spend: roughly one embed re-run + one downstream
re-run + Stage-0/2 verification tasks; no re-pretrain (corpus is identical by construction).

## What this buys the presentation

- "We distributed NVIDIA's data pipeline without changing a single output token" — currently
  unclaimable, becomes demonstrable with the identity checks as receipts.
- Real before/after numbers: corpus build 1 GPU vs N CPU workers; embed throughput vs GPU
  count; CPU and GPU pools autoscaling independently on one screen.
- Kills the false "mini is CPU-only" claim and makes it true.
