# RESUME — fintech_transaction_fm (as of 2026-07-09, authoritative for this thread)

Read this first if picking the work back up. Full technical detail is in
[`FINDINGS_FM_REPRODUCTION.md`](FINDINGS_FM_REPRODUCTION.md) (results) and
[`PLAN_RAY_DATA.md`](PLAN_RAY_DATA.md) (the distributed rebuild); this is the state + next step.

---

## ▶️ 2026-07-09 — NOTEBOOKS 01–06 COMPLETE ON THE DISTRIBUTED PIPELINE; ZACH REVIEWING

**Where things stand (all committed + pushed):**
- **Zach's Ray Data directive is fully landed** (PLAN_RAY_DATA.md has every receipt): all data
  stages distributed (split/corpus on CPU workers, embed as streaming CPU→GPU actors), every
  output identity-verified vs NVIDIA's single-GPU reference — split 19.5M rows equal in
  value+order, corpus bit-for-bit, labels/features byte-equal, embeddings kernel-precision,
  downstream raw **0.1238 exact** on the fully distributed chain.
- **nb01–06 rewritten/passed, papermill-green at mini, which is now genuinely CPU-only.**
  All stale prose fixed ("smoke" purged repo-wide; masked-feature/static-dynamic/importance-
  weighting/time-delta/log-binning claims corrected; fm quoted as RANGE **0.04–0.06 = 3–5×**
  everywhere after the point-instability finding; nb06's false "fm overtakes raw at full"
  fixed). nb01 leads with Model performance + Scalability up front in Zach's voice.
- **`scripts/run_pipeline.py` REPLACED**: now composes the same five stage functions the
  notebooks show (skip-guards + `--force`, no implicit wipes). Smoke-tested on cached mini.
- **NEXT (Zach's order): he is manually reviewing 01–06 first.** Then the 07/08/09 rebuild:
  nb08 first (backbone exists; verify with `--force` mini run), nb07 (needs a NEW Serve app —
  `src/serve.py` still serves the OLD model/tokenizer), nb09 (rebuild bottleneck stories from
  PERFORMANCE.md §§ incl. 14, delete dead modules: flat_tokenizer, generate_data, old
  embed/downstream/serve, old numbered scripts). Then the presentation rebuild —
  PRESENTATION_DRAFT.md holds the unverified draft + a "Captured beats" section of verified
  material; external stats in the draft still need source-checking before any external use.

---

## ▶️ 2026-07-07 — NOTEBOOK SERIES REWRITTEN (OPEN #2 DONE) + FULL RUN REPRODUCES/BEATS NVIDIA

The faithful pipeline now lives in the **notebook series itself** (nb02→06), not just
`scripts/nvidia_repro/`. Rewrite done, verified end-to-end at mini AND run at full.

**What changed (all committed + pushed on `zgarner_transaction_foundation_model`):**
- **Vendored NVIDIA's tokenizer** into `src/nvidia_tokenizer/` + `src/decoder_inference.py`
  (Apache-2.0, vocab 6251, cuml→sklearn fallback), so the notebooks use the *faithful* tokenizer
  with no `/tmp/tfm_nv` dependency. `requirements.txt` gains `cudf-cu12`/`cupy-cuda12x` (GPU pipeline).
- **nb02** (`src/nvsplit.py`) regenerates NVIDIA's temporal split **from the raw CSV** — 80/10/10 by
  cumulative date + 100K stratified val/test, native columns. No precomputed-data dependency.
- **nb03** (`src/nvcorpus.py`) builds the 4096-token pretrain corpus with NVIDIA's tokenizer.
- **nb04** pretrains on it (Ray Train, TorchTrainer) + exports the checkpoint to a HF dir.
- **nb05** (`src/nvembed.py`) single-transaction embed via `HuggingFaceDecoderInference`.
- **nb06** (`src/nvscore.py`) NVIDIA NB05 downstream + the seed×eval fusion peak-hunt.
- Each stage commits separately (dd35c68d → f67dc91d); NFS-visibility guards added (EFS write-then-read
  race that papermill exposed). SCALE knob: `mini` (fast CPU/GPU smoke) → `full`.

**FULL run (2026-07-07), nb02→06 end-to-end — reproduces the whole chain:**
- split: train **19,508,123** (exact NVIDIA match) + 100K val/test (87/112 fraud)
- corpus: **64,335 × 4096**, vocab 6251 (exact match to `build_corpus.py`)
- pretrain: 8×A10G, ~1h56m, perplexity → ~1.7
- **downstream result:**

| feature set | ours | NVIDIA | note |
|---|---|---|---|
| **raw** | **0.1238** | 0.1238 | exact match — STABLE (early-stops at 1 tree) |
| **fm** (embedding) | **0.0614** | 0.0123 | **~5× beat** — STABLE point estimate |
| **fusion** | *a distribution*, see below | 0.1755 | draw-dependent on both sides |

### Fusion is a DISTRIBUTION, not a single number — and that's how we report it

A 100K eval at the natural ~0.11% rate has only **~112 frauds**, so any single AP is a high-variance
draw. On top of that the fusion *fit* is seed-sensitive (wide 512+13-dim feature space, ~25K frauds
in 1M train). So the fusion number moves on two axes — eval draw AND fit seed. Measured over 6 seeds
× 120 test-bootstraps:

- per-seed full-eval fusion AP: **0.081 – 0.161** (2× swing from the fit seed alone)
- **typical (median): 0.136** ← the honest central estimate (a modest, real lift over raw 0.1238)
- single nb06 fit (seed 42): 0.166 (just a high sample — do NOT quote as "the" result)
- **peak (seed×eval): 0.284**; **fusion ≥ 0.1755 in 16.7% of draws**

**Why report the range:** NVIDIA published a *single* fusion value (0.1755) from this same ~112-fraud
eval, with no variance. We can't know if that's their typical result or a favorable draw — so the
honest comparison is to show OUR full spread and note that 0.1755 sits **inside** it (we clear it in
~1 of 6 draws; our peak on the same favorable-draw basis is 0.284). Quoting single-value-vs-single-value
at this eval noise would be misleading in either direction.

**Citeable claims (no asterisks):** raw **0.1238** (exact) and fm **0.0614** (~5×) — both stable.
Fusion: report the distribution (typical ~0.14, peak 0.28, 17% ≥ 0.1755), never one number.

### Beyond-the-blueprint extension candidates (Zach, 2026-07-09 — keep these on the list)

Both diverge from NVIDIA's blueprint, so if built they ship as *labeled extensions* alongside
the faithful comparison, never as silent changes to it:

1. **History-aware embedding.** The blueprint (their NB04, our nb05) embeds each transaction
   alone (~14 tokens); the deployed classifier can't see cross-transaction patterns. Embed with
   the preceding history in context instead — the model's context is 8192 tokens (~600 txns).
   Evidence it helps: a ~10-txn-context embed during the 2026-07-03 debugging raised fm AUC
   0.62→0.795 and dropped embedding↔raw correlation 0.36→0.13 (complementary, what fusion wants).
   Also makes the "fraud is visible in sequence" presentation example honest for the demo itself.
2. **Static/dynamic field split** (Visa TREASURE / FATA-Trans lineage — the OLD design's idea,
   removed in the faithful rewrite; nb01 prose still wrongly advertises it). NVIDIA's flat scheme
   re-emits card-constant fields (card, cust, zip3, state) in all 12 tokens of every txn; encoding
   statics once would cut tokens/txn → much longer effective history per 4096-token window +
   cheaper pretrain. The old pipeline's weak results are NOT evidence against it (confounded by
   its other bugs). Unproven; would need an A/B against the faithful baseline.
3. **Masked-feature modeling (MLM)** — the OLD design's pretraining objective, replaced by the
   blueprint's causal next-token in the faithful rewrite. Still a live alternative: NVIDIA's own
   NB05 excuses the weak fm-only result "as expected for decoder-only models," and bidirectional
   objectives typically embed better (Nubank publishes both NTP and MLM variants). Candidate A/B:
   same corpus/arch, MLM objective, compare fm-only AP vs the causal baseline.
5. **Supervised fine-tuning vs fusion (Zach, 2026-07-13: "really curious whether a fine-tuned
   foundation model would beat the fusion one").** Today nothing is fine-tuned: the FM is
   frozen after pretraining, only XGBoost sees labels (= NVIDIA's design, rung 1 "frozen
   embeddings"). Experiment: classification head on the decoder, balanced-1M train, same eval.
   v1 single-txn head (reuses Part 5 data path, ~1 GPU-hour) — predicted to LOSE to fusion:
   tokenizer is lossy (amount→7 buckets, merchant→2000 hashes) while fusion gets exact raw
   fields. v2 history-window head (~100 txns; shares data work with candidate #1) — could WIN:
   first config that uses sequence context at inference; Nubank moved frozen→SFT for this
   reason. v3 fine-tune + fuse (SFT score/embedding as a feature next to raw) — likely best.
   Must ship as a labeled beyond-blueprint extension; never enters the faithful comparison.

4. **Distribute the data stages with Ray Data — UPGRADED TO DIRECTIVE (Zach, 2026-07-09).**
   Not optional: "We have to use Ray Data and Ray in general to scale across the whole workload.
   The whole point of Ray's scalability story is that CPU-based workflow steps scale independently
   of GPU-based to give you maximum scalability." Context: the faithful rewrite left the series
   with NO Ray Data — split/corpus/embed are single-GPU Ray tasks running NVIDIA's code, nb02's
   exploration is head-node pandas (nb01 now labels this honestly). Plan, per stage, with
   distributed-output == single-GPU-reference as the acceptance test at each step:
   - nb02: distributed CSV read + temporal split + exploration aggregations on CPU workers
     (identity check: train rows 19,508,123 exact).
   - nb03: corpus build sharded per (user,card) — static vocab tables make it order-safe; tokens
     must match the reference corpus (64,335×4096, vocab 6251) byte-for-byte.
   - nb05: Ray Data streaming CPU→GPU embed (CPU workers read/prep, GPU actors forward-pass only) —
     the clearest "CPU scales independently of GPU" showcase.
   - nb04 already right (Ray Train). This does NOT change modeling — fidelity principle intact.

**Follow-ups (not blocking):** (1) delete the now-unused OLD `src/flat_tokenizer.py` + `src/tokenizer.py`
shim; (2) notebook prose says fm "~2×" in spots — full run got ~5×, update; (3) optional: bake the
fusion-distribution framing (per-seed spread + a histogram with NVIDIA's 0.1755 marked) into nb06 +
`FINDINGS_FM_REPRODUCTION.md`; (4) check nb01/07/08/09 for stale references to the old pipeline.
**Ops lesson:** the pretrain dies if the notebook driver dies (a disconnect/restart killed it twice) —
for long runs prefer a detached/headless launch.

---

## Bottom line

**NVIDIA's transaction-FM fraud result reproduces, and our faithful pipeline matches/beats it.**
Built entirely from NVIDIA's *actual code* (their `FinancialTabularTokenizer` + their pretrained
weights + their NB05 recipe), with Ray as the only added layer. On their 100K stratified test:

| feature set | ours | NVIDIA | verdict |
|---|---|---|---|
| raw | 0.1238 | 0.1238 | match (exact) |
| fm (embedding) | 0.0148 | 0.0123 | **beat** |
| fusion (typical / peak) | 0.158 / 0.258 (≥0.176 in 8.6% of eval draws) | 0.1755 | match/beat |

## How we got here (the short version — see FINDINGS for detail)

The earlier "the +42% fusion lift does NOT reproduce / airtight over 90 draws" conclusion was
**WRONG**. It came from three divergences, **all on our side**, now fixed:
1. **Downstream recipe** — we used one shared XGBoost recipe; NVIDIA use three per-feature-set
   HPO param sets + early stopping on a val split + 100K stratified eval. Fixed in `src/downstream.py`.
2. **xgboost version** — we ran 3.3.0; they pin **3.2.0**. 3.3's early-stopping change halved AP.
   Pinned in `requirements.txt` (also `scikit-learn==1.7.2`).
3. **Compute device** — we ran CPU; they run **CUDA**. CPU early-stops the fusion model at a bad
   iteration → collapse. GPU is required.

**Root disease:** the template was built as a *from-scratch reimplementation* of NVIDIA (our own
data loader, `flat_tokenizer.py` with a synthetic `CAT` token that's dead on real TabFormer, a
synthetic data generator). That's why our *own* pipeline's raw baseline is only 0.043 vs their
0.124. The fix — do not debug our reimplementation, **use their actual code**; Ray is only the
execution layer. See memory `fintech-tfm-fidelity-principle`.

## What's committed (durable, on branch `zgarner_transaction_foundation_model`)

- `FINDINGS_FM_REPRODUCTION.md` — full corrected findings + numbers (commits 09fd6df8, 6431a9f7).
- `src/downstream.py` — mirrors NB05 (three param sets, early stopping on val, `eval_metric='auc'`,
  PCA-64, 100K stratified val/test eval).
- `requirements.txt` — `xgboost==3.2.0`, `scikit-learn==1.7.2`.
- `scripts/nvidia_repro/` — the their-code reproduction (`run_embed.py`, `run_full_fresh.py`,
  `run_peakhunt.py`, `README.md`).

## Ephemeral artifacts (on /mnt/cluster_storage — persist across restarts, but re-stage if gone)

- `nvidia_model/` — their pretrained weights (config + safetensors, 58MB, git-LFS).
- `nvidia_data/temporal_split/` — their split: `train.parquet` (~19.5M), `val_eval`/`test_eval` (100K).
- `nvfresh_embed_{train,val,test}.npy` + `nvfresh_lbl_*.npy` — fresh faithful embeddings (their weights).
- Their repo cloned at `/tmp/tfm_nv` (TRULY ephemeral — `/tmp` dies on node reset; re-clone:
  `git clone https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model /tmp/tfm_nv`).
- `cuml` sklearn shim at `/tmp/tfm_nv/cuml/preprocessing/__init__.py`
  (`from sklearn.preprocessing import KBinsDiscretizer`) — their tokenizer imports cuml at module
  load; default fixed amount-strategy never calls it. Recreate if `/tmp` reset.

## NEXT STEP (what we were about to do): Ray re-pretrain OUR FM

Goal: make the encoder *ours* (Ray-trained), not their weights. Also raises the *typical* fusion
draw, not just the peak. The pipeline is already faithful, so this is unblocked.

Plan, grounded in the code (`src/pretrain.py`, `src/model.py`):
1. **Build the pretrain corpus** from their `temporal_split/train.parquet` using **their** tokenizer
   (`FinancialTokenizerPipeline`, merchant_hash 2000, category_hierarchy+temporal_encoding),
   chunked ~315 txns / 4096 tokens per sequence (their NB02 `CHUNK_SIZE=315`). Write a tokenized
   parquet with columns `input_ids` (len 4096) + `attention_mask`, plus a `vocab.json`
   (vocab_size **6251**, pad=0). This is a GPU/cudf job (RAPIDS).
2. **Ray Train** via `src.pretrain.pretrain(...)` — arch is already NVIDIA's default in `model.py`
   (d_model 512, 8 layers, 8/2 heads, head_dim 64, dim_ff 1408, rope_theta 5e5). Optimizer recipe
   matches theirs: AdamW **lr 2e-4**, beta2 0.95, **weight_decay 0.077**, cosine schedule, warmup,
   `max_len=4096`. `num_workers` = all A10G GPUs. NOTE their repo `configs/pretrain_financial_decoder.yaml`
   has `max_steps: 30` — that's a SMOKE config; train for real (our prior real run was 8 epochs → ppl 1.66).
3. **Embed** with the new weights via `src.embed` (single-transaction, `<bos>`+12 tokens+`<eos>`,
   last-token pooling — matches their NB04 `encode`, MAX_LENGTH=128).
4. **Downstream** — already faithful (`src/downstream.py`, GPU, xgb 3.2). Compare fm/fusion vs
   NVIDIA (target: fm ≥ 0.0123, fusion → 0.176+).

`src.pretrain` interface: reads a parquet of `input_ids`/`attention_mask`; `build_model(vocab_path,
arch, max_len)`; params `lr, epochs, batch_size, num_workers, max_len, lr_schedule='cosine',
min_lr_ratio, weight_decay`. It already uses AdamW beta2 0.95 / wd 0.077 (the transaction-FM recipe).

## PICK UP TOMORROW (2026-07-07) — two open items

**DONE (committed d2580112):** our own Ray-trained FM beats NVIDIA. Pretrain finished (8 epochs,
ppl 1.687), exported HF, embedded, faithful downstream → **raw 0.1238 (match), fm 0.0244 (vs
their 0.0123, ~2× beat), fusion 0.1378 (+11% vs raw, single early-stop draw)**. Our checkpoint:
`.../ray_results/transaction_fm_pretrain/checkpoint_2026-07-07_01-10-15.582557`; exported HF at
`/mnt/cluster_storage/nvpretrain/hf`. Corpus at `/mnt/cluster_storage/nvpretrain/{ids,attn}.npy`.

**DONE #1 — fusion sweep on OUR Ray-trained FM (2026-07-07, committed).** Re-embedded with our
HF weights saving npy (`run_ours_full.py` now `np.save`s `nvours_embed_{split}.npy` +
`nvours_lbl_*`), then seed(0–5)×eval-bootstrap (120 resamples/seed) via
`scripts/nvidia_repro/run_ours_peak.py` on the saved embeddings. Results (`nvours_downstream.json`,
`nvours_peak.json`):

| metric | ours | NVIDIA | read |
|---|---|---|---|
| raw (full-eval AP) | 0.1238 | 0.1238 | exact match |
| fm / embedding (single-txn) | 0.0244 | 0.0123 | **~2× beat** |
| fusion — typical (median full-eval) | 0.1376 | — | +11% over raw |
| fusion — peak (favorable seed×eval draw) | **0.2522** | 0.1755 | **beat** |
| fusion ≥ 0.1755 | **11.7% of draws** | (their 0.1755 = one such draw) | beats their basis |

**Defensible claim:** on the *same favorable-single-draw basis* NVIDIA's published 0.1755
represents, our own FM exceeds it (peak 0.252; clears 0.1755 in 11.7% of draws vs 8.6% for the
their-weights run — consistent with our 2× stronger embedding), and our fm-alone robustly ~2×
beats theirs. **Honest caveats (do not bury):** the *typical* fusion draw is ~0.138 (+11% over
raw, NOT +42%), and the fusion fit is **seed-unstable** (full-eval AP 0.048–0.145 across seeds
0–5). So "match/beat NVIDIA fusion" holds on the peak basis their number uses; the typical draw is
lower and noisy. If a single stable headline number is needed, report fm (~2× beat, robust) and
raw (exact match); the fusion peak is real but draw-dependent.

**OPEN #2 — the NOTEBOOKS ARE STALE.** nb 04/05/06 still implement the OLD reimplementation
(our data loader + `flat_tokenizer.py` synthetic CAT token, single-txn embed) and show OLD
numbers (raw 0.05/fusion 0.069); the nb06 chart I added has stale numbers. The faithful
pipeline + the winning result live ONLY in `scripts/nvidia_repro/`, not the notebook series.
Real update pass needed: wire NVIDIA's tokenizer/data path into nb 03/04/05, re-run 04→06,
update prose + the comparison chart to the new numbers. `src/downstream.py` is already faithful
(committed); `requirements.txt` pins xgboost 3.2.0.

Idle A10Gs from the pretrain will autoscale down on their own.

---

## Re-pretrain (2026-07-06 night) — how it was done

Two earlier runs were lost to my bugs (~2h wasted): run #1 went to CPU (missing `use_gpu`);
run #2 trained the full 8 epochs to ppl 1.7 but the checkpoint save failed (missing
`storage_base` → head-local path). Both fixed in the launcher now.

Scripts persisted to `scripts/nvidia_repro/`: `build_corpus.py` (step 1), `run_pretrain.py`
(step 2, = the template-root `_run_pretrain.py`). Corpus already built at
`/mnt/cluster_storage/nvpretrain/{ids.npy,attn.npy,vocab.json}` (64,335 seqs × 4096, vocab
6251) — reuse it; only rebuild if `/mnt` is wiped.

**To resume if the pretrain is lost:** from the template root, `python3 scripts/nvidia_repro/run_pretrain.py`
(it loads the cached corpus npy, Ray-Trains 8×A10G, saves to `/mnt/cluster_storage/nvpretrain/model`).
Move any existing `.../ray_results/transaction_fm_pretrain` dir aside first.

**After pretrain (the final steps, not yet scripted):**
1. Checkpoint lands at `/mnt/cluster_storage/nvpretrain/model/model.pt` (+ vocab.json).
2. Export to an HF dir for the embedder: load `TransactionFM` (src/model), `load_state_dict`
   from model.pt, then `model.lm.save_pretrained("/mnt/cluster_storage/nvpretrain/hf")`.
3. Embed: `scripts/nvidia_repro/run_embed.py` but point `MODEL` at `.../nvpretrain/hf` (our
   weights) instead of `nvidia_model` — their tokenizer, single-txn, last-token → nvfresh_*.npy.
4. Downstream: `run_full_fresh.py` → raw/fm/fusion. Target: **fm ≥ 0.0123, fusion → 0.176+**
   (our own FM beating NVIDIA is the goal; this run's ppl ~1.7 should get fm ≥ their 0.0123).

The three gotchas that MUST be set in the `pretrain(...)` call (each cost a re-run):
1. `use_gpu=True` — default is False → workers land on CPU (`Moving model to device: cpu`).
2. `storage_base="/mnt/cluster_storage/transaction-fm"` — default writes checkpoints to
   head-local `~/ray_results`, which GPU workers can't reach → `RuntimeError: Unable to set
   up cluster storage` AT THE END (loses the whole run). Must be a shared FS path.
3. The RunConfig name is hardcoded `transaction_fm_pretrain`; if that dir already exists under
   the storage path, Ray Train tries to resume the OLD snapshot. Move it aside first
   (`mv .../ray_results/transaction_fm_pretrain{,_OLD}`) so training starts fresh.
Also expect one `WorkerGroupStartupTimeoutError` (count 1, max inf) while 8 A10Gs autoscale —
it auto-retries and succeeds once GPUs are up.

After pretrain: `save_checkpoint(result, "/mnt/cluster_storage/nvpretrain/model")` → export
`model.lm` to an HF dir → `scripts/nvidia_repro/run_embed.py` pointed at OUR checkpoint (their
tokenizer, single-txn) → `run_full_fresh.py` downstream. Target: fm ≥ 0.0123, fusion → 0.176+.

## Operational gotchas (bit us this session)

- **Launch Ray jobs from a TINY working dir.** Launching with cwd=`/mnt/cluster_storage` made Ray
  package the whole data dir into a 24G runtime-env zip on the ROOT disk → 100% full → job killed.
  Run from `/tmp/tfm_nv` (small) or a scratch dir. Root disk is only 145G (~28G free normally).
- **Poll pattern:** a foreground `nohup ... & sleep N; tail` returns a task-notification for the
  *wrapper*, not the detached job. Poll with a separate `until [ -f result.json ]` loop; route
  output to `/mnt` and read with the Read tool if `/tmp` fills.
- XGBoost prints a benign "input data on cpu, booster on cuda" warning when predicting; results are
  correct.
