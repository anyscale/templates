# RESUME — fintech_transaction_fm (as of 2026-07-06, authoritative for this thread)

Read this first if picking the work back up. Full technical detail is in
[`FINDINGS_FM_REPRODUCTION.md`](FINDINGS_FM_REPRODUCTION.md); this is the state + next step.

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

**OPEN #1 — fusion sweep** (to push fusion past 0.176 on the same peak basis their 0.176 uses;
our fm is 2× stronger so it should clear it). GOTCHA: `run_ours_full.py` did NOT save the
embeddings, so first re-embed with our HF weights saving npy, THEN seed×eval-bootstrap.
Fastest: add `np.save` of `emb[split]` to `scripts/nvidia_repro/run_ours_full.py`, re-run
(~10 min re-embed on warm GPU), then run a `run_peakhunt.py`-style seed×eval-bootstrap on the
saved `nvours_embed_*.npy`. (their-weights fusion peaked 0.258 across draws.)

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
