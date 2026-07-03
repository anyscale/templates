# fintech_transaction_fm — changelog & status

## ⏸️ RESUME HERE — stopped 2026-07-02, cluster being terminated

**All code is committed to git (branch `zgarner_transaction_foundation_model`). The
`/mnt/cluster_storage/transaction-fm/*` artifacts (raw, tokenized, model, embeddings,
the TabFormer CSV) are on the ephemeral cluster and WILL BE GONE.** So on a fresh
cluster you start the data pipeline from scratch.

**State:** the full re-architecture to NVIDIA's design (flat tokenizer + Llama causal
decoder + next-token pretrain + last-token embed) is DONE and `mini`-smoke-validated.
The last thing running was the **first full pretrain (04)** — it works functionally
but was **too slow to finish**: the cluster could only get **T4 GPUs** (A10G/L4/L40S
all `LaunchFailed` — capacity), and 4096-token training on T4 is ~2 h/epoch ≈ 16 h.
Never produced a full model/embeddings/06 table for the new architecture.

**To resume (fresh cluster):**
1. Run notebooks **01 → 06** in order, `SCALE="full"`, fresh kernel each. (raw data must
   be regenerated — 02 re-downloads the TabFormer CSV + normalizes, ~15-20 min.)
2. **DECIDE the pretrain size vs the GPUs you get** (this was the open question):
   - If you get **A10G/A100/L40S**: keep `configs/full.yaml` as-is (seq_len 4096, epochs 8, batch 4). ~2-4 h.
   - If you're stuck on **T4s**: cut `full.yaml` `tokenize.seq_len` to **1024** and `pretrain.epochs` to **4**
     (≈8× less compute → ~1.5-2 h on T4). Lower fidelity (~78 vs 314 txns context) but a real result.
3. Memory levers already in the code (keep): bf16 autocast + `gradient_checkpointing_enable()`
   + `expandable_segments`. GPU-probed safe at batch 4 / seq 4096 / 15.6 GiB = 9.2 GiB peak.
4. Then read 06's fusion PR-AUC vs NVIDIA's **0.176** — the whole point.

**Gotchas (bit us this session):** restart the kernel after any `src/*.py` edit; notebooks
skip-cache on existing stage outputs (clear the relevant `full/` dir to force regen);
`scripts/run_pipeline.py --scale mini` is the fast headless smoke.

---

**Goal:** make the downstream fraud numbers **match then beat NVIDIA's transaction-FM
blueprint** on real IBM TabFormer, keeping the Ray pipeline clean. Metric = **PR-AUC
(Average Precision) at ~0.1% natural prevalence** (NVIDIA's choice; ROC saturates at
that imbalance).

NVIDIA blueprint targets:

| feature set | AUC-ROC | PR-AUC (AP) |
|---|---|---|
| raw (13 cols) | 0.9885 | 0.1238 |
| embedding-only (64-dim) | 0.8775 | 0.0123 |
| fusion / combined | 0.9925 | **0.1755** |

---

## WHERE WE ARE (2026-07-02)

The pipeline has been **fully re-architected to NVIDIA's blueprint** (flat tokenizer +
Llama causal decoder + next-token pretraining + last-token embedding), replacing the
original field-split / masked-feature design. Validated end-to-end by a `mini` smoke.

**The first full run of the new architecture is in progress.** raw/full was reused
(01/02 skipped); 03 tokenize is done (64,561 pretrain sequences, 5.2M eval samples);
04 pretrain (Llama, fp32, 8 epochs, 8 GPU) is running (~2–4 h), then 05 → 06.

Data is confirmed the **real IBM TabFormer** benchmark (24,386,900 txns, 6,139 cards,
0.12% fraud) — not the repo's synthetic generator.

## Results so far (full scale, real TabFormer)

| run | raw AUC/AP | fm AUC/AP | fusion AUC/AP |
|---|---|---|---|
| 1. original MLM, 4-feature raw | 0.785 / 0.0070 | 0.817 / 0.0088 | 0.854 / 0.0139 |
| 2. MLM + 14-feature raw + cosine | 0.944 / 0.0257 | 0.821 / 0.0071 | 0.967 / **0.0301** |
| 3. Llama causal decoder (NEW) | *(running — this is the real "match NVIDIA" test)* | | |

Best complete result so far: **fusion AP 0.0301 vs NVIDIA 0.176** — level-ish on
AUC-ROC, still ~6× below on AP. The fm-only gap (0.0071 vs 0.0123) said the
*representation* was the problem → hence the re-architecture (run 3).

---

## Change history

### A. Notebook 06 crash fix — distributed scoring
Eval pulled the 2.44M×512 test split onto the 30 GB driver → OOM. `evaluate()` now scores
via `map_batches` on the cluster, returns only thin (proba,label,weight). (`src/downstream.py`, nb 06)

### B. Raw baseline → NVIDIA's 13 columns (big lever)
The loader was destroying the strongest signals — `Use Chip` (online/in-person) collapsed to
a per-card mode; `Merchant City`/`Zip` dropped. Now carries all 13 per-transaction.
(`src/tabformer.py`, tokenizer `raw_*` passthrough, `src/downstream.py` RAW_FEATURE_COLS 4→14).
Lifted raw AP 0.007→0.026, fusion 0.014→0.030.

### C. Stronger XGBoost recipe
`xgb_params`: max_depth 5→8, eta 0.1→0.03, +min_child_weight 5, +reg_lambda 2.0;
num_boost_round 400→800, early_stopping 50. (in effect for run 3.)

### D. Full re-architecture to NVIDIA's blueprint  ← the big one
- **`src/flat_tokenizer.py`** (new) — 12 tokens/txn `[AMT MERCH CAT MCC HOUR DOW MONTH
  CARD CHIP ZIP3 STATE CUST]`, one shared vocab (**6,259**), `<bos>…<sep>…<eos>`, seq_len in
  TOKENS. Deterministic (fixed ranges + hashing), stateless `map_groups`. Keeps the `raw_*`
  passthrough for the downstream baseline. `src/tokenizer.py` re-exports it (import sites unchanged).
- **`src/model.py`** — replaced the field-split encoder with a **Llama causal decoder** built
  from `transformers` (`LlamaConfig`/`LlamaForCausalLM`), NVIDIA's exact config: hidden 512,
  8 layers, GQA 8/2, head_dim 64, SwiGLU 1408, RMSNorm, rope_theta 5e5. full = **29.0M params**.
  `sequence_embedding()` = last-token pooling.
- **`src/pretrain.py`** — causal-LM (next-token) loop; AdamW β(0.9,0.95)/wd 0.077 + warmup+cosine.
- **`src/embed.py`** — feeds `input_ids`, last-token pool.
- **configs** full/mini/small — Llama model blocks; token-based seq_len (full 4096); full epochs 8,
  batch 8. Deleted the obsolete `full_causal_nvidia.yaml`.
- **notebooks 03/04** — code + prose updated for the flat/causal design.
- **deps:** `transformers`+`accelerate` installed cluster-wide. No HF token needed (from-scratch init).
- **validated:** `mini` end-to-end smoke ran green (pretrain ppl 63→5.6; embeddings + 14 raw
  features reach downstream). One Ray `iter_torch_batches` dtypes bug found + fixed.

---

## How to run (full)

Fresh kernel per notebook. `raw/full` is intact with all fields, so **skip 01 and 02**.
Run **03 → 04 → 05 → 06** at `SCALE="full"`. Before a schema-changing rerun, clear
`/mnt/cluster_storage/transaction-fm/{tokenized,model,embeddings,downstream}/full` (notebooks
skip-cache on existing output); keep `raw/full` and `source/`.

Timing (fp32): 03 ~15 min · 04 ~2–4 h (8 GPU) · 05 ~20 min · 06 ~25 min.

---

## WHAT'S NEXT

1. **Read run 3's 06 table** — does the Llama embedding close the AP gap? fm-only vs NVIDIA's
   0.0123 is the clean representation test; fusion vs 0.176 is the headline.
2. **bf16 mixed precision** in `train_func` (`src/pretrain.py`) — currently fp32, ~2× slower than
   needed on A10G (NVIDIA used bf16+FSDP). ~halves pretrain time, frees memory to raise batch.
   Deferred (Zach OK'd); do before the next pretrain.
3. **If AP still trails after run 3:** tune pretrain (more steps/epochs, LR), then consider
   packing multiple cards per 4096 sequence (NVIDIA-style) for more/denser training signal.
4. **Cleanup:** residual stale prose in nb 03 intro cells (cosmetic); `ttest.yaml` still has the
   old seq_len/model block (dev-only smoke).

## Gotchas learned this session
- **Kernel cache:** editing `src/*.py` doesn't reach an open notebook kernel — always **restart
  the kernel** (or the notebook silently runs old code → stale/wrong results).
- **Notebooks skip-cache** on existing stage outputs (`if not os.path.exists(...)`) — clear the
  relevant `full` dirs to force regeneration.
- The headless `scripts/run_pipeline.py --scale mini` is the fastest way to smoke-test `src`
  changes end-to-end (fresh process, no kernel-cache issue).
