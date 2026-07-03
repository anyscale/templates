# fintech_transaction_fm — changelog & status

## ▶️ RESUME HERE — 2026-07-03 (evening), FIRST FULL RUN DONE + two bugs found

The Llama-arch full run **finished** and came back WORSE than run 2, exposing two
independent bugs (raw/XGBoost and the TFM representation). Results:

| feature set | our AP | NVIDIA AP | our AUC | NVIDIA AUC |
|---|---|---|---|---|
| raw    | 0.017  | 0.124  | 0.879 | 0.989 |
| fm     | 0.0023 | 0.0123 | 0.634 | 0.878 |
| fusion | 0.021  | 0.176  | 0.878 | 0.993 |

### BUG 1 — XGBoost/raw training set starved to 1.5% (FIXED in code, re-run pending)
Full run's downstream splits: **train=321,077 · val=2,438,693 · test=2,438,690** — the
*training* split is smaller than either eval split and only ~1.5% of the ~19.5M
train-period txns. NVIDIA trains raw XGBoost on the **full** training set.
Root cause: `eval_normal_keep()` derived a normal-keep of **0.0152** from
`target_eval_samples: 400000` and the tokenizer applied it to the **train period**
(`flat_tokenizer.py:253`). That knob was meant to size the *eval* set, but
`holdout_keep: 1.0` overrides eval sizing — so it silently only throttled the
*training* set. (296K normals × 0.0152 + all frauds = 321,077 ✓.)
**Fix:** new `train_keep` config knob (full=1.0 → train on the whole training set,
NVIDIA-parity; also the honest "distributed XGBoost scales" story). `holdout_keep`
still governs val/test (1.0 = exact metrics); `target_eval_samples` now only bites
when `holdout_keep` is null (mini/small CI). Touched: `scale_config.py` REQUIRED_KEYS,
`02_tokenize.py`, `run_pipeline.py`, all `configs/*.yaml`. Validated at mini
(`train_keep=0.5 holdout_keep=0.51`, green).

### BUG 2 — TFM representation weak (NOT yet fixed)
fm-only AP 0.0023 is WORSE than the old MLM's 0.0071 and far below NVIDIA's 0.0123;
fm AUC 0.634 is near-random. The Llama last-token embedding isn't discriminative.
Suspects to investigate after raw is fixed: pretrain under-converged, last-token
pooling vs mean-pooling, embedding not carrying fraud signal. Part of this is the
SAME starvation (fm downstream also trained on 321K) — re-measure after Bug 1.

### Plan / next steps (be careful: do NOTHING heavy on the HEAD node — it OOM-crashed
once on CPU work. All tokenize/embed/XGBoost must run distributed on the A10G workers.)
1. Re-tokenize `full` with `train_keep=1.0` (distributed). Clear `tokenized/full`
   first (Ray Data write_parquet would mix old+new files). **KEEP `model/full`** —
   pretrain windows are deterministic & independent of `train_keep`, so the existing
   pretrained Llama is still valid; skip the 2-4h pretrain.
2. FAST raw signal: distributed raw-only XGBoost probe straight off `tokenized/full/eval`
   (no embeddings needed — raw_* cols are in that parquet). Confirm raw AP → ~0.12.
   MUST use `XGBoostTrainer` on workers, NOT in-process pandas on the driver.
3. Then re-embed (24.4M windows now, ~2h on workers) → downstream fm+fusion.

Note: `train_keep=1.0` grows the eval/embed set from ~5M to ~24.4M windows (~33G
tokenized, embed ~2h). That's the real full run; heavy stages are referred out as Jobs.

---

## (prior) 2026-07-03, first full run of the new architecture LAUNCHED

**GPU problem solved.** Last session was stuck on T4s (A10G/L4/L40S all `LaunchFailed`
on capacity; 4096-token training on T4 ≈ 16 h). After a workspace reboot, **A10G (23.7 GB)
launches cleanly**, autoscaling to 8 for pretrain. The uncommitted `job_config.yaml` fix
(g4dn/T4 → g5/A10G) + `requirements.txt` (+transformers/accelerate) are now committed.

**`/mnt/cluster_storage` PERSISTS across cluster restarts** — it is workspace-scoped, NOT
ephemeral as an earlier note feared. So the expensive prep survived the reboot: `source/`
(2.2 GB TabFormer CSV), `raw/full`, and `tokenized/full` (6.7 GB — 64,561 pretrain seqs,
~5M eval) are all intact. **On resume you skip nb 01/02/03** and run only the
pretrain → embed → downstream tail (the three artifacts the T4s never produced).

**State (2026-07-03 ~11:55):** the full-scale tail is RUNNING headless on 8× A10G via the
chained stage scripts — `scripts/03_pretrain.py → 04_extract_embeddings.py →
05_train_downstream.py --scale full` (see scratchpad `run_full_tail.sh`; logs to
`full_tail.log`). Config unchanged: `configs/full.yaml` seq_len 4096, 8 epochs, batch 4/worker
(global 32), bf16. ~2-4 h pretrain + ~20 min embed + ~25 min downstream. This is the real
"did the Llama re-architecture close the gap to NVIDIA" test.
- One benign hiccup at startup: Ray Train's 60 s worker-group-startup timeout fired once while
  the 7 extra A10G nodes were still autoscaling; it auto-rescheduled and reached RUNNING on the
  retry. If it ever loops, set `RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S=300`.

**When it finishes:** read 06's fusion PR-AUC vs NVIDIA's **0.1755** (old-arch best was 0.0301).
The stage scripts read the persisted disk artifacts and DON'T wipe — only `run_pipeline.py` wipes.

**Memory levers in the code (keep):** bf16 autocast (`src/pretrain.py`),
`gradient_checkpointing_enable()` (`src/model.py`), `expandable_segments`.

**Gotchas:** restart the kernel after any `src/*.py` edit (notebooks cache old code); notebooks
skip-cache on existing stage outputs (clear the relevant `full/` dir to force regen);
`scripts/run_pipeline.py --scale mini` is the fast headless smoke (but it WIPES all stage
outputs — don't run it against the persisted `full/` data).

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
| 3. Llama causal decoder (NEW) | *(running 2026-07-03 on 8× A10G — the real "match NVIDIA" test)* | | |

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
2. **bf16 mixed precision** — DONE. `train_func` (`src/pretrain.py`) wraps the forward in
   `torch.autocast(bfloat16)`; no GradScaler needed. Live for run 3.
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
