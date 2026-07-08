# Integrating zgarner_transaction_foundation_model — keep / migrate / ignore

Written 2026-07-08, after both branches concluded. His branch = a faithful
replication of NVIDIA's blueprint on Ray (their tokenizer vendored, their
split regenerated from CSV, their Llama decoder trained with their recipe,
their NB04 single-txn embedding, their NB05 downstream). Status per his
FINDINGS_FM_REPRODUCTION.md: **resolved — their numbers reproduce**, and his
own Ray-trained same-arch model beats their published embedding ~5x on
fm-only (0.0614 vs 0.0123). Ours = a different tokenizer (1 position/txn),
objective (per-field MLM + merchant InfoNCE), and readout (history-window
pooled-last) that beats their published FUSION embedding-alone
(0.258 @ 512 / 0.273 @ 1024 vs their 0.1755).

The two branches answer complementary questions. Nothing in his findings
invalidates ours; two of his findings HARDEN ours cheaply.

---

## 1. Does his work invalidate ours? NO — here's the reasoning

- His three "why it didn't reproduce at first" causes were all divergences
  from NVIDIA's env, and we only share one of them (see §2): we already use
  their three separately-HPO'd XGBoost param sets and their early stopping
  (src/nvidia_baseline.py transcribes them; his cause #1 doesn't apply).
- Our headline table is **internally paired**: every row (baseline, embed_*)
  is trained/evaluated on identical rows in an identical environment, so
  within-table lifts (+92% AP etc.) are valid regardless of xgboost version
  or device. Env skew only affects comparisons against NVIDIA's *published*
  numbers (their 0.1238 / 0.1755).
- Pretraining and embedding extraction do not touch xgboost or the split
  machinery at all. **The expensive artifacts — full/xl/xxl models and their
  extracted embeddings — are valid, full stop. No GPU retraining needed.**
- The controls stand: the shuffled-label collapse is env-independent; the
  velocity-baseline *conclusion* (velocity features don't recover the FM
  lift) is robust, though its point numbers would shift a little under a
  pinned xgboost (cheap to rerun, see §2).

## 2. MIGRATE NOW (cheap, blog-hardening — no retraining)

### 2a. Pin the downstream environment (his finding #2 + #3)
- `requirements.txt`: `xgboost>=2.0` → **`xgboost==3.2.0`** (NVIDIA's pin).
  He measured xgboost 3.3's changed early-stopping selecting a much worse
  iteration (raw 0.124 → 0.055 under their params). Our jobs resolved 3.3.x.
- Decide the device story: NVIDIA ran XGBoost on **CUDA**; our stage 05 runs
  CPU. Either run 05 with `--device cuda` on a GPU head, or note the device
  in the blog. His finding: device moves fusion-style models materially.
- **Then rerun STAGE 05 ONLY** (reads existing embeddings; CPU/GPU minutes)
  for full, xl (and xxl when it lands): 3 cheap jobs. Expect our baseline
  gate to move from 0.1421 toward their exact 0.1238 — if it does, that
  closes the "why does your baseline differ from theirs" question with one
  sentence. Publish the pinned-env table. Also rerun job_control_velocity
  (~4 min) so its numbers are on the same pins.

### 2b. Report the headline with a bootstrap CI (his "fusion is a
distribution" insight)
The 100K stratified test has only ~112 frauds; single-draw AP is noisy. His
seed×eval-bootstrap harness (scripts/nvidia_repro/run_peakhunt.py on his
branch) is the right idea — port the *method*, not the code: a ~40-line
script that bootstrap-resamples the test set under our saved model
predictions and reports AP as point + 95% CI for every row of our table.
Our 2x margin over their fusion will survive easily; showing the CI
preempts the obvious reviewer attack. (Do NOT adopt his "peak" framing —
report the distribution, never the max.)

### 2c. Take his results into the blog as the missing sections
- "First, does their blueprint even reproduce? Yes — exactly" (his raw
  0.1238 exact match, their weights through their recipe). Kills the
  "maybe their baseline was broken" attack on our headline.
- "Their design, faithfully trained by us: fm-only 0.0614" — the strongest
  possible ablation FOR our thesis: same data, same capacity, their
  tokenizer/objective/readout → 0.06; ours → 0.27. The gap IS the design.
  Cite his branch + checkpoint (/mnt/cluster_storage/nvpretrain/hf).
- His eos-token insight (causal LM: last token gets no gradient → pool the
  token before EOS) — one sentence in the blog's design section explaining
  a real pitfall of the causal readout. N/A to our bidirectional MLM.

## 3. MIGRATE OPTIONALLY (robustness appendix, ~1 GPU-hour)

**Cross-split check:** evaluate OUR full-scale embeddings under HIS
NVIDIA-native regenerated split (src/nvsplit.py on his branch). Our
benchmark.parquet implements the same protocol but is an independently
constructed draw (different normalize path + stratified sample), so the two
tables are "same protocol, not same rows." One eval-only run of our stack
against his split rows (rebuild benchmark from nvsplit → re-tokenize eval
windows for those rows → re-extract → 05) proves the headline is
split-robust. Do NOT switch our canonical eval to his split — our pinned
benchmark.parquet is what every measured number (controls, seeds, 512/1024)
is anchored to; keep it, add the cross-check as a column/appendix.

## 4. DO NOT MIGRATE (his replication stack — reference it, don't absorb it)

- src/nvidia_tokenizer/ (vendored NVIDIA tokenizer), nvcorpus.py, nvembed.py,
  nvscore.py, decoder_inference.py, the Llama model.py, max_ctx/_truncate
  machinery, scripts/nvidia_repro/* — that IS his deliverable. The cookbook
  stays 1-position/txn; the blog links his branch for the faithful repro.
- cudf-cu12/cupy deps (his GPU tokenizer path needs them; we don't).
- His nvsplit as our canonical split (see §3 — cross-check only).

## 5. RECONCILE WITH ZACH (questions before the blog cites his numbers)

1. His full.yaml says downstream trains on the full ~19.5M-row train set
   ("matching NVIDIA's raw baseline") while our reading of NB01 is a 1M
   balanced sample — which did his final 0.0614 table use?
2. Confirm his fm rows are single-txn embeddings (max_ctx≈14, NB04-style) so
   our table caption "their readout throws away history" is precise.
3. Swap benchmark row files to attempt a row-identical comparison (nice to
   have; §3 covers it if not).
4. His fusion-distribution machinery: agree on reporting convention
   (point + CI) so the two branches' numbers appear under one method.

## 6. Rerun matrix (what the pins actually cost)

| artifact | invalidated? | action |
|---|---|---|
| pretrained models (full/xl/xxl) | no | none |
| extracted embeddings | no | none |
| stage-05 tables (full/xl/xxl) | numbers move under pins | rerun 05 only (minutes each) |
| baseline gate 0.9875/0.1421 | expected to shift toward 0.1238 | same 05 rerun |
| shuffled-label control | no (env-independent conclusion) | none |
| velocity control | conclusion stands; numbers move | rerun job (~4 min) |
| probe seed CIs (512 logistic 0.226±0.006) | torch-based, xgboost-free | none |

Bottom line: **zero retraining; ~4 cheap eval jobs** to move the whole
campaign onto NVIDIA's pinned environment, plus one optional GPU-hour for
the cross-split appendix.
