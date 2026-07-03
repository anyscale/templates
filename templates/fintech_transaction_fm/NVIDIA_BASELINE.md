# NVIDIA blueprint — exact recipe, our divergences, and the parity plan

Extracted 2026-07-03 from NVIDIA's open code so we stop guessing at the baseline.
This is the authoritative reference for "what does NVIDIA actually do" and the plan
to reach comparable numbers. Pair with `CHANGES.md` (current run state).

## Sources
- Repo: https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model
  - `01_dataset_baseline.ipynb` — raw XGBoost baseline
  - `05_xgboost_fraud_detection.ipynb` — raw vs fm vs fusion comparison
  - `src/tokenizer/` — their financial tokenizer (financial_pipeline.py etc.)
- Blog: https://blogs.nvidia.com/blog/financial-institutions-transaction-foundation-models/
- Data: original TabFormer CSV `card_transaction.v1.csv` (2.35 GB) is on disk at
  `/mnt/cluster_storage/transaction-fm/source/card_transaction.v1.csv` — so exact
  replication from raw is possible.

## NVIDIA's reported numbers (their eval protocol — see caveat below)
| feature set | AUC-ROC | PR-AUC (AP) |
|---|---|---|
| raw (13d) | 0.9885 | 0.1238 |
| embedding-only | 0.8775 | 0.0123 |
| fusion | 0.9925 | 0.1755 |

## NVIDIA's exact recipe

### Data prep (from the raw CSV)
- `Hour = Time.split(':')[0].astype(int)`
- `Amount = Amount.replace('$','').replace(',','').astype(float)`  ← **raw float, NO log transform**
- `_target = (Is Fraud? == 'Yes'|'1')`
- 13 features, verbatim:
  `['User','Card','Year','Month','Day','Hour','Amount','Use Chip','Merchant Name',
    'Merchant City','Merchant State','Zip','MCC']`
  Note: **'Merchant Name'** (a real high-cardinality string) and **'User'/'Card'**
  kept separate. Year/Month/Day are separate columns.

### Categorical encoding — the big one
```python
OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
  applied to all dtype ∈ {object, category} columns, remainder passthrough
```
Dense, **collision-free** label codes, **fit on the (balanced) train sample**.
NOT hashing. (We currently crc32-hash merchant/city/zip into 100k buckets — lossy.)

### Class imbalance — balanced SAMPLING, not weighting
```python
# NB01: total=1_000_000 ; NB05: total = n_train
n_fraud  = min(#fraud, int(total*0.1))     # cap fraud at 10% of the sample
n_normal = min(#normal, total - n_fraud)   # fill the rest with normals
train = all/sampled fraud + sampled normals   # ~2.4% fraud when 24k fraud / 1M
scale_pos_weight = 1.0                         # NO reweighting — sample is enriched
```

### Eval protocol — stratified SUBSAMPLE (caveat!)
```python
EVAL_SAMPLES = 100_000
stratified_subsample(val/test, n=100_000, stratify=_target)  # preserves fraud rate
```
So their AP is measured on **100k stratified rows (~120 frauds)**, NOT the full
holdout. Our pipeline scores the **full 2.44M holdout (~2700 frauds, holdout_keep=1.0)**
— a different, more stable metric. **Their 0.124 is therefore not directly comparable
to our full-population number.** To compare, match their protocol (or report both).

### XGBoost params — HPO-tuned SEPARATELY per feature set
All share: `tree_method='hist'`, `device=cuda`, `early_stopping_rounds=20`,
`eval_metric='auc'`, `scale_pos_weight=1.0`.
```python
XGB_PARAMS_RAW      = {n_estimators:400, max_depth:8,  lr:0.0023,  colsample_bytree:0.95,
                       min_child_weight:12,   subsample:0.673, reg_alpha:0.01,   reg_lambda:0.001}
XGB_PARAMS_EMBED    = {n_estimators:435, max_depth:12, lr:0.03774, colsample_bytree:0.587,
                       min_child_weight:2.61, subsample:0.569, reg_alpha:0.01364,reg_lambda:9.7e-5, gamma:1.7}
XGB_PARAMS_COMBINED = {n_estimators:512, max_depth:12, lr:0.00305, colsample_bytree:0.768,
                       min_child_weight:25.85,subsample:0.65,  reg_alpha:0.01,   reg_lambda:0.0001, gamma:4.8}
```

## Where WE diverge (and why raw stalled at ~0.05)
| aspect | NVIDIA | ours (current) | likely impact |
|---|---|---|---|
| categorical encoding | OrdinalEncoder (dense) | crc32 hash % 100k | med–high (collisions/scatter; loses Merchant Name entirely — we only have merchant_id) |
| Amount | raw float | sign·log1p | low |
| imbalance | balanced 1M sample + spw=1.0 | full natural + spw (was neg/pos≈700 → now sqrt≈26) | **high** — spw=neg/pos was catastrophic |
| learning_rate | 0.0023 (raw), HPO per set | 0.03 shared | med |
| params | 3 separate HPO sets | 1 shared recipe (by design) | med — fm/fusion untuned vs NVIDIA |
| eval | 100k stratified subsample | full 2.44M holdout | changes the number; ours is harder/stabler, not comparable |
| Merchant Name | yes (string) | **missing** — our loader kept merchant_id (int) only | possibly high |

## Confirmed results so far (our pipeline, real TabFormer)
- Old (spw=neg/pos, tiny enriched train): raw AP 0.017.
- train_keep=1.0 + spw=neg/pos≈700: raw AP 0.0085 (spw crushed it).
- train_keep=1.0 + spw=sqrt≈26 (sample sweep, 300 rounds, 218 test frauds): raw
  AP ~0.05, AUC ~0.96. **Still ~2.4x below NVIDIA's 0.124, and on a different eval.**
- fm-only AP 0.0023, AUC 0.634 (bug #2, untouched).

## ✅ REPRODUCED (2026-07-03) — NVIDIA's raw baseline, exactly
`scripts/repro_nvidia_raw.py` (their NB01 recipe, verbatim, run on a GPU worker off
the on-disk CSV) gives **test AUC 0.9866 / AP 0.1248** vs NVIDIA's 0.9885 / 0.1238 —
matched. Split reproduced too (train 19.5M/0.128%, test 2.44M, eval=100k stratified).
**Conclusion: our DATA and FEATURES are sound.** Our pipeline's low raw AP
(0.017→0.05) was ENTIRELY recipe divergence, not a data/feature defect. The knobs
that matter, in order: OrdinalEncoder (not hashing) + having Merchant Name, balanced
1M sample + spw=1.0 (not neg/pos), and the 100k-stratified eval (vs our full-holdout).
Caveat: best_iter=1, fit 1.4s — the 100k/~112-fraud eval is noisy and a single strong
tree already separates; this faithfully reproduces THEIR (equally noisy) number.

## HARNESS: `scripts/nv_downstream.py` (NVIDIA recipe on OUR embeddings)
Ports NB05 onto our pipeline: balanced train sample + stratified eval + spw=1.0 +
per-feature-set HPO params, measuring raw/fm/fusion the way NVIDIA does. KEY
efficiency: embeds only the SAMPLED windows (~1.2M), not all 24.4M — but seq_len-4096
embed is still the bottleneck (~45 min for 1.2M on ~8 A10G; I initially under-estimated
this as "5 min"). Fast-iteration profile for tuning the FM:
```
python scripts/nv_downstream.py --tag full --train-total 1000000 --eval-n 100000  # final number
python scripts/nv_downstream.py --tag fast --train-total 200000  --eval-n 30000   # fast iterate
python scripts/nv_downstream.py --tag fast --skip-embed                           # re-fit XGB only (secs)
python scripts/nv_downstream.py --tag exp --model-dir <new_fm_ckpt> ...           # eval a new FM
```
Headline = fm-only vs NVIDIA 0.0123 (needs no raw). raw here is APPROXIMATE (passthrough
still hashes merchant_id/city); for exact raw parity, fix `_raw_features` to carry the
full merchant_id + raw city/state strings and ordinal-encode at downstream (needs a
re-tokenize of eval). fm-only is the real question and is exact now.
First full run (tag=sample_embeddings, 1M/100k) launched 2026-07-03 ~19:37; result TBD.

## ✅ FM FIXED (2026-07-03) — the divergence was EMBEDDING CONTEXT LENGTH
Symptom: fm-only AUC 0.62 (near random), fusion ≤ raw. Ruled out: pooling position
(last vs last_real both collapsed), undertraining (we run ~16k steps to ppl 1.7 vs
their 30-step demo), weight-load bug (75/75 keys match), tokenizer (ours == their
default 12-field pipeline). **Root cause:** NVIDIA (NB04) pretrains at 4096 but
**extracts each txn's embedding from only its last MAX_LENGTH=128 tokens (~10 txns),
last_token pooled.** We embedded from up to 314 txns (seq_len 4096) → the last-token
vector averages away into anisotropy. Fix: `EmbeddingExtractor(max_ctx=128)` truncates
each window to BOS + last 127 real tokens (…EOS). Result (fast profile, noisy 34
frauds): **fm AUC 0.62→0.795, fm AP 0.0023→0.0092, fusion 0.146 > raw 0.131** — FM
now adds lift like NVIDIA. (High pairwise cosine 0.97 was benign transformer
anisotropy, NOT collapse — a red herring.) Full 100k run in progress for stable nums.
Faithfulness principle reaffirmed: this was a divergence to eliminate, not a knob.
TODO make it clean in the pipeline: either tokenize eval windows at seq_len 128
(matches NB04, 32x cheaper embed) or keep max_ctx truncation; wire into configs/nbs.

## ⚠️ UPDATE — 128-ctx wasn't enough; the REAL divergence is SINGLE-TXN embedding
Full 100k, 128-token-truncated embedding (stable, ~113 frauds):
  raw AUC 0.986/AP 0.162 · fm AUC 0.781/AP 0.031 · fusion AUC 0.984/AP 0.142.
Gap NOT closed: fm AUC 0.781 < NVIDIA 0.878, and **fusion 0.142 < raw 0.162** (FM adds
NO lift — the whole point fails). Also the low-lr fusion/raw fits early-stop almost
immediately on the noisy 100k val (fusion best_iter=0) — a fitting-stability confound.

**THE fundamental divergence (from reading `src/tokenizer/pipeline.py::encode`):**
NVIDIA's embedding (NB04) runs `encode(token_df, max_length=128)` where token_df has
**one row per transaction**, and encode makes `<bos> col1..col12 <eos>` — i.e. **each
transaction is embedded ALONE** (~14 real tokens; 128 is just pad headroom). Their FM is
pretrained on multi-txn sequences (`to_corpus_lines`, chunk 315) but at INFERENCE embeds
**single transactions**. WE embed the whole history window (up to 314 txns) → a blurry
history summary, not NVIDIA's crisp per-transaction vector. That's why fusion doesn't add
lift. **Fix = embed single transactions.** Our eval window ends with the target txn's 12
tokens, so `--embed-max-len 14` (BOS + target 12 tokens + EOS) reproduces NVIDIA's
single-txn encoding without re-tokenizing. Testing now (tag=fasttxn). If it works, make
it clean: tokenize/encode eval as single-txn (or keep max_ctx=14) and wire to nb/configs.
Reminder: high embedding cosine (~0.97) is benign anisotropy, NOT the problem.

## THE PLAN (resume here)
Goal Zach set: match then beat NVIDIA, honestly, keeping the Ray pipeline clean.

1. **Reproduce NVIDIA's raw 0.124 exactly first** (validation that our data/features
   are sound). Build a standalone baseline from the raw CSV with their EXACT recipe:
   OrdinalEncoder, raw Amount, 13 cols incl. Merchant Name (needs the original CSV —
   our raw/full dropped Merchant Name & split User/Card into card_id, so read the CSV),
   balanced 1M sample, spw=1.0, XGB_PARAMS_RAW, 100k stratified eval. Small (1M train) —
   runs in-process, but keep heavy work OFF the head node. If we hit ~0.124 → the gap
   was purely recipe/encoding, and we know exactly which knobs.
2. **Decide the template's stance** (design fork for Zach):
   - (a) Adopt NVIDIA's protocol wholesale (balanced sampling, OrdinalEncoder, per-set
     HPO params, stratified eval) → directly comparable numbers, less "our own" story.
   - (b) Keep our cleaner design (full-population exact eval, one shared recipe, Ray
     distributed) and report BOTH our numbers and NVIDIA's, explaining the protocol
     difference. More honest/defensible, not a single headline number.
   - Recommendation: (b) for the template narrative, but run (1) as the credibility check.
3. **Fix encoding regardless**: replace crc32 hashing in `_raw_features`
   (`src/flat_tokenizer.py`) with dense ordinal codes; recover Merchant Name if we
   re-ingest the CSV with that column. This helps raw AND fusion.
4. **Then bug #2 (TFM representation)**: fm-only AP 0.0023 / AUC 0.634 (~random). NVIDIA
   gets 0.0123 with EMBED params (depth 12, lr 0.038) on their embeddings. Part of our
   gap may be (i) untuned shared recipe, (ii) last-token pooling, (iii) under-converged
   pretrain. Re-measure fm with NVIDIA's EMBED params before blaming the representation.

## Status of code changes (committed + pushed, branch zgarner_transaction_foundation_model)
- `train_keep` config knob (full=1.0) — decouples downstream train size from eval sizing.
- `02_tokenize.py` streams per-emit (no giant materialize); empty-group Arrow crash fixed
  (`<U16` strings) — but empty-TENSOR crash on `emit="pretrain"` at full still open
  (run_pipeline.py path); eval-only (`emit="eval"`) is safe and is what we used.
- `scale_pos_weight = sqrt(neg/pos)` in downstream.py + probe_raw.py (interim; NVIDIA
  uses balanced-sample + spw=1.0 — may switch to that per plan step 2).
- `scripts/probe_raw.py` — distributed raw-only XGBoost off tokenized eval (no embed).
- Full eval re-tokenized at train_keep=1.0 (25 GB, 24.4M windows) at
  `/mnt/cluster_storage/transaction-fm/tokenized/full/eval`; `model/full` intact
  (pretrained Llama — skip re-pretrain).

## Gotchas
- HEAD NODE OOM-crashed once on CPU work — keep tokenize/embed/XGBoost DISTRIBUTED on
  workers; nothing big pulled to the driver.
- Ray Train's 60s worker-group-startup timeout fires when the cluster is scaling up
  post-downscale — set `RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S=600` and/or fewer workers.
- `tokenized/full/pretrain` was deleted this session (deterministic; regenerate only if
  re-pretraining, which we're not — `model/full` already trained).
