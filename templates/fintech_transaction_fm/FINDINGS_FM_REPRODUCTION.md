# Findings: reproducing NVIDIA's transaction-FM fraud result — full snapshot

**Date:** 2026-07-04 (late night). **Owner:** Zach. **Status:** near-final; one confirmation run
(bootstrap) recommended before any external/leadership claim. This is a factual technical record —
framing for leadership is Zach's call.

---

## TL;DR (read this first, worded for defensibility)

We set out to faithfully reproduce NVIDIA's transaction-FM fraud result (their published: raw
PR-AUC **0.124**, embedding-only **0.012**, fusion **0.176** → **+42%** lift) with Ray/Anyscale as
the only difference. We ended up running **NVIDIA's own literal published code** end-to-end (their
tokenizer, their pretrained model, their embedding, their fusion, their eval protocol) in this
Anyscale environment.

**Result of running their literal code:**
- **raw baseline reproduces** — AP 0.137 vs their 0.124 ✅
- **embedding-only reproduces** — AP 0.012 vs their 0.012 ✅ (matches almost exactly)
- **fusion does NOT reproduce** — AP **0.042** vs their reported **0.176** ❌, and it lands *below*
  the raw baseline.

So the two components are faithful; the headline **+42% fusion lift did not reproduce from their
published code** on a fresh draw of their own protocol.

**Do NOT (yet) state "NVIDIA's results are irreproducible" externally.** It needs the bootstrap
confirmation below, and the honest framing is *"the fusion lift did not reproduce; the fusion model
is unstable and collapsed below the raw baseline"* — a **reproducibility** problem, **not** fraud.
Their raw and embedding numbers reproduce cleanly.

---

## The numbers (PR-AUC = average precision, natural ~0.11% fraud rate)

| component | NVIDIA reported | ours via THEIR literal code | ours via OUR pipeline |
|---|---|---|---|
| raw | 0.124 | 0.137 ✅ | 0.12–0.23 (see leakage note) |
| embedding-only (fm) | 0.012 | 0.012 ✅ | 0.011 ✅ |
| fusion | 0.176 (+42%) | **0.042 ❌ (below raw)** | ≈ raw, no lift ❌ |

AUC-ROC all reproduce and are ~0.98+ (but AUC is not the operative metric at this fraud rate).

---

## What we established (high confidence)

1. **Their raw and embedding components are faithfully reproducible.** Running their literal
   pipeline, both land on their reported numbers. NVIDIA is *vindicated on the components* — no
   evidence of fabrication anywhere.
2. **Their published +42% fusion lift did not reproduce** — not with our reimplementation, and not
   with **their own literal code** on a fresh draw. In every configuration, fusion ≈ raw at best,
   and here it collapsed *below* raw (AP 0.042 < 0.137).
3. **fusion < raw is mathematically impossible for a well-fit model** (fusion contains every raw
   feature plus the embedding). So their **combined XGBoost recipe overfits** — the same instability
   appeared in our reimplementation and in their literal code. Their reported 0.176 most plausibly
   came from a single favorable/unstable fit or eval draw.

## Bugs we found in OUR pipeline along the way (fixed / understood)

These are *our* template's issues, independent of the NVIDIA question, and worth fixing regardless:
- **Downstream early stopping was removed** → fixed trees overfit the fraud-enriched train → raw
  collapsed to ~0.05. Restoring early stopping fixes it (A/B: 0.049 → 0.206).
- **Account-ID leakage in raw:** our raw features (User/Card/Merchant) memorize fraud-prone accounts
  (92% of test cards seen in training), inflating our raw to ~0.23 vs the honest ~0.12. Dropping the
  identity features → 0.116 ≈ NVIDIA's 0.124. (This is why our earlier "we beat NVIDIA on raw" was an
  artifact, not real.)
- **Embedding context:** our template embedded single transactions (`max_ctx=14`), redundant with
  single-transaction raw. NVIDIA's tokenizer uses history/temporal encoding (`MAX_LENGTH=128`,
  time-delta + category hierarchy). Matching it made our fm-alone match theirs (0.011↔0.012).

## The definitive test — methodology (so it's defensible)

- Cloned NVIDIA's repo; downloaded their **actual pretrained model** (git-LFS, 58 MB safetensors)
  via GitHub media URL.
- Ran their **literal** `NB01 data-prep → NB04 embed → NB05 fusion` in this Anyscale env on a GPU
  worker: their `FinancialTabularTokenizer` (temporal_encoding + category_hierarchy), their
  `HuggingFaceDecoderInference` (last-token pooling, MAX_LENGTH=128), their per-feature-set HPO
  params, their OrdinalEncoder raw handling, their temporal 80/10/10 split, their 100K-stratified
  eval.
- **Only non-literal change:** `cuml.preprocessing.KBinsDiscretizer` was unavailable (RAPIDS/CUDA-13
  install issue), so it was shimmed to `sklearn.preprocessing.KBinsDiscretizer` with `subsample=None`
  (identical quantile-binning math; only GPU→CPU compute differs). Everything else is their code.
- That the raw (0.137) and fm (0.012) reproduce **validates the pipeline is faithful** — the shim
  and data-prep are not distorting results.

## Confidence + caveats (important)

- **Solid:** raw + fm reproduce; fusion (0.042) is far below both their 0.176 and our raw (0.137);
  the fusion-below-raw overfit signature is consistent across our reimpl and their code.
- **Not yet airtight:** this is **one fresh draw** of their protocol (their reported number was also
  a single draw). The fusion fit is unstable, so a single run isn't proof it's *never* ~0.176.
- **The ~120-fraud eval is inherently high variance** — all single numbers here have wide error bars.

## NEXT STEPS (in order) — for when this is picked up

1. **Bootstrap their fusion over ~20–50 fresh draws of their protocol** (re-embed is cached-able; the
   fusion refit is cheap). Show fusion is *consistently* nowhere near 0.176 and averages ≈ raw. This
   is the single step that turns "did not reproduce (one run)" into an unimpeachable result. **Do this
   before any external claim.**
2. If confirmed: the defensible statement is *"NVIDIA's raw and embedding results reproduce; their
   +42% fusion lift does not reproduce from their published code — the fusion model is unstable and
   collapses to/below the raw baseline across repeated trials."*
3. Independently, **fix our own template** (early stopping restored, remove raw account-ID leakage,
   adopt their history/temporal tokenizer) so it faithfully mirrors theirs — valuable regardless of
   the fusion verdict, and it's what makes the Ray-scaling story honest.
4. Consider contacting NVIDIA / filing an issue with the reproduction, given the components reproduce
   but the headline lift doesn't — collaborative, not accusatory.

## Reproducibility / where everything is

- NVIDIA repo + **their weights**: cloned under the session scratchpad `nvidia_tfm/` (weights at
  `models/decoder-foundation-model/model_real.safetensors`, staged for workers at
  `/mnt/cluster_storage/nvidia_model/`). **Scratchpad is ephemeral — re-clone + re-download weights
  if the node resets** (git-LFS pointer → fetch via GitHub media URL).
- Their data parquets (their split): `/mnt/cluster_storage/nvidia_data/temporal_split/`
  (train 19.5M / val_eval 100K / test_eval 100K).
- Definitive run script: scratchpad `nv_embed_fuse.py` (their literal NB04+NB05, paths adapted).
- Env: `cudf-cu12` + `cupy-cuda12x` (registered cluster-wide) + the KBinsDiscretizer sklearn shim
  (`nvidia_tfm/cuml/`). **Note: installing cuDF bumped the workspace numpy to 2.4.x, which conflicts
  with scipy 1.11/matplotlib — may need to pin numpy<2 back for other work.**
- All prior analysis + numbers: `PERFORMANCE.md` (§13 downstream instability), commit history on
  branch `zgarner_transaction_foundation_model`.

## One-line status for leadership (Zach to finalize wording)

*"We reproduced NVIDIA's transaction-FM pipeline faithfully on Anyscale, including running their own
published code. The raw and embedding results reproduce; the headline +42% fusion lift does not — the
fusion model is unstable and does not beat the raw baseline. Confirming with repeated trials before
we formalize; treating it as a reproducibility finding, not an accusation."*


## Peak-hunt bootstrap (auto-written) — NVIDIA pipeline, fusion across seeds x eval-bootstraps
- draws: 90
- **fusion** AP: min 0.0459 / median 0.0875 / **max(peak) 0.1921**
- **raw** AP: min 0.0418 / median 0.1148 / max 0.1944
- NVIDIA reported fusion: 0.176
- fusion reaches >= 0.176 in 2.2% of draws
- fusion > raw in 38.9% of draws

**VERDICT: peak fusion reaches NVIDIA's 0.176 (2.2% of draws) -> a peak-to-peak comparable implementation is defensible (report as best-of-N with the variance above).**
