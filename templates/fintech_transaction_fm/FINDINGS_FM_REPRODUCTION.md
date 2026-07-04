# Findings: reproducing NVIDIA's transaction-FM fraud result on Ray

**Date:** 2026-07-04. **Status:** components reproduce; fusion lift does not reproduce in our
reimplementation; one definitive check (running NVIDIA's literal fusion code) remains undone.

This is a factual technical record. Confidence levels are stated explicitly because the eval on
this benchmark is high-variance and because everything below is *our reimplementation* of NVIDIA's
approach, not their literal code (except where noted).

## The question

NVIDIA's transaction-FM blueprint reports, on the IBM TabFormer fraud benchmark (24.4M txns,
~0.12% fraud), that a foundation-model embedding fused with raw features beats raw alone by ~**+42%**
PR-AUC (their published: raw AP **0.124**, embedding-only **0.012**, fusion **0.176**). Our goal:
faithfully reproduce that, with Ray/Anyscale as the only difference (the scaling layer).

## Headline finding (stated carefully)

- **NVIDIA's raw and embedding-only components reproduce** almost exactly once we corrected our
  own errors. **They are vindicated on the components — no evidence of fabrication.**
- **Their fusion *lift* (+42%) does not reproduce in our faithful reimplementation.** With a raw
  baseline matching theirs and an embedding matching theirs, a combiner that *mathematically cannot
  score below raw* still shows the embedding adding **zero** lift.
- **We have NOT yet run NVIDIA's literal fusion code (their NB05).** So the defensible statement is
  *"our faithful reimplementation does not reproduce the fusion lift,"* **not** *"the result is
  irreproducible."* That distinction is the one remaining test (see bottom).

## Numbers (PR-AUC / average precision, natural ~0.11% prevalence)

| component | NVIDIA reported | ours (faithful) | status |
|---|---|---|---|
| raw (13 tabular features) | 0.124 | **0.11–0.12** | ✅ reproduces |
| embedding-only (fm) | 0.012 | **0.011** | ✅ reproduces |
| fusion (raw + fm) | 0.176 (+42%) | **≈ raw, +0.00 lift** | ❌ does not reproduce |

Bootstrapped over 50× 100K-stratified draws (NVIDIA's eval protocol): **fusion beats raw in 0–2%
of draws.** NVIDIA's reported raw/fusion values are each individually reproducible as *single* draws
(their raw ≈ our low end, their fusion ≈ our high end), but the paired +42% lift essentially never
co-occurs in a faithful run.

## What we found and fixed along the way (methodology / defensibility)

1. **Downstream bug (ours):** our fraud classifier had early stopping removed, training a fixed
   large number of trees → overfit the fraud-enriched training sample → raw collapsed to ~0.05.
   Restoring early stopping on a natural-rate validation set fixed it (raw → 0.16+). Confirmed via
   A/B on identical features: fixed-trees → 0.049, early-stop → 0.206.
2. **Our raw was inflated by account-ID leakage (ours, the key error):** the raw features included
   account identity (User, Card, Merchant Name). **92% of test transactions are from cards also seen
   in training**, so a shallow tree memorizes which specific accounts are fraud-prone. This alone
   accounted for ~+0.12 of our raw AP. **Dropping those identity features → raw = 0.116, matching
   NVIDIA's 0.124.** So our earlier "raw beats NVIDIA" was an artifact of this leakage, not a real
   result. This is why our raw was ~2× theirs.
3. **Embedding context was the fm lever:** we had been embedding each transaction *alone*
   (`max_ctx=14`), which is redundant with single-transaction raw features. NVIDIA's `MAX_LENGTH=128`
   embeds each transaction with ~10 transactions of *history*. Re-embedding at `max_ctx=128`
   (nb 05 knob `embed.max_ctx`) made our fm-alone match theirs exactly (0.011 vs 0.012) and made it
   far more independent of raw (score correlation 0.36 → 0.13) — i.e. genuinely complementary.
4. **Fusion still shows no lift** even after (1)–(3): a bulletproof AP-maximizing rank blend (which
   can weight the embedding to zero and thus can never underperform raw) selects **pure raw** — the
   embedding adds nothing on top of the honest raw. Single-XGBoost fusions overfit to *below* raw
   (a broken-fit signature), and multiple recipes did not recover a lift.

## Confidence

- **High confidence:** raw and fm components reproduce; our raw-leakage bug is real and quantified;
  the early-stopping bug is real. These are backed by clean, bootstrapped runs.
- **Well-evidenced but not final:** the fusion lift does not reproduce in our reimplementation
  (matched components + bulletproof combiner + 50-draw bootstrap + multiple joint-model attempts).
- **Caveats:** (a) the benchmark eval is high-variance (~120 frauds on NVIDIA's 100K-stratified
  protocol), so all single numbers have wide error bars; (b) everything here is our reimplementation
  of NVIDIA's method — we reproduced their *reported component numbers*, but did not run their
  literal fusion code.

## The one remaining definitive test

Run **NVIDIA's actual repository code** for the fusion step (`05_xgboost_fraud_detection.ipynb`,
their real cuDF/XGBoost pipeline) end-to-end on this data.
- If **their code yields +42% here** → the lift is real and the gap is a divergence in *our* fusion
  reimplementation (most likely raw×embedding interactions our combiner/tuning doesn't capture); we
  then diff their code against ours to find it. The FM's value is recovered.
- If **their code also shows no lift here** → the reported lift does not reproduce even with their own
  code, which is then a defensible, hard result.

This is bounded work (clone repo, match RAPIDS/cuDF environment, run their pipeline) and is the step
that converts "our reimplementation couldn't reproduce it" into a claim that holds up under scrutiny.
It has **not** been done yet; no conclusion about NVIDIA's result should be published before it is.

## What is NOT in question

The Ray/Anyscale scaling pipeline itself — distributed tokenization of 24.4M transactions, decoder
pretraining across GPU workers with Ray Train, distributed embedding extraction, downstream on the
cluster — works and is independent of the modeling comparison above.
