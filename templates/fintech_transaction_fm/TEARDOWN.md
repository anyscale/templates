# TEARDOWN — independent design review + experiment state (2026-07-07)

An adversarial fresh-eyes review (independent agent, no access to our
hypotheses, code + literature only) plus the empirical probes we ran the same
night. This file is the source of truth for WHY the FM underperforms the
benchmark and WHAT we are doing about it. Numbers below are measured.

## The bombshell finding

**NVIDIA's winning "foundation model embedding" is not a sequence embedding.**
Their notebook 04 encodes each benchmark transaction as a standalone
`<bos> + 12 field tokens + <eos>` sequence (MAX_LENGTH=128, one row per
transaction) — zero history. Their +41.8% AP is a learned nonlinear
re-encoding of the same 13 raw columns fused back with those columns, plus
the LM's co-occurrence prior acting as a density model. The blog's "final
position has observed the entire sequence" narrative does not match the
shipped evaluation code. Also: their embeddings-ONLY score is weak —
AP 0.0123 / ROC 0.8775 — the entire game is fusion complementarity.

We optimized the opposite quantity: a 512-txn history summary from a
bidirectional MLM whose pooled states are never trained to summarize
anything, minus the per-transaction fields that carry the interaction signal
the benchmark rewards.

## Measured state (all on the pinned benchmark: 1M balanced train /
## 100k stratified val + test; baseline = 13 raw feats + their XGB params)

| what | ROC-AUC | AP |
|---|--:|--:|
| baseline (reproduced 5x, byte-stable) | 0.9875 | 0.1421 |
| NVIDIA published: baseline / combined | 0.9885 / 0.9925 | 0.1238 / 0.1755 (+41.8%) |
| NVIDIA published: embeddings-only | 0.8775 | 0.0123 |
| ours embed-only — BEST across {navy, b128, +CoLES} x {last, mean, max} x {logistic, MLP, XGB} x {PCA64, no-PCA} | 0.798 | 0.0042 |
| ours fused — BEST (CoLES-last, no-PCA XGB) | 0.9889 | 0.1005 (−29%) |

Exonerated by experiment: convergence (navy 3,180-step vs b128 1,600-step
both fail), pooling (last/mean/max), PCA (minor: navy fusion 0.041→0.081
without it), the classifier (linear/MLP/no-PCA XGB all ~0.7-0.8 embed-only),
the join (100.00% match after the amount-cents fix), leakage (none found).
The CoLES contrastive term un-collapsed the space (pairwise cos 0.85→0.67)
and helped fusion slightly (0.081→0.101) but aimed at the wrong invariance.

## Ranked design flaws (from the independent teardown; file:line evidence in
## the review, summarized here)

1. **Readout structurally incapable of a per-transaction label.**
   Mean-pooling: two eval windows of the same card at adjacent targets share
   ~99.8% of content → near-identical vectors, opposite labels → embed-only
   AP ≈ prevalence is the PREDICTED outcome. Last-pooling: an MLM's
   unmasked-position state is never supervised (field_loss gathers
   hidden[masked] only) → untrained byproduct at inference. Frozen pooled
   MLM vectors for per-event labels have no precedent in the literature —
   every MLM system (TabBERT/FATA/UniTTab) fine-tunes or trains a head over
   per-position states; frozen-readout protocols are won by causal/predictive
   objectives (NVIDIA, Nubank, Featurespace NPPR).
2. **Per-transaction geography + month deleted.** Merchant State reduced to a
   per-card modal static (cannot express "suddenly transacting in a new
   state" — a top handcrafted fraud feature per Bahnsen et al.); Zip dropped
   entirely; Month absent. NVIDIA's 12 tokens include STATE, ZIP3, MONTH per
   transaction.
3. **Whole-transaction masking makes the pretext solvable from card
   marginals.** All 7 dynamic fields masked jointly while identity statics,
   position, and the time-gap token stay visible → the task reduces to
   "recall this card's profile," which XGBoost already extracts from raw
   User/Card. Also forfeits intra-transaction conditional structure
   (P(state|merchant,amount)) — what NVIDIA's 12-tokens-attending learn for
   free. Every lit system masks/predicts at FIELD granularity.
4. **The CoLES seq-CL term trains the ANTI-fraud invariance** — pulling two
   temporal halves of a card to the same point suppresses within-card
   temporal deviation, which IS the fraud signal. CoLES's own scope is
   card-level labels (churn/default), not per-event fraud.
5. **Identity statics broadcast into every position** → pooled vector and
   PCA (fit on the fraud-user-oversampled balanced train) dominated by user
   identity → depth-12 trees memorize fraud users → plausible mechanism for
   fused-below-baseline.
6. **Frozen window boundaries** (non-overlapping, identical every epoch) —
   threw away ~13x free augmentation vs strided windows (TabBERT: stride 5).
7. **Gratuitous micro-losses:** MCC hashed mod 128 despite a closed 109-code
   vocab (guaranteed collisions); refund sign destroyed by abs(); Errors?
   excluded even as HISTORY input on an incorrect leak argument (a bad-PIN
   streak BEFORE the target is legitimate signal; NVIDIA doesn't use it —
   free lift available via previous-txn error).

Cleared as NOT the problem: amount bucket count (ours 16 vs their 7),
merchant vocab (ours 20k+tail vs their 2000 hash), time-delta features
(ours richer), LR/AMP/checkpointing mechanics, capacity/steps (winner has
the same ~29M params and ~3k steps).

## Misquote corrections vs the papers we cite

- FATA-Trans: uses a field transformer per row (intra-row interactions),
  statics in ONE dedicated position (not broadcast), learnable time-position
  mixing, and FINE-TUNED evaluation. We kept its compression trick and
  discarded what made it work.
- CoLES: random overlapping slices, margin loss + hard negatives, GRU,
  card-LEVEL downstream labels.

## RESULTS UPDATE (2026-07-08, overnight)

**Run 1 (readout surgery, existing checkpoints — 33/36 fits):** every
target-conditioned readout beat every pooled readout. The 7-dim SURPRISE
vector: best standalone signal of the campaign (navy 0.853 ROC embed-only)
and the first baseline-beater (cl surprise + 13 raw, RAW params:
0.9859/0.1476 = +3.9% AP). Flaw #1 empirically confirmed. Fusion fits are
params/variance-fragile (navy surprise: best standalone, worst fused) —
never conclude from a single fusion fit.

**Run 2 + 2b (tokenizer parity + G2/G3/recency retrain):** the pooled
embedding CAME ALIVE — embeddings-only 0.9914 ROC / 0.1623 AP (**+14.2% over
the 0.1421 baseline**, vs -98% before; NVIDIA's own embeddings-only is
0.8775/0.0123). Embedding collapse cured (pairwise cos 0.854 -> 0.219).
acc_mcc 0.28->0.955, acc_merchant_bucket 0.12->0.758 (sibling-visible
masking learning intra-txn conditional structure, as designed). Leak canary
clean (amount epoch-0 0.475, gradual curve). Caveats: trio's combined
(PCA64 + their COMBINED params) collapsed AGAIN (0.0563) — third instance
of fusion-harness pathology, resolved by the probe's rawparams/linear/MLP
fusions; reco regressed hard (HR@10 0.077, ledgered, deprioritized); the
navy-vs-R2 per-field crowding comparison is confounded by the masking
change (task difficulty differs) — crowding remains unmeasured.

Verdict so far: flaws #1 (readout) and #2/#3/#7 (inputs/masking) were both
real and BOTH fixes pay. The causal pivot (old Run-3) looks unnecessary —
parked unless the remaining evals reverse.

**R2 probe (no-PCA) + independent leak audit (2026-07-08 ~01:00):**
- HEADLINE PROBE RESULT: R2 pooled-LAST embedding, raw 512d, small MLP:
  **0.9975 ROC / 0.4655 AP** (baseline 0.1421; NVIDIA combined 0.1755).
  Same embeddings under their exact PCA64+XGB harness: 0.1623 (+14.2%) —
  THE protocol-faithful claim. Decomposition to report honestly:
  0.1421 (their baseline) -> 0.1746 (single-txn, MLP) -> 0.4655 (full
  history, MLP); PCA was destroying ~3x signal; mean pooling still dead
  (position matters, not pooling flavor); fusion now HURTS (the
  parity-complete FM subsumes the 13 features).
- ADVERSARIAL AUDIT VERDICT: SUSPICIOUS-but-mechanically-CLEAN. All leak
  vectors verified clean with empirical harnesses (prev_error shift,
  stride/cutoff, probe alignment, dedup label-neutrality, metric wiring).
  Mechanism validated in-data: 90.0% of test frauds have another fraud in
  the previous 512 same-card txns vs 7.3% of normals — the history readout
  legitimately detects mid-burst cards from auth-time-legal features.
- DISCLOSED IMPURITY: 1,394 val-period txns (10.6h of the boundary day)
  visible to pretraining via the quantile-vs-date cutoff mismatch; ZERO
  test-period rows. Fix: unify splits.json to the date cutoff.
- SEED REPLICATION (4 seeds, pooled-last no-PCA): MLP AP 0.33 +/- 0.10
  (0.4655 was a favorable draw — report the range, never the max);
  logistic 0.226 +/- 0.006 (tight); XGB 0.2581 (deterministic). DURABLE
  CLAIM: embedding-only under stable heads = AP 0.23-0.26 at ROC ~0.997,
  beating NVIDIA's combined headline (0.1755) by 30-47% and the baseline
  by 60-82%.
- REQUIRED CONTROLS before publishing (status):
  1. MLP-on-13-raw fair-head control — DONE: raw-13 under torch heads is
     ROC 0.47-0.61 / AP ~0.001 (ordinal ids need trees) -> the lift is the
     EMBEDDING, not the head choice
  2. seed variance — DONE: 4 shuffle-seeds + 1 post-fix init-seed all in
     band (logistic 0.216-0.232, MLP 0.24-0.47, XGB 0.2581 det.)
  3. shuffled-label / shuffled-embedding sanity — QUEUED (morning)
  4. checkpoint provenance — ATTESTED: embeddings/full extracted by the R2
     job's own entrypoint immediately after 03's final checkpoint; no
     probe_by_epoch selection involved
  5. classical history-aggregates baseline (XGB on 13 + burst/velocity
     features) — QUEUED (morning): quantifies how much of the lift a
     non-FM burst detector recovers
  6. cutoff unification — QUEUED

**RUN-G1 (attention fusion): FAILED — OOM, parked (2026-07-08 ~01:30).**
Both attempts died in step 1: the intra-tx attention stack tried to allocate
+8.00 GiB on top of 15.2 in use (A10G 22 GiB) — the exact pre-registered
risk. Per the pre-agreed rule, NOT batch-hacked (update-count trap). Fix if
G1 is revisited: chunk the [B*S, F, d] attention into slices (linear memory,
identical math). R2's model/embeddings were MOVED aside by G1's entrypoint
(model/full_old_<stamp>, embeddings/full_old_<stamp>) — intact. G1 is no
longer on the critical path: the pooled-last result made the summation-vs-
attention question academic for the blog.

## The three-run plan (agreed 2026-07-07 night)

- **Run 1 — readout surgery, ZERO retraining** (STATUS: built, running).
  From the existing checkpoints, per benchmark row extract: (a) the
  target-position hidden state with the target's dynamic fields MASKED (the
  state the MLM was actually trained to make informative); (b) the per-field
  SURPRISE vector — CE of the true target fields under the MLM heads — an
  anomaly score conditioned on history; (c) a single-transaction window as
  the direct NVIDIA analogue. Fuse each with the 13 raw features.
  Decides: "representation fine, readout wrong" vs "representation empty."
- **Run 2 — tokenizer parity retrain** (STATUS: built, queued behind Run 1).
  Add per-txn merchant_state / zip3 / month / direction(sign) /
  prev_error(shifted, leak-safe) dynamic fields; exact 109-code MCC vocab
  (kill the modulo); DROP user/card identity statics from FM input (identity
  stays in the raw features); independent per-field masking (15%/field);
  seq_cl_weight: 0; stride-halved pretrain windows (2x data). Same 29M arch,
  same budget, evaluated with Run 1's winning readout.
- **Run 3 — causal next-transaction variant** (STATUS: parked on purpose —
  more ideas to evaluate first). Same heads, shifted targets, every position
  supervised; the maximal test of "frozen readouts need predictive
  objectives" while keeping 1-position-per-txn context.

Explicitly rejected: "train longer / bigger" — nothing suggests capacity or
steps bind; the system computes a careful answer to a question the benchmark
doesn't ask.

## Standing lessons (process)

- Reproduce the published baseline through the SAME pipeline before any
  model work (gate job; EXPERIMENT_LOG.md has the 0.76-era confession).
- Audit every feature end-to-end against the reference system BEFORE
  training anything (this file exists because we didn't).
- Never delete artifacts; move aside with a shared timestamp.
- Batch-size bumps to "saturate GRAM" halve update counts — small-model
  pretraining is update-count-bound (b128 regression, caught by TB).
- Join keys on (card, minute) collide on fraud bursts; use amount-cents
  tiebreak.
