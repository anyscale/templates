# [BLOG DRAFT] One position per transaction: beating NVIDIA's transaction-FM blueprint with 4–13x longer context on Ray

> Status: full draft, numbered blanks `[B#]`. Anyscale house style (result
> first → setup → hero table → mechanism sections → cost/scale → 3-command
> repro → takeaways → CTA).
>
> REMAINING BLANKS / GATES:
> - DONE: 2048 fulltest (0.2665 — dilution confirmed, peak at 1024)
> - DONE: paired bootstrap — strict ordering 1024 > 512 > 2048, all significant
> - DONE: reco readout ladder (MLP 0.523; blend hybrid optional)
> - figures to export; Zach confirms repro-branch numbers/framings
> - DONE: xgboost==3.2.0 + CUDA parity, bootstrap CIs, full-test-period eval

---

NVIDIA's [transaction foundation model blueprint](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model) sets a public bar for fraud detection with foundation models: pretrain a small Llama decoder on 24M card transactions (IBM TabFormer), embed transactions, and fuse the embedding with 13 raw features into XGBoost — test average precision **0.1755**, a +42% lift over their raw baseline.

We built a transaction FM on Ray with a different tokenizer, objective, and readout, and evaluated it two ways. On **their exact protocol** (100k stratified test), our embedding **alone** — no fusion, no PCA — clears their published fusion headline with bootstrap probability **0.94–0.99 at every context length we trained** (512, 1024, 2048 transactions, at matched 20-epoch training budget). And on the **full 2.44M-transaction test period** (2,724 frauds, where confidence intervals shrink 5x), the embedding beats the identical-rows raw baseline by **+34% at 512 context and +45% at 1024**, with non-overlapping intervals. The core change fits in one sentence: **one position per transaction instead of ~12 tokens per transaction**, which buys 4–13x their ~315-transaction context at identical model capacity — plus a masked-field objective and a last-position readout that make the history actually usable. Training runs in about two hours on four A10Gs; everything reproduces from a public template with three job submissions.

[B1: hero figure — two panels sharing a y-axis. Left: their-protocol 100k, AP dot + 95% CI per model with NVIDIA baseline 0.1238 / fusion 0.1755 reference lines. Right: full test period, baseline 0.2081 vs embed_xgb 0.2788 (512) / 0.3027 (1024) / 0.2665 (2048) with CI whiskers. Caption: the two panels are different eval sets — never compare across them.]

## Why transaction foundation models — and why NVIDIA's blueprint is the right bar

Banks and fintechs are converging on transaction foundation models: one self-supervised transformer over raw transaction sequences whose embedding feeds fraud, churn, credit, and personalization, replacing per-task feature pipelines. Stripe, Visa (TREASURE), Nubank, and Revolut have all published variants. NVIDIA's blueprint is the most reproducible public version — real dataset (TabFormer: 24.4M transactions, ~6k cards, 0.12% fraud), a published protocol (temporal 80/10/10 split, 1M balanced train / 100k stratified test, HPO'd XGBoost), and published numbers.

Credit where due: **the blueprint reproduces**. We independently rebuilt their entire stack — their tokenizer (vendored), their Llama decoder config, their training recipe, their pinned library versions — and their raw baseline lands at **0.1238 to the fourth decimal**. Everything we report below is measured against a protocol we first verified we could reproduce exactly. [TODO: link Zach's faithful-repro branch/appendix.]

## Our approach: spend positions on transactions, not tokens

Their tokenizer flattens each transaction into ~12 tokens in one shared vocabulary, so a 4096-token context window holds ~315 transactions. We use a **field-split tokenizer** (FATA-Trans lineage): static card-level fields are embedded once and broadcast; each dynamic field (amount, merchant, MCC, hour, channel, …) gets its own embedding table; a transaction is **one position** whose vector is the sum of its field embeddings. The same ~29M-parameter budget then reads:

| | NVIDIA blueprint | ours |
|---|---|---|
| tokens per transaction | ~12 | **1 position** |
| transactions in context | ~315 | **512 / 1024 / 2048** |
| objective | next-token (causal) | **per-field masked modeling** (+ merchant InfoNCE) |
| embedding readout | single transaction, no history | **last position of the full history window** |

Three details that turned out to be load-bearing (mechanism sections below):
- **Independent per-field masking**: each field draws its own mask, so the pretext task is predicting a hidden field from its visible siblings — P(state | merchant, amount) — not reconstructing whole rows from card averages.
- **Continuous periodic channels**: learned Fourier/Time2Vec banks over signed log-amount and log time-delta ride alongside the bucketed tokens.
- **The readout**: windows are right-aligned, so the last position *is* the target transaction, read with the card's full history behind it. That 512-d state goes straight into XGBoost — no PCA.

The pipeline is Ray end-to-end: Ray Data builds the ~100k-merchant vocab and streams the field-split tokenization (`map_groups`, no global shuffle); Ray Train runs DDP masked pretraining (4 GPU workers, one `ScalingConfig` change from laptop to cluster); Ray Data streams CPU-read + GPU-infer embedding extraction; XGBoost consumes the parquet. [B2: pipeline architecture figure — reuse the README ASCII diagram as a proper graphic.]

## Benchmark setup

- **Data & protocol**: IBM TabFormer; NVIDIA's notebook-01 protocol transcribed verbatim (temporal 80/10/10 by date, 1M balanced train, 100k stratified val/test, their three separately-HPO'd XGBoost param sets, early stopping on val AUC). Environment matched to theirs: xgboost==3.2.0, XGBoost on CUDA.
- **Models**: ~29M params (their capacity), 512d/8L/8H, trained from scratch per context length — no pretrained weights anywhere. [Footnote: the 2048 model trained 20 epochs + a 20-epoch warm-restarted continuation; see the undertraining section.]
- **Two evaluation sets, used for different claims.** (1) *Their protocol*: the 100k stratified test draw — the only set comparable to NVIDIA's published numbers, but it contains just **112 frauds**, so AP intervals are wide (±0.05–0.08) and we report bootstrap CIs on every row. (2) *Full test period*: all 2.44M test-period transactions (2,724 frauds) — the same protocol with the sampling variance removed. Intervals shrink ~5x and model-vs-model differences become resolvable. **AP values are not comparable across the two sets** (the full-period baseline alone scores 0.208 vs 0.140 on the 100k draw), so each table serves exactly one purpose: table 1 compares against NVIDIA, table 2 compares our models against each other.
- **Fairness**: identical benchmark rows for every model within a table; embeddings joined exactly (card, timestamp, amount-cents; 100.00% match); every AP carries a bootstrap 95% CI; ordering claims use *paired* bootstrap (same resampled rows scored by both models). Controls in the trust section.

## Results

### Table 1 — their protocol (100k test, 112 frauds): the NVIDIA comparison

| model | context (txns) | AP | AP 95% CI | P(beats their fusion 0.1755) |
|---|---|---|---|---|
| raw 13 features + their XGB (baseline) | — | 0.1399 | [0.091, 0.214] | 0.16 |
| *NVIDIA published: baseline* | *~315* | *0.1238* | — | — |
| *NVIDIA published: fusion (their headline)* | *~315* | *0.1755* | — | — |
| their protocol (PCA64+XGB) on our embedding | 512 | 0.1679 | [0.117, 0.245] | 0.49 |
| their protocol (PCA64+XGB) on our embedding | 1024 | 0.1957 | [0.140, 0.278] | 0.79 |
| our embedding → linear head, no PCA | 512 | 0.2185 | [0.172, 0.280] | **0.96** |
| **our embedding → XGBoost, no PCA** | 512 | **0.2581** | [0.188, 0.352] | **0.99** |
| **our embedding → XGBoost, no PCA** | 1024 | **0.2203** | [0.165, 0.303] | **0.95** |
| **our embedding → XGBoost, no PCA** | 2048 (20 ep)* | **0.2273** | [0.158, 0.310] | **0.94** |

\* The 2048 row uses the 20-epoch checkpoint — the same training budget as the 512/1024 rows. The 40-epoch continuation (the 2048 model in table 2) reads 0.1946 (P=0.78) on this 112-fraud draw yet 0.2665 with tight CIs on the full test period — exactly the single-draw noise the caveat below is about. Artifacts: `downstream/xxl_old_1783532341/` (20 ep) and `downstream/xxl/` (40 ep).

The claim this table supports: the embedding **alone** clears their published *fusion* headline with probability 0.94–0.99 at every context length, at matched 20-epoch training budget. It also shows why this table can't support more than that: with 112 test frauds, even the raw **baseline** exceeds their fusion number in 16% of bootstrap draws — single-point "X% lift" claims on this benchmark (including their +41.8%) deserve intervals around them. We learned this the hard way: re-running our own eval on a different XGBoost device moved the 1024 point by 0.05 while 512 didn't move at all. The 100k set cannot rank our context lengths. So we built an eval that can.

### Table 2 — full test period (2.44M rows, 2,724 frauds): the tight comparison

Same protocol, same trained models, same 1M training rows — the test split is simply *all* test-period transactions instead of a 100k draw. Do not compare these numbers to table 1 or to NVIDIA's published points; within the table, every row is scored on identical rows.

| model | context (txns) | AP | AP 95% CI |
|---|---|---|---|
| raw 13 features + their XGB (baseline) | — | 0.2081 | [0.193, 0.226] |
| their protocol (PCA64+XGB) on our embedding | 512 | 0.1998 | [0.186, 0.216] |
| their protocol (PCA64+XGB) on our embedding | 1024 | 0.1999 | [0.187, 0.213] |
| our embedding → linear head, no PCA | 512 | 0.2129 | [0.204, 0.224] |
| our embedding → linear head, no PCA | 1024 | 0.1213 | [0.115, 0.128] |
| **our embedding → XGBoost, no PCA** | 512 | **0.2788** | [0.262, 0.295] |
| **our embedding → XGBoost, no PCA** | 1024 | **0.3027** | [0.285, 0.320] |
| **our embedding → XGBoost, no PCA** | 2048 (40 ep) | **0.2665** | [0.250, 0.284] |

[B7: figure — dot-and-CI plot of table 2, baseline as reference line.]

What this table resolves:

- **The embedding's lift is CI-separated, not just probable**: embed_xgb at 512 ([0.262, 0.295]) sits entirely above the baseline ([0.193, 0.226]) on identical rows — +34%; at 1024, +45%.
- **Longer context pays — up to a data-dependent peak**: 1024 beats 512 by +0.024 AP, and 2048 falls back below both *even after doubling its training budget* (a 20-epoch continuation to control for undertraining) — 1024's interval is disjoint above 2048's. Paired bootstrap (same resampled rows scored by every model) gives a strict ordering — **1024 > 512 > 2048**: 1024−512 = +0.024 [+0.011, +0.038] (P=1.000), 512−2048 = +0.012 [+0.001, +0.024] (P=0.982), 1024−2048 = +0.036 [+0.022, +0.049] (P=1.000). The peak is what the burst mechanism predicts: most fraud-relevant history sits within ~1024 transactions; beyond that the window adds stale history that dilutes the last-position readout.
- **PCA is where the context advantage dies — until there's nothing left to lose**: their notebook-05 PCA64 step scores *identically* at 512 and 1024 (0.1998 vs 0.1999) and below the raw baseline — the incremental history signal doesn't survive 64 components. Corroborating the dilution story: at 2048, PCA64 catches up to the full embedding (0.270 vs 0.267) — by then there's little beyond 64 components left to destroy. If you take one harness lesson: don't compress a foundation-model embedding before the classifier without measuring what it costs.
- **The linear head collapses at long context** (0.213 → 0.121) while trees improve. The signal is there — XGBoost finds more of it at 1024 than at 512 — but it stops being linearly separable as the window grows. Our best current explanation is in the undertraining section below.

## Why it works — and how we know it's real

### The fraud signal lives in the history, not the transaction

TabFormer fraud is **bursty**: [B8: exact ledgered stat] **90% of test frauds have a prior same-card fraud within the preceding 512 transactions, versus 7.3% of normal transactions.** A model that reads one transaction can never see this; a model that reads 512+ can. That asymmetry is the whole game. [B9: figure — burst illustration or histogram of distance-to-previous-fraud.]

### The strongest ablation we could build: their design, trained by us

The obvious objection: "maybe any transformer trained on this data does well, and NVIDIA just undertrained theirs." We tested exactly that. We trained **their architecture, with their tokenizer, on their recipe**, on the same Ray cluster — and its single-transaction embedding reaches fm-only AP **0.0614** on their protocol (their published weights: 0.0123). Better training helps their design by 5x — and it's still 4x short of our embedding, which reads the target transaction with the full history window behind it. Same data, same capacity, same trainer; the difference is the tokenizer, objective, and readout. [TODO: confirm final numbers + framing with Zach; link appendix.]

### Two controls that could have killed the story

**Shuffled labels**: we permuted the 1M training labels and refit every head. Embedding-only AP collapsed to **0.0016–0.0023**, at the theoretical floor (test prevalence 0.00112) — the harness cannot manufacture the headline without real labels.

**Velocity features**: fraud burstiness suggests skipping the FM and handing XGBoost cheap card-velocity features (counts in 1h/24h/7d windows, spend velocity, time-since-last, amount z-score vs card history — all causal). Result: **AP 0.0757** — they didn't close the gap; under the baseline's tuned params they *halved* raw AP. Counting "how much is happening" isn't the signal; the FM's advantage is knowing **whether this activity is anomalous for this card** — the composition of the burst, not its size. [B10: small table raw13 / raw13+velocity / embed_xgb.]

### The readout is where FM signal goes to die (it happened to us twice)

Early versions of our own model looked *dead* downstream — because an MLM is only trained to make masked positions informative, and we were mean-pooling unmasked states and squashing them through PCA. The fix wasn't a bigger model: per-field masking, reading the **last position**, and dropping PCA took the same architecture from "no signal" to the headline. NVIDIA's pipeline embeds each transaction independently — by design, its embedding represents the transaction itself rather than the card's history. And the same lesson repeated on the recommendation side (below): a readout change alone moved next-merchant HR@10 from 0.08 to 0.40. If you take one modeling lesson from this post: before concluding your foundation model learned nothing, check what state you're reading and what your eval harness can see. [B11: optional figure — the surprise-vector diagnostic anecdote.]

### What long context costs at training time

Doubling the window halves the number of windows per epoch — so at fixed epochs, every positional embedding and every long-range attention pattern gets half the training exposure. You can watch this in TensorBoard: per-field MLM accuracy *falls* with context length (macro 0.824 → 0.770 → 0.754 for 512/1024/2048 at 20 epochs) — the long-context models aren't under-challenged, they're **under-optimized**. The sharpest symptom is a "canary" field: predicting a masked transaction's *month* is nearly free (copy a neighbor's visible month), yet the 512 model only acquires that skill in a late, sharp phase transition — and the longer-context models never reach theirs within budget ([B16: field_ce/month figure across the three runs]). We continued the 2048 model to 40 epochs to separate this from genuine information saturation: pretext skill kept improving (macro 0.754 → 0.767, the month canary still grinding without its transition) — and the downstream verdict came back **dilution, not undertraining**: with double the budget, 2048 still lands below 1024 with disjoint intervals (0.2665 vs 0.3027 on the full test period). Long context is architecturally free in our tokenizer, but on this data its *value* peaks around 1024 transactions. The undertraining effect is real too — it explains the falling pretext accuracy and, we suspect, the linear-head collapse: a less crisply trained long-context readout state is *blurrier* — the signal survives (trees find it) but its linear structure degrades.

## A second consumer: what the embedding knows about the next merchant

The template also ships a next-merchant recommendation eval off the same backbone. This subplot ends with the production answer, via the readout lesson one more time. First the ladder — same frozen embedding, readouts only, 21k-way merchant space:

| readout of the same frozen embedding | HR@10 |
|---|---|
| masked-position readout (the original eval) | 0.077 |
| the model's InfoNCE merchant table, zero-shot | 0.184 |
| trained linear head | 0.397 |
| trained 2-layer MLP (+ hour/day/gap context features) | 0.535 |

A readout change alone moves HR@10 by **7x** — but on data where 96% of transactions repeat a known merchant, a 3-line memorization baseline (the card's historical merchants ranked by count) still scores **0.647**, and the recommendation literature says that's normal: under fair evaluation, frequency baselines routinely beat neural sequence models [refs: next-basket "reality check" / Ludewig & Jannach]. The neural model shouldn't fight memorization — it should *correct* it. The standard production hybrid does exactly that:

| hybrid (identical test events) | HR@10 |
|---|---|
| strongest memorization baseline (causal full-history counts) | 0.6474 |
| **0.1·softmax(MLP) + 0.9·frequency prior (swept on val)** | **0.6582** |

Small margin, unambiguous (paired on 400k events), and the decomposition is where the FM earns its budget: on the **35% of events where the next merchant is NOT in the card's top-10** — where memorization scores exactly 0.000 — the model recovers **HR@10 ≈ 0.16**, and on never-before-seen merchants (6% of events) the full-vocab MLP scores 0.077 where any history-based method scores zero. One backbone: fraud headline, plus a recommender that beats memorization overall *and* covers the slice memorization is mathematically blind to. [B-RECO-FIG: bar chart — overall / not-in-top10 / never-seen, naive vs hybrid.]

## Cost and scale on Anyscale

- 512-context: end-to-end job (data regen → tokenize → 20-epoch pretrain on 4x A10G → 8-GPU extraction → eval) ≈ 2 h. 1024-context: 1 h 50 m measured, ~45 min of it pretraining. 2048-context: ~2.5 h (+ ~2 h for the 40-epoch continuation). [B15: exact $ totals — roughly $15–25 per headline run at on-demand A10G prices.]
- Tokenization is Ray Data `map_groups` over the full 24.4M-row table — the stage NVIDIA runs on a single GPU with RAPIDS; ours scales horizontally and streams into pretraining through the object store.
- The full-test-period eval (3.5M windows tokenized + embedded + scored) runs as one ~1h job per scale on autoscaled 0→8 A10Gs — the eval infrastructure is itself a Ray Data story.
- [B16 alt: TensorBoard curves figure if not used in the undertraining section.]

## Reproduce it in three commands

The template is public — bring an Anyscale account (or your own Ray cluster) and:

```bash
# 1. the gate: reproduce NVIDIA's baseline through this pipeline
anyscale job submit -f job_baseline.yaml
# 2. the headline: pretrain from scratch + print the fraud table
anyscale job submit -f job_full.yaml
# 3. the scaling story: 1024 / 2048 transactions of context
anyscale job submit -f job_xl.yaml     # and job_xxl.yaml
```

`README` in the template covers bring-your-own-data (the CSV columns you need), hardware knobs, and the full-test-period eval (`scripts/fulltest_eval.py`). [TODO: template link once published.]

## Takeaways

- **A transaction FM's embedding, alone, beats NVIDIA's published fusion headline** on their exact protocol (P = 0.94–0.99 across 512–2048 context at matched training budget), and beats the identical-rows raw baseline by **+34–45% with non-overlapping CIs** on the full 2.44M-transaction test period.
- **Context is architecturally free, pays measurably, and peaks**: one position per transaction buys 4–13x their context at identical capacity; on the tight eval 1024 beats 512 (+0.024 AP) and 2048 falls back even at double training budget — the sweet spot tracks the data's burst structure (~1024 transactions here), and finding yours is a config change, not an architecture change.
- **The readout matters as much as the model**: per-field masking + last-position extraction + no PCA is the difference between "the FM learned nothing" and the headline — a lesson that repeated on the recommendation task (0.08 → 0.54 from readout changes alone, and past the memorization baseline once blended with the frequency prior). PCA-before-classifier destroyed the context advantage entirely in our measurements.
- **Benchmark like you mean it**: exact baseline reproduction, shuffled-label control at the AP floor, a velocity-feature bar that doesn't reach, bootstrap CIs on a 112-fraud test set that nearly fooled us twice, and a full-test-period eval that finally resolves what the sample couldn't.
- **What we didn't run**: we evaluate NVIDIA's model through its shipped pipeline only. A history-aware readout of their architecture would plausibly improve it too — that's the readout lesson applied to them — but at ~12 tokens per transaction, their 4096-token window caps readable history at ~315 transactions, well short of the ~1024 where our context curve peaks.
- Disclosures: one dataset (TabFormer, static-poor); the 2048 model's training used a warm-restarted continuation; linear-head readout degrades at long context while trees improve; full-period and 100k-sample APs are not mutually comparable.

## What's next

Longer contexts with training budgets that keep per-position exposure constant; the frequency-blend recommender; richer static fields than TabFormer carries. Run the template, swap in your transactions, and tell us what your context-length curve looks like.

[CTA block: run the template on Anyscale / talk to us / Ray Slack.]
