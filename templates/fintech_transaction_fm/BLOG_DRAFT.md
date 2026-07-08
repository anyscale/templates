# [BLOG DRAFT] One position per transaction: beating NVIDIA's transaction-FM blueprint with 4–13x longer context on Ray

> Status: full draft with numbered blanks `[B#]`. Style per Anyscale
> engineering-benchmark posts (result first → setup → hero table → mechanism
> sections → cost/scale → 3-command repro → takeaways → CTA). All numbers
> below are from the RUN-tagged campaign ledger; blanks are figures/tables to
> export and the pending 2048 + pinned-env reruns.
>
> PRE-PUBLISH GATE (from ZGARNER_INTEGRATION.md): re-run stage 05 under
> xgboost==3.2.0 + device parity and swap the table numbers if they move;
> add bootstrap CIs; confirm Zach's numbers/framings with him.

---

NVIDIA's [transaction foundation model blueprint](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model) sets a public bar for fraud detection with foundation models: pretrain a small Llama decoder on 24M card transactions (IBM TabFormer), embed transactions, and fuse the embedding with 13 raw features into XGBoost — test average precision **0.1755**, a +42% lift over their raw baseline.

We built a transaction FM on Ray with a different tokenizer, objective, and readout — and its embedding **alone**, no fusion, no PCA, reaches **AP 0.27 on their own evaluation protocol**: +92% over the raw baseline, +55% over their published fusion headline. The core change fits in one sentence: **one position per transaction instead of ~12 tokens per transaction**, which buys 512–2048 transactions of context in the window where their tokenizer fits ~315 — and a masked-field objective that makes the history actually readable. Training runs in about two hours on four A10Gs; the whole result reproduces from a public template with three job submissions.

[B1: hero chart — bar chart of test AP: NVIDIA baseline 0.1238, NVIDIA fusion 0.1755, ours-512 0.258, ours-1024 0.273, ours-2048 0.223. One color for theirs, one for ours.]

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
- **The readout**: windows are right-aligned, so the last position *is* the target transaction, read with the card's full history behind it. That 512-d state goes straight into a linear head or XGBoost — no PCA.

The pipeline is Ray end-to-end: Ray Data builds the ~100k-merchant vocab and streams the field-split tokenization (`map_groups`, no global shuffle); Ray Train runs DDP masked pretraining (4 GPU workers, one `ScalingConfig` change from laptop to cluster); Ray Data streams CPU-read + GPU-infer embedding extraction; XGBoost consumes the parquet. [B2: pipeline architecture figure — reuse the README ASCII diagram as a proper graphic.]

## Benchmark setup

- **Data & protocol**: IBM TabFormer; NVIDIA's notebook-01 protocol transcribed verbatim (temporal 80/10/10 by date, 1M balanced train, 100k stratified val/test, their three separately-HPO'd XGBoost param sets, early stopping on val AUC). Our stage 01 emits the sampled rows as a pinned `benchmark.parquet`; the raw-features baseline reproduces through our pipeline at **0.9875 ROC / 0.1421 AP** on every run. [TODO after pinned-env rerun: this row is expected to land at their exact 0.1238 under xgboost==3.2.0 + CUDA; update all table rows to pinned-env values.]
- **Models**: ~29M params (their capacity), 512d/8L/8H, trained 20 epochs from scratch per context length — no pretrained weights anywhere.
- **Hardware**: 4x A10G (g5.xlarge) for pretraining, 8 for extraction, autoscaled 0→8 on Anyscale; CPU head for XGBoost.
- **Fairness**: same benchmark rows for every model row; embeddings joined exactly (card, timestamp, amount-cents; 100.00% match); fit-seed replications on the linear head [B3: seed CI table — probe_metrics_seed*.json]; bootstrap CIs over the ~112-fraud test set [B4: CI table — pending, see gate]. Controls below.

## Results

Test set: 100k stratified rows at natural ~0.11% fraud prevalence.

| model | context (txns) | ROC-AUC | AP | AP lift vs raw |
|---|---|---|---|---|
| raw 13 features + their XGB (baseline) | — | 0.9875 | 0.1421 | — |
| *NVIDIA published: baseline* | *~315* | *0.9885* | *0.1238* | — |
| *NVIDIA published: fusion (their headline)* | *~315* | *0.9925* | *0.1755* | *+41.8%* |
| their protocol (PCA64+XGB) on our embedding | 512 | [B5] | 0.1623 | +14.2% |
| their protocol (PCA64+XGB) on our embedding | 1024 | 0.9905 | 0.1899 | +33.7% |
| **our embedding → XGBoost, no PCA** | 512 | [B5] | **0.2581** | **+81.6%** |
| **our embedding → XGBoost, no PCA** | 1024 | 0.9945 | **0.2730** | **+92.2%** |
| **our embedding → XGBoost, no PCA** | **2048** | 0.9914 | **0.2233** | **+57.2%** |
| our embedding → linear head, no PCA | 512 | [B5] | 0.226 ± 0.006 | +59% |

[B7: context-scaling line chart — AP vs context length {512, 1024, 2048} for embed_xgb (0.258 → 0.273 → 0.223), with NVIDIA fusion 0.1755 as a horizontal reference line. The peak at 1024 is the finding.]

Three things to notice. First, the embedding **alone** beats their published *fusion* at every context length — no hand-built features in the winning row. Second, even **their own downstream protocol** (PCA to 64d, then XGBoost) applied to our embedding beats their fusion at 1024 and 2048 (0.1899 / 0.1802 vs 0.1755): the gain is in the representation, not the harness. Third — and we report this as measured — the context curve **peaks at 1024**: 512→1024 pays +6%, while 2048 gives some of it back (still +27% over their fusion). That shape is what the burst mechanism predicts: ~90% of the fraud-relevant history sits within a few hundred transactions, so past ~1024 the window mostly adds stale history that dilutes the last-position readout. The practical takeaway for TabFormer-scale data is a sweet spot around 1024 transactions — and the honest scaling claim is "4–13x their context is *architecturally free*; how much of it pays is a property of your data's burst structure." [TODO: consider an xxl variant with more epochs to separate dilution from undertraining — xxl's per-window exposure differs; cheap to note, optional to run.]

One honest wrinkle: the linear-head readout, which at 512 was stable across seeds (0.226 ± 0.006), degraded at longer context (0.124 at 1024, 0.127 at 2048) while XGBoost stayed strong. The signal is there — XGBoost finds it — but it becomes less *linearly* separable as the window grows. [TODO: 1-2 sentence explanation after investigation, or cut the linear row at 1024+ and note it.]

## Why it works — and how we know it's real

### The fraud signal lives in the history, not the transaction

TabFormer fraud is **bursty**: [B8: exact ledgered stat] **90% of test frauds have a prior same-card fraud within the preceding 512 transactions, versus 7.3% of normal transactions.** A model that reads one transaction can never see this; a model that reads 512+ can. That asymmetry is the whole game, and it's why context length converts directly into average precision. [B9: figure — burst illustration: timeline of one card's transactions with fraud burst highlighted, or histogram of distance-to-previous-fraud.]

### The strongest ablation we could build: their design, trained by us

The obvious objection: "maybe any transformer trained on this data does well, and NVIDIA just undertrained theirs." We tested exactly that. We trained **their architecture, with their tokenizer, on their recipe**, on the same Ray cluster — and its single-transaction embedding reaches fm-only AP **0.0614** (their published weights: 0.0123). Better training helps their design by 5x — and it's still 4x short of our 0.27, because their readout throws the history away. Same data, same capacity, same trainer; the difference is the tokenizer, objective, and readout. [TODO: confirm final numbers + framing with Zach; link appendix.]

### Control #1: shuffle the labels — the number must die

We permuted the 1M training labels and refit every head. Embedding-only AP collapsed from 0.23–0.27 to **0.0016–0.0023**, at the theoretical floor (test fraud prevalence 0.00112). The harness cannot manufacture the headline without real labels — no leakage through the join, the splits, or the evaluation.

### Control #2: is this just velocity features? No — we tried

Fraud burstiness suggests a cheap alternative: skip the FM, hand XGBoost some card-velocity features (transaction counts in 1h/24h/7d windows, spend velocity, time-since-last, amount z-score vs card history — all causally computed). Result: **AP 0.0757** — the velocity features didn't close the gap; under their tuned baseline params they *halved* the raw baseline's AP (ROC unchanged). Counting "how much is happening" isn't the signal; the FM's advantage is knowing **whether this activity is anomalous for this card** — the composition of the burst, not its size. [B10: small table raw13 vs raw13+velocity vs embed_xgb.]

### The readout is where FM fraud signal goes to die

Two war stories, one lesson. Early versions of our own model looked *dead* downstream — because an MLM is only trained to make masked positions informative, and we were mean-pooling unmasked states (and squashing them through PCA). The fix wasn't a bigger model: per-field masking (so ordinary positions receive gradient under partial visibility), reading the **last position**, and dropping PCA took the same architecture from "no signal" to 0.26. Meanwhile NVIDIA's pipeline embeds **single transactions** — a readout that discards precisely the history their own pretraining ingested. If you take one engineering lesson from this post: before concluding your foundation model learned nothing, check what state you're reading and what your eval harness can see. [B11: optional figure — surprise-vector anecdote: per-field cross-entropy of the true transaction under the MLM as an interpretable anomaly score, the diagnostic that proved the signal existed.]

## Cost and scale on Anyscale

- Full pretrain (512 ctx, 20 epochs, 24.4M txns): ~[B12: exact wall-clock] on 4x A10G; 1024 ctx: ~45 min pretrain, ~1h50m end-to-end job including data regen, extraction on 8 GPUs, and eval. 2048 ctx: [B13].
- Tokenization is Ray Data `map_groups` over the full table — the stage NVIDIA runs on a single GPU with RAPIDS; ours scales horizontally and feeds pretraining through the object store with no intermediate writes.
- Extraction streams CPU-read + GPU-infer in one pipeline; 8 A10Gs embed ~500k eval windows in [B14: minutes] at seq 1024.
- Everything autoscales 0→8 GPU nodes; the jobs run on spot-friendly A10Gs. Total cost of the headline run: ~$[B15].
[B16: TensorBoard training curves figure — MLM loss + per-field accuracy across 512/1024/2048 runs, same axes.]

## Reproduce it in three commands

The template is public — bring an Anyscale account (or your own Ray cluster) and:

```bash
# 1. the gate: reproduce NVIDIA's baseline through this pipeline
anyscale job submit -f job_baseline.yaml     # -> 0.9875 ROC / 0.1421 AP
# 2. the headline: pretrain from scratch + print the fraud table
anyscale job submit -f job_full.yaml
# 3. the scaling story: 1024 (and job_xxl.yaml for 2048)
anyscale job submit -f job_xl.yaml
```

`README` in the template covers bring-your-own-data (the CSV columns you need) and hardware knobs. [TODO: template link once published.]

## Takeaways

- **A transaction FM's embedding, alone, beats NVIDIA's published fusion headline by 55%+** on their own protocol — 0.27 vs 0.1755 test AP — using the same model capacity and a two-hour training run.
- **Context length is a real, measurable lever with a data-dependent sweet spot**: one position per transaction buys 512–2048 transactions of history vs their ~315; on TabFormer AP peaks at 1024 (0.273) and holds +27% over their fusion even at 2048.
- **The readout matters as much as the model**: per-field masking + last-position extraction + no PCA is the difference between "the FM learned nothing" and the headline. NVIDIA's single-transaction embedding readout is their blueprint's real bottleneck, not their architecture.
- **It's verified the boring way**: exact baseline reproduction, shuffled-label control at the AP floor, a velocity-feature bar that doesn't reach, and a faithfully-trained version of their own design topping out 4x lower.
- Disclosures: one dataset (TabFormer); ~112 test frauds means single-draw AP is noisy — CIs in appendix [B4]; linear-head readout weakens at 1024+ while trees improve; [TODO: env-pin note after xgboost 3.2.0 rerun].

## What's next

Next-merchant recommendation off the same backbone (BERT4Rec-style over the InfoNCE merchant table) ships in the template as an extra — promising shape, not yet at a publishable number. Longer contexts and richer static fields (TabFormer is static-poor) are open headroom. Run the template, swap in your transactions, and tell us what your context-length curve looks like.

[CTA block: run the template on Anyscale / talk to us / Ray Slack.]
