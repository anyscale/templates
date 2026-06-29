# Design notes — Transaction Foundation Model on Ray

> Internal design doc (not part of the published template). Captures the "why"
> behind the structure and the honest goals/limitations so reviewers can
> sanity-check the direction before reading code. Safe to delete before publish.

## Blog thesis

**Foundation models are becoming the backbone for both recommendations and
fraud — and Ray + Anyscale are how you actually get there.**

One self-supervised transformer, pretrained on raw transaction sequences,
produces a reusable customer/transaction embedding that feeds *two* downstream
consumers off the same backbone:

1. **Fraud** — embedding (± raw features) → XGBoost. The NVIDIA-blueprint story.
2. **Recommendation** — next-merchant ranking (HR@K / NDCG@K). The story NVIDIA's
   blog doesn't tell at all.

The model is deliberately small and boring; the value and the engineering are in
**distributed tokenization + vocab construction, distributed pretraining, batch
embedding, and serving** — exactly the Ray/Anyscale story. "One backbone, two
consumers, distributed end-to-end" is a strict superset of NVIDIA's single-task,
single-node recipe.

## Goals (honest)

1. **A blog post** with a credible, reproducible result on a public benchmark
   (IBM TabFormer), showing one FM backbone serving fraud *and* recommendation.
2. **An Anyscale template** fintech MLEs can run end-to-end and adapt.
3. **Showcase Ray/Anyscale strengths NVIDIA's blog can't match:** streaming
   distributed tokenization, distributed high-cardinality vocab construction,
   DDP multi-GPU training (FSDP as the scale-up flag), heterogeneous CPU+GPU
   batch embedding, and online serving.

**On "beat NVIDIA":** the *durable* win is structural — a distributed pipeline
their single-node RAPIDS path can't follow at scale, plus a second (reco) task
they don't do. Beating their *fraud number* is a stretch goal, not a guarantee:
it hinges on the network-signal head + channel-as-input + fusion (see below) and
must be shown with an honest ablation, not asserted. If we match their fraud lift
and add recommendation on the same backbone, that is already a win.

## Where this actually sits (lineage — be precise)

The earlier draft of this doc called this "NVIDIA's blueprint + one upgrade,
inspired by TREASURE." That was imprecise. The accurate attribution:

| Piece | Source | In the template? |
|-------|--------|------------------|
| Static/dynamic field split | **FATA-Trans** (CIKM '23); TREASURE productionizes it | ✅ core |
| Time-aware positional embedding | **FATA-Trans** | ✅ core |
| Masked-feature (MLM) pretraining | **FATA-Trans** (15% static + 15% dynamic) | ✅ core |
| End-to-end shape (tokenize → pretrain → embed → XGBoost fraud, TabFormer, temporal split) | **NVIDIA blueprint** (baseline to beat) | ✅ protocol |
| InfoNCE high-cardinality loss + shared negatives | **TREASURE** (Alg. 1) | ➕ adding now |
| Network-signal modeling (response/decline codes) | **TREASURE** | ➕ adding now |
| Masked-modeling *as* sequential recommendation | **BERT4Rec** (mask last item, rank) | ➕ adding now |
| Joint fusion (embedding ++ raw features) downstream | **Nubank** | ✅ done |

So: **the architecture we built is FATA-Trans, on Ray.** NVIDIA is the baseline
and the eval protocol. TREASURE is the production north-star and the source of
the two pillars we're now adding (InfoNCE, network signals). Naming it
"FATA-Trans on Ray, scaling toward TREASURE" is the honest framing for the blog.

**Why MLM and not autoregressive (NVIDIA/TREASURE are causal)?** Two reasons,
both now defensible by citation rather than hand-waving: (1) FATA-Trans is MLM,
so we're consistent with our architecture anchor; (2) MLM is *not* a compromise
for recommendation — **BERT4Rec** shows masked-item prediction is SOTA-competitive
with (often beats) causal SASRec for sequential rec. So one MLM objective serves
both heads. We still owe the blog one ablation: AR vs MLM on the same tokenizer,
reported honestly.

We do **not** chase HSTU (right for Meta's 10^5-length / billion-vocab regime,
wrong for transactions).

## The two-headed backbone (what we're building this round)

```
raw transactions (Parquet/S3)
   → [Ray Data]  distributed merchant-vocab construction (freq count, top-K + tail aggregation)
   → [Ray Data]  static/dynamic tokenization                 (map_groups)
   → [Ray Train] masked-feature pretraining (DDP)            ← InfoNCE for high-card fields
                                                              ← network-signal head (Errors?)
   → [Ray Data]  batch embedding extraction (CPU read + GPU infer)
   → ├─ [XGBoost]   fraud: raw vs FM vs fusion                (AUC-ROC / PR-AUC)
     └─ [rank]      recommendation: next-merchant             (HR@K / NDCG@K)
   → [Ray Serve] online embedding + fraud + next-merchant
```

### 1. InfoNCE high-cardinality loss (TREASURE Alg. 1)

TabFormer has **100,343 unique merchants** (verified, see de-risk below). A
full-softmax MLM head over the real vocab is heavy (≈100k-row head, multi-GB of
logits at seq 512); at Visa's 150M it's impossible. InfoNCE computes the logit
for the positive + a pool of **shared negatives** (sampled once per batch, reused
across all timesteps and samples) — the memory trick that makes high-card
training tractable. We implement it for fields above the **1024 cardinality
threshold** (the paper's number) — merchant qualifies; everything else keeps
plain cross-entropy. The merchant InfoNCE head is tied to the merchant embedding
table (logits = hidden · E[candidates]ᵀ).

Honest caveat: at TabFormer's 100k, a full softmax is actually *feasible*, so
InfoNCE isn't strictly *necessary* here. We include it because (a) it's the same
code path that scales to 150M, and (b) we can reproduce TREASURE's Fig. 6 memory
curve (shared vs independent negatives) on our own model as a concrete blog
figure. We do **not** claim InfoNCE materially improves the *fraud* number — it
helps the *merchant/reco* head far more than fraud.

### 2. Learned merchant vocab + long-tail aggregation (Ray Data)

InfoNCE is pointless on a hashed 2000-bucket merchant field, so the `learned`
path drops the hash and builds a real vocab: a **distributed frequency count**
(Ray Data), keep the **top-K** merchants, fold the long tail + inference-time OOV
into a small set of **shared aggregate buckets** — exactly TREASURE's "infrequent
or new entities are mapped to shared aggregated identifiers." De-risk numbers say
top-10k merchants cover 95.5% of transactions, so K≈10–20k is the sweet spot.

This changes the tokenizer's selling point from "stateless, no global
aggregation" to "**one distributed vocab pass, then stateless map**" — still a
great Ray Data story (distributed vocab construction over a huge table is its
wheelhouse), just messaged accurately.

**Config-gated:** `merchant_vocab: hashed | learned`. `hashed` stays the CI/smoke
default (CPU can't afford a 100k-row head); `learned` + InfoNCE runs at
`small`/`full`. Keeps CI green; the blog runs the big version.

### 3. Network-signal head + channel-as-input (TREASURE's distinguishing pillar)

TabFormer carries two payment-network fields the loader currently **discards**.
The de-risk shows both are strongly fraud-predictive:

* **`Use Chip` (channel: swipe/chip/online)** — known at auth time, varies per
  transaction, and Online txns have **16× the fraud rate** of Swipe (0.68% vs
  0.043%). It's a legitimate **dynamic input field**. The current loader wrongly
  collapses it to a *static per-card modal proxy*, throwing the signal away. Fix:
  promote channel to a per-transaction dynamic field.
* **`Errors?` (decline/response codes: Insufficient Balance, Bad PIN, …)** —
  known only *after* processing (2.7× fraud lift). Following TREASURE, this is an
  **output-only network-signal prediction head**, never an input (as an input it
  would leak the label). Predicting the current transaction's signal is the
  literal thing TREASURE says makes it "first to holistically model
  transactions," and it's a plausible lever on our fraud number.

### 4. Recommendation eval — next-merchant (BERT4Rec-style)

Mask the target position's merchant, rank candidates with the InfoNCE-tied
merchant embedding table, report **HR@K / NDCG@K**. Runs on **TabFormer only** —
the synthetic generator draws `merchant_id` as a uniform random int with no
repeat structure, so reco there is meaningless. The *same* batch-embedding job
feeds both fraud and reco: the literal "one backbone, two consumers" picture.

## De-risk findings (TabFormer, full 24.4M-txn dataset)

Verified before committing to the plan (`scratchpad/derisk.py`):

* **Cardinality:** 100,343 unique merchants, 109 MCC, 6,139 cards. Top-1k
  merchants cover 80% of txns; top-10k cover 95.5%; top-50k cover 99.6%.
  → InfoNCE + top-K-with-tail-aggregation is the right call.
* **Merchant repeat structure:** **96.0%** of transactions are at a merchant the
  card has seen before; only 978k distinct (card, merchant) pairs. Naive "top-K
  historical merchants" baseline: top-1 = 18.3%, top-5 = 48.3%, top-10 = 65.2%,
  top-20 = 79.7%. → strong next-merchant signal *with* headroom. Reco is viable.
* **Network signals:** `Errors?` present on 1.59% of txns, 2.7× fraud lift.
  `Use Chip`: Online 0.68% fraud vs Swipe 0.043% (~16×). → both worth modeling;
  channel is currently discarded.
* **Fraud prevalence:** 0.122% transaction-level (matches NVIDIA's setup).

## Honest limitations

* **TabFormer is static-poor.** It has no issuer / BIN / card-product; the loader
  sets `issuer`/`bin_region` to `"UNKNOWN"` and derives `home_state`/`card_type`
  as modal proxies. So the static/dynamic split's "embed rich card attributes
  once" benefit is *muted* on TabFormer — the position-count win (N vs ~12N) is
  real, the static-richness win is not. **Mitigation:** use the **synthetic**
  generator (which has real issuer/bin_region/card_type/home_state) to
  demonstrate the static/dynamic architecture benefit cleanly, and TabFormer for
  the NVIDIA-comparable fraud number + the reco story.
* **Reco only on TabFormer** (synthetic merchants are random — no signal).
* **InfoNCE necessity is a scale argument, not a TabFormer one** (see above) —
  frame it honestly.
* **Beating NVIDIA's fraud number is not guaranteed** — see Goals.

## What Ray / Anyscale earns at each stage

| Stage | Ray primitive | The win |
|-------|---------------|---------|
| Vocab | **Ray Data** (freq count / groupby) | Distributed high-cardinality vocab construction over the full table — the precondition for InfoNCE, NVIDIA's single-GPU path can't do at scale. |
| Tokenize | **Ray Data** `map_groups` | Streaming static/dynamic tokenization over petabyte-scale Parquet. NVIDIA's RAPIDS path is single-GPU and breaks past one node. |
| Pretrain | **Ray Train** (DDP; FSDP flag) | Same code 1→N GPUs by changing `ScalingConfig`. DDP is correct at our ~29M scale; FSDP is the one-flag switch when scaling toward TREASURE's 400M–1.8B. |
| Batch embeddings | **Ray Data** + model actor | Score every customer; heterogeneous CPU(read)+GPU(infer) in one pool, streaming. One job feeds *both* fraud and reco. |
| Online | **Ray Serve** | Real-time embedding + fraud score + next-merchant, with cached static-field embeddings. |
| Observability | Ray Dashboard / Anyscale Metrics | Watch stages stream, per-worker throughput, autoscaling, cost. |

## Serving — both paths, on purpose

Default = **batch-embed → feature store → XGBoost / candidate ranking** (what
shops actually do; the transformer is not on the hot path). Also ship a **Ray
Serve** online deployment that caches static (card-level) embeddings, runs the
transformer only over the recent dynamic window, and returns embedding + fraud
score + next-merchant in one call — the two-tier pattern (small model real-time /
bigger model batch, e.g. Revolut PRAGMA).

## Field representation choices

**Time-aware positions (implemented, FATA-Trans).** Keep the learned ordinal
position embedding *and* add a learned embedding of the **log-bucketed hours
since the previous transaction** (`time_bucket`), summed into each position.
Upgrade paths: Time2Vec (continuous), or a relative-time attention bias
(FATA-Trans/HSTU — needs a custom attention layer).

**Channel (`Use Chip`) — dynamic input field (adding).** Known at auth time,
16× fraud signal, varies per transaction. Promote from the current static modal
proxy to a per-event dynamic categorical field.

**Amount representation (bucketing default, pluggable).** Money is heavy-tailed;
we log-bucket into a categorical token. Spectrum, worst→best: (1) raw log1p+
z-score scalar via `Linear(1,d)` — weak, forks MLM into regression; (2) hard
buckets (current default); (3) soft binning (interpolate adjacent bin embeddings);
(4) learned numerical embeddings — PLE or periodic/Fourier (Gorishniy et al.).

## Pretraining objective

**Masked feature modeling (MLM)** on dynamic-field tokens (15% mask, FATA-Trans),
bidirectional. High-cardinality fields (merchant) use **InfoNCE**; low-card
fields use cross-entropy. Plus an **output-only network-signal head** (Errors?).
Embedding for downstream = pooled final-layer states (last for per-transaction
tasks, mean for whole-history). MLM doubles as the reco objective (BERT4Rec).

## Eval

**Fraud** (NVIDIA protocol): temporal 80/10/10 split, per-transaction last-event
labels, AUC-ROC + PR-AUC at natural prevalence (downsampled normals
importance-weighted back). Three feature sets side by side: raw / FM / fusion.

**Recommendation** (TabFormer): next-merchant HR@K / NDCG@K, BERT4Rec-style,
scored via the InfoNCE merchant table. Naive top-K-historical baseline reported
alongside (floor to beat: HR@10 ≈ 65%).

## Scale knobs & config flags

`--scale {smoke,small,full}` everywhere. `smoke` = ~1–2 layer / hidden-128 on a
few thousand sequences, CPU, hashed merchant (papermill CI, no GPU). `full` =
~29M params (NVIDIA capacity parity), `learned` merchant vocab + InfoNCE,
multi-GPU DDP. Key flags: `merchant_vocab: hashed|learned`, `merchant_top_k`,
`infonce_negatives`, `amount_mode: hard|soft`, `use_fsdp`.

## Pipeline / file tree

```
fintech_transaction_fm/
├── README.md                 # notebook-style walkthrough (the shipped artifact)
├── DESIGN.md                 # this file (delete before publish)
├── requirements.txt
├── job_config.yaml           # end-to-end as an Anyscale Job
├── src/
│   ├── paths.py              # demo base dir resolution (cluster storage vs local)
│   ├── generate_data.py      # synthetic TabFormer-like transactions → Parquet
│   ├── tabformer.py          # real TabFormer loader (Ray Data)
│   ├── merchant_vocab.py     # distributed vocab build + long-tail aggregation (NEW)
│   ├── tokenizer.py          # static/dynamic split tokenizer (Ray Data UDFs)
│   ├── model.py              # TransactionFM: split embeddings + encoder + MLM/InfoNCE/signal heads
│   ├── pretrain.py           # Ray Train (DDP/FSDP) masked-feature-modeling loop
│   ├── embed.py              # Ray Data batch embedding extractor (callable class)
│   ├── downstream.py         # XGBoost fraud: raw vs FM vs fusion
│   ├── recommend.py          # next-merchant HR@K / NDCG@K eval (NEW)
│   ├── serve.py              # Ray Serve online embedding/fraud/next-merchant deployment
│   └── utils.py
└── scripts/
    ├── 01_generate_data.py
    ├── 02_tokenize.py
    ├── 03_pretrain.py
    ├── 04_extract_embeddings.py
    ├── 05_train_downstream.py
    ├── 06_recommend.py       # (NEW)
    └── validate_results.py
```
