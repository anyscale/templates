# Design notes — Transaction Foundation Model on Ray

> Internal design doc (not part of the published template). Captures the "why"
> behind the structure so reviewers can sanity-check the direction before
> reading code. Safe to delete before publish.

## Thesis

Take NVIDIA's transaction-foundation-model blueprint (a vanilla decoder trained
next-token on tokenized transaction sequences, where the cleverness is the
tokenizer — not the architecture) and rebuild it **production-scale on Ray**,
with one architectural upgrade that makes it *more interesting than the NVIDIA
recipe* without becoming a pure research artifact:

- **Static / dynamic field split** in the tokenizer + embedding layer
  (the idea crystallized in FATA-Trans and productionized in Visa's TREASURE).
  Card-level fields that never change within a sequence (issuer, card type, BIN
  region) are embedded **once per card**; per-transaction fields (amount,
  merchant, MCC, time deltas) are tokenized **per event**. This is both cheaper
  and empirically stronger than flattening every field into the token stream
  NVIDIA-style.

We do **not** chase HSTU. HSTU is the right answer for Meta's regime (10^5-length
sequences, billion-scale dynamic vocab, batch-bound training). Transactions
don't have those properties; a small decoder + a thoughtful tokenizer wins.

This is **variant B only** — we build the static/dynamic model end-to-end and
*reference* the NVIDIA flat-token numbers rather than shipping a second model.
A flat-token baseline toggle is left as a documented extension point in
`tokenizer.py` (`SPLIT_FIELDS = False`) so the A/B story is available for the
blog post without doubling the template's surface area.

## Audience

Fintech MLEs. Code-first, reproducible, runs on synthetic data out of the box,
points at real datasets for the next step. Every stage answers "what does Ray /
Anyscale give me here that my current stack doesn't."

## What Ray / Anyscale earns at each stage

| Stage | Ray primitive | The win |
|-------|---------------|---------|
| Tokenize | **Ray Data** `map_batches` | Distributed, streaming tokenization over petabyte-scale Parquet. NVIDIA's RAPIDS path is single-GPU and breaks past one node. |
| Pretrain | **Ray Train** + PyTorch FSDP | Clean multi-node data-parallel/FSDP, fault tolerant, same code dev→prod. |
| Batch embeddings | **Ray Data** + model actor | Score every customer periodically — heterogeneous CPU(read)+GPU(infer) in one pool, streaming, idempotent writes. The piece with no clean public reference. |
| Online embeddings | **Ray Serve** | Real-time per-transaction embedding + fraud score, with cached static-field embeddings. |
| Observability | Ray Dashboard / Anyscale Metrics | Watch stages stream, per-worker throughput, autoscaling, cost. |

## Serving — both paths, on purpose

The honest production pattern is **batch-embed → feature store → XGBoost**, and
that's the default demo path. But fintech MLEs always ask "what about real time?"
so we also ship a **Ray Serve online deployment** that:
- caches static (card-level) embeddings,
- runs the transformer only on the recent dynamic window,
- returns an embedding + fraud probability in one call.

This mirrors the two-tier pattern real shops use (small model real-time / bigger
model batch, e.g. Revolut PRAGMA) without pretending we run a 1B model on the
hot path.

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
│   ├── tokenizer.py          # static/dynamic split tokenizer (Ray Data UDFs)
│   ├── model.py              # TransactionFM: split embeddings + decoder + MLM head
│   ├── pretrain.py           # Ray Train + FSDP masked-feature-modeling loop
│   ├── embed.py              # Ray Data batch embedding extractor (callable class)
│   ├── downstream.py         # XGBoost on embeddings + raw-feature baseline
│   ├── serve.py              # Ray Serve online embedding/fraud deployment
│   └── utils.py
└── scripts/
    ├── 01_generate_data.py
    ├── 02_tokenize.py
    ├── 03_pretrain.py
    ├── 04_extract_embeddings.py
    ├── 05_train_downstream.py
    └── validate_results.py
```

## Dataset

Synthetic generator ships in-repo (no license friction, runs in CI). Schema
mirrors **IBM TabFormer** (the de-facto public benchmark): per-card static fields
+ per-transaction dynamic fields + a planted fraud label with several realistic
patterns. README points at TabFormer / Sparkov / IBMCard for the real next step.

## Pretraining objective

**Masked feature modeling (MLM)** on dynamic-field tokens — bidirectional context
is better for the fixed-window embedding tasks fintech cares about (fraud, churn,
credit). NTP is mentioned as the swap-in for generative use. Embedding = mean of
final-layer dynamic-token states (+ the static embedding), which is what
downstream consumes.

## Eval

Downstream fraud classification with XGBoost on three feature sets, reported
side by side:
1. raw hand-crafted features (the "what you have today" baseline),
2. FM embedding only,
3. joint fusion (embedding ++ raw features) — Nubank's recipe.

AUC-ROC / PR-AUC lift of (2) and (3) over (1) is the headline result.

## Scale knobs

`--scale {smoke,small,medium}` everywhere. `smoke` trains a ~1–2 layer / hidden-128
model on a few thousand sequences in a couple minutes on CPU (so papermill CI
passes without a GPU). `medium` is the credible distributed story on GPU workers.
