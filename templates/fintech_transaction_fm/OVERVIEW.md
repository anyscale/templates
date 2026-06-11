# Transaction Foundation Model — design overview & reading list

A primer for designing the transaction-FM template + blog. Read this instead of
the original brainstorm. ~15 min of reading; the starred papers are the must-reads.

---

## TL;DR

Build a **transaction foundation model** (TFM): one self-supervised transformer
pretrained on raw transaction sequences that emits a reusable **customer
embedding**, which downstream models (fraud, churn, credit) consume instead of
hand-built features. The model is small and boring on purpose — the value and
the work are in the **data engineering, distributed pretraining, and serving**,
which is exactly the Ray/Anyscale story.

Our one differentiator over the off-the-shelf NVIDIA recipe: a **static/dynamic
field split** in the tokenizer and embedding layer.

---

## The pipeline

```
raw transactions (Parquet/S3)
   → [Ray Data]  static/dynamic tokenization        (map_groups, no shuffle)
   → [Ray Train] masked-feature pretraining          (PyTorch + FSDP)
   → [Ray Data]  batch embedding extraction           (CPU read + GPU infer)
   → [XGBoost]   downstream fraud: raw vs FM vs fusion (the headline result)
   → [Ray Serve] online embedding + fraud score        (cached static embeddings)
```

Same code laptop → multi-node; you change `ScalingConfig`, not the program.

---

## Key design decisions

**Copy from NVIDIA's blueprint (the scaffolding):**
- A **plain decoder/encoder transformer trained from scratch** on tokenized
  transactions — *not* a custom architecture. NVIDIA proves a stock ~29M model +
  a thoughtful tokenizer gets you ~80% of the way.
- The **end-to-end shape**: tokenize → pretrain (next-token in their case) →
  take a hidden state as the customer embedding → feed XGBoost for fraud → show
  lift over hand-crafted features. We keep this spine.
- Their **TabFormer-based demo** structure (small enough to run, big enough to be
  credible).

**Change / add (what makes us better than off-the-shelf):**
- **Static/dynamic field split** (from Visa TREASURE & FATA-Trans). Card-level
  fields that never change (issuer, card type, BIN region, home state) are
  embedded **once** and broadcast to every position; per-transaction fields
  (amount, merchant, MCC, time) each get their own embedding table and occupy
  **one** position per transaction. A sequence of N transactions is N positions,
  not ~12N — cheaper and a stronger inductive bias.
- **Masked-feature modeling (MLM)** instead of next-token. Bidirectional context
  is better for the fixed-window classification tasks fintech actually runs
  (fraud/churn/credit). Keep next-token only if you want generation.
- **Replace the single-GPU RAPIDS tokenizer with Ray Data**, single-node training
  with **Ray Train + FSDP**, and add a **Ray Data batch embedding** job — the
  piece with no clean public reference and the clearest Ray win.
- **Time-aware positions** — ordinal slot is the wrong signal for transactions;
  the inter-event *gap* matters. We embed the log-bucketed hours-since-previous
  transaction and add it alongside the ordinal position. (Upgrades: Time2Vec, or
  a relative-time attention bias à la FATA-Trans/HSTU.)
- **Amount representation** — log-bucketed by default (robust, clean MLM). Raw
  log1p+z-score scalar is the weak baseline; soft-binning and learned numerical
  embeddings (PLE / periodic, Gorishniy et al.) are the upgrades.

**Borrow from Nubank (the downstream pattern):**
- **Joint fusion** — concatenate the FM embedding with existing tabular features
  and feed both to the downstream model. It beats either alone, and it's the
  honest "ship it in a real bank" path: you keep your feature pipeline and just
  *add* the embedding. Our eval reports **raw vs FM-only vs fusion** side by side.

**Serving — both paths, on purpose:**
- Default = **batch-embed → feature store → XGBoost** (what almost everyone
  actually does; the transformer is not on the hot path).
- Also ship a **Ray Serve** online deployment that caches static (card-level)
  embeddings and runs the model only over the recent dynamic window — the
  two-tier pattern (small model real-time / bigger model batch) that Revolut's
  PRAGMA and others use.

**A/B framing for the blog:** same Ray pipeline, two tokenizers — NVIDIA-style
flat (baseline A) vs our static/dynamic split (B). The delta is the result. The
template ships **B**; A is a documented toggle (`SPLIT_FIELDS=False`).

**Explicitly out of scope:** HSTU / Meta's generative-recommender architecture.
Right for 10^5-length sequences and billion-scale dynamic vocab; wrong for
transactions. Don't spend time here.

---

## Datasets (all synthetic — fine for a template, the real limiter for production)

- **IBM TabFormer** — 24M transactions, 20K users, 12 fields, binary fraud label.
  *The* benchmark for this subfield and what NVIDIA uses. **Start here.**
- **Sparkov / Kaggle "Credit Card Transactions Fraud Detection"** — simulated,
  1K cardholders × 800 merchants, 2019–2020. Good second set to show transfer.
- **IBMCard** — multi-agent sim, 2K cardholders, spans decades, global travel;
  longer per-card sequences, better for the long-context story.

---

## Reading list

### Must read (start here) ★
- **NVIDIA — Build Your Own Transaction Foundation Model (blueprint + repo)** —
  the scaffolding we copy. Read the tokenizer notebook closely.
  https://build.nvidia.com/nvidia/build-your-own-transaction-foundation-model ·
  https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model
- **★ TREASURE: The Visa Payment Foundation Model** (arXiv 2511.19693, Nov 2025) —
  the most credible public production system; source of the static/dynamic split.
  https://arxiv.org/abs/2511.19693
- **★ Towards a Foundation Purchasing Model** (arXiv 2401.01641, Featurespace/Visa)
  — cleanest open, reproducible academic anchor; autoregressive TFM across 180
  banks. https://arxiv.org/abs/2401.01641

### Architecture & tokenization
- **FATA-Trans: Field And Time-Aware Transformer** (CIKM 2023) — where the
  static/dynamic split + time-aware positions were crystallized.
  https://arxiv.org/pdf/2310.13818
- **Open Banking Foundational Model** (arXiv 2511.12154, Nov 2025) — 10M accounts,
  masked-token pretraining, the cross-institution generalization test.
  https://arxiv.org/abs/2511.12154
- **TransactionGPT** (arXiv 2511.08939, Nov 2025) — recent generative TFM; useful
  contrast for the NTP-vs-MLM objective discussion.
  https://arxiv.org/pdf/2511.08939

### Production patterns & serving
- **Stripe Payments Foundation Model** (Cognitive Revolution, Emily Sands) —
  masked-feature modeling at scale; the 59%→97% card-testing fraud result.
  https://www.cognitiverevolution.ai/stripes-payments-foundation-model-how-data-infra-create-compounding-advantage-w-emily-sands/
- **Nubank — Foundation Models transformation** + **joint-fusion writeup (ZenML)**
  — the embedding+tabular fusion recipe we use downstream.
  https://building.nubank.com/foundation-models-ai-nubank-transformation/ ·
  https://www.zenml.io/llmops-database/fine-tuning-transaction-foundation-models-with-joint-fusion
- **NVIDIA blog — Why FIs are converging on transaction foundation models** —
  good framing/business context for the blog intro.
  https://blogs.nvidia.com/blog/financial-institutions-transaction-foundation-models/
- **Building transaction FMs on Nebius** — a competing infra writeup; references
  Revolut PRAGMA's two-tier (10M real-time / 1B batch) serving split.
  https://nebius.com/blog/posts/building-transaction-foundation-models-on-nebius-ai-cloud
- **From XGBoost to Foundation Models: Fraud Detection** (ML Frontiers) —
  accessible explainer of the embedding→XGBoost downstream story.
  https://mlfrontiers.substack.com/p/from-xgboost-to-foundation-models
