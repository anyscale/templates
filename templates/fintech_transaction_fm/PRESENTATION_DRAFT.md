# Transaction Foundation Models — Presentation Outline (DRAFT, unreviewed)

## Captured beats for the rebuild (ours, verified — unlike the draft below)

- **FINE-TUNE FRAMING CORRECTION (Zach, 2026-07-13, ANGRY — this ordering is the rule):**
  the point of fine-tuning is REMOVING the raw pipeline. The story is fine-tuned ALONE:
  0.1263, no raw features, no XGBoost, at/above the raw baseline it replaces ("retire the
  fragile, ad-hoc, manually built xgboost pipeline" — his words in nb01). fine-tuned+raw
  0.1988 stays as an ensemble footnote ("best score if you keep the pipeline — but keeping
  it is what fine-tuning exists to avoid"), NEVER as the headline. NOTE for claims: 0.1263
  vs raw 0.1238 is ~2% at ~112 frauds — run the seed×bootstrap before saying "beats" alone;
  "matches, with no pipeline" is the safe wording until then.
- **THE FINE-TUNE ENDING (2026-07-13, verified, finetune/full/RESULTS.json):** the deck's
  third act. We reproduce the blueprint, beat it, then go where it never went: fine-tune the
  foundation model. History-window fine-tune + raw = **AP 0.1988, single draw** — above
  fusion typical (0.136), every measured fusion draw (max 0.161), and NVIDIA's published
  0.1755 on the same basis. History fine-tune ALONE (0.1263) is the first no-raw-features
  detector to beat the raw baseline — the "fraud is visible in sequence" thesis, finally
  demonstrated by a deployed-shape model. Also honest beats: fine-tuning converges in ONE
  epoch then overfits (same signature as their 1-tree XGBoost — the task, not the tooling);
  single-txn variant confirms the lossy-tokenizer bottleneck (0.0907 < raw). Caveat before
  externalizing: single draw, ~112 frauds; run the seed×bootstrap first.
- **RESULT HIERARCHY (Zach, 2026-07-09 — non-negotiable ordering):** headline = (1) OUR
  foundation model beats NVIDIA's foundation model, 0.04–0.06 vs 0.0123 = 3–5×, trained by
  us from scratch; (2) OUR fusion beats their fusion — peak 0.284 vs their single-draw
  0.1755, cleared in ~1/6 draws. The raw 0.1238 exact match is TERTIARY — the control,
  pinned to match by construction; it appears only as the credibility line ("that's how you
  know the wins are the model, not the methodology"), never as an achievement.

- **WHY our FM beats theirs — the narrative seed (Zach, 2026-07-09):** the 3–5× fm win
  isolates to the WEIGHTS (their checkpoint in our harness reproduces their 0.0123; ours in
  the same harness gets 0.024–0.06). The one known difference: they trained ~3,000 steps ≈
  263M tokens ≈ ONE epoch (their committed config is literally max_steps:30); we trained
  ~16,000 steps ≈ 2.1B tokens = 8 epochs with THEIR recipe. Leakage ruled out (label never
  tokenized, temporal split, per-txn embedding; AUC 0.96-vs-0.878 agrees and is draw-stable).
  Zach's narrative direction: something like "with Ray you don't have to give up early on
  training because you ran out of compute" — his words, he called the literal line too dorky,
  but the shape is right: they shipped an undertrained checkpoint because one node is what
  they had; we finished the training run. Anticipates the audience's "why do you beat them
  so much?" red-flag question head-on.
  **Queued experiment (cheap, makes it bulletproof):** 1-epoch ablation — train our pipeline
  ~2,000 steps (~15 min on 8×A10G), show fm lands ≈ their 0.012–0.015 → a two-point
  tokens-vs-embedding-quality curve: "quality tracks training compute; they stopped at 1
  epoch." Run before the deck is finalized.
- **The hardware economics contrast (Zach, 2026-07-09):** NVIDIA's blueprint = single node,
  single top-shelf GPU (their prereq: 1× A100 80GB / H100), not distributed — and that GPU
  spends part of its life on dataframe work. Ours = data stages distributed across cheap
  autoscaling CPU nodes; GPUs (A10Gs, not A100s) reserved for pretrain + embed forward passes
  only; each pool scales independently and to zero. More scale, lower unit cost, no idle
  expensive hardware. One-liner: "cudf made one GPU fast; Ray Data makes the pipeline wide.
  We kept their math — provably, to the byte — and changed where it runs."
- **Scale-to-zero vs warm floors (Zach, 2026-07-09):** the cluster sawtoothing 0→56→0→72
  CPUs during one afternoon is (a) tailored-to-workload AND (b) real waiting — which one
  depends on `startup_time / job_time` (~2% for the 2h pretrain, ~30% for the 8-min split,
  dominant for interactive dev). Resolution is asymmetric by pool cost: warm floor on the
  cheap CPU pool when iterating or presenting, strict min=0 on GPUs. "Elasticity is a
  per-pool policy — floor for latency, ceiling for cost." Full write-up: PERFORMANCE.md §14.
  Deliberately NOT optimized in the repo — it's a conversation topic for nb09 + the deck.
- **The identity-check story (Stage 0, 2026-07-09):** distributing without changing results is
  provable, not asserted — 91,265/91,265 merchant hashes, 200K/200K rows, 12/12 token columns,
  vocab 6251 equal. Includes a good war story: cuDF's hash_values is murmur3 + Boost
  hash_combine (reversed from known pairs) — 0/200K rows matched until that was found.
  Deep-dive beat: "why you verify identity before you trust a port."

> Provenance: sketched by a separate Claude session that had NO access to this template's
> notebooks or results. Pasted in verbatim 2026-07-09. Treat every number and claim as
> unverified until checked against (a) our actual run results (RESUME_HERE.md,
> FINDINGS_FM_REPRODUCTION.md) and (b) the cited external sources. Known conflicts with
> our ground truth are being tracked as we co-develop this into the real deck.

Segments 1 & 2, slide-by-slide with talk tracks. Workshop (Segment 3) mapping included at the end so every claim planted early pays off in code.

**Running example (use in all three segments):** one cardholder's history —
`paycheck deposit → grocery → transit fare → streaming subscription → [travel: airport food, hotel] → sudden burst of small card-not-present authorizations`
The burst looks fine row-by-row; it only looks like fraud *in sequence context*. This single example carries the core argument for the whole approach.

---

## SEGMENT 1 — Business Context (5–15 min)

### Slide 1.1 — Cold open: the number, not the definition
- **Slide:** "$112B" huge on screen. Subtitle: fraud blocked by Stripe last year using foundation models on transaction data; ~38% average reduction in fraud rates.
- **Talk track:** Don't define anything yet. "This number exists because of a shift in how the payments industry builds AI. That shift is what the next hour is about." Sets stakes for every audience tier.
- **Source:** NVIDIA blog, "Why Financial Institutions Are Converging on Transaction Foundation Models" (blogs.nvidia.com/blog/financial-institutions-transaction-foundation-models/)

### Slide 1.2 — What is a TFM (one slide, one analogy)
- **Slide:** Left: a sentence being read by an LLM. Right: your running-example transaction sequence being read by a TFM. Same picture, different tokens.
- **Talk track:** "A language model reads sequences of words and learns to predict what comes next. A transaction foundation model reads sequences of *payments* and does the same. In doing so it learns how money moves, how customers behave, how merchants operate, how risk emerges." Explicitly park the how: "Segment 2 is the how. Right now: the why."
- Introduce the running example here and promise: "We'll build a model that reads exactly this in the workshop."

### Slide 1.3 — The problem TFMs replace: model sprawl
- **Slide:** Diagram of a bank's ML estate today: fraud team → its own XGBoost + feature pipeline; credit team → another; churn team → another; AML → another. Dozens of parallel stacks.
- **Talk track:** For ~20 years the industry standard has been gradient-boosted trees on hand-engineered features. It works — but every new product means a new pipeline, new labels, months of feature engineering. Rules-based fraud systems are reactive by construction: they encode yesterday's fraud. Cite NVIDIA's 2026 State of AI in Financial Services framing: as AI scales, *fragmented model architecture* becomes the limiting factor.
- **Audience note:** this slide resonates most with managers — it describes costs they already recognize (duplicated pipelines, slow product launches, per-team headcount).

### Slide 1.4 — Foundation model as business strategy
- **Slide:** Hub-and-spoke: one pretrained TFM in the center; spokes = fraud, credit, LTV, personalization, AML. Label the spokes "LoRA / fine-tune on 1–2% of parameters."
- **Talk track:** Three-part economics:
  1. **Pretrain once** on data you already own (no labels needed — self-supervised).
  2. **Adapt cheaply** — new tasks via lightweight fine-tuning/LoRA rather than a new pipeline (Revolut-on-Nebius pattern: LoRA on 1–2% of params).
  3. **The moat is the data.** Your transaction history is proprietary; competitors cannot replicate it. "The data already exists. The architecture is proven. The infrastructure is ready." (NVIDIA's framing — worth borrowing directly.)
- Optional second beat: "embeddings become an internal product" — one team ships representations, every team consumes them.

### Slide 1.5 — Use-case tour
- **Slide:** Grid of 6–8 use cases, one line each: fraud detection · credit underwriting · AML/financial crime · churn & lifetime value · personalization/recommendations · authorization optimization · recurring-payment detection · merchant analytics.
- **Talk track:** Anchor with Revolut PRAGMA: one pretrained backbone demonstrably supports fraud, credit scoring, engagement, recurrence detection, recommendations, and LTV. Then the customer-impact example — Nubank's nuFormer: precision fine enough to *safely extend credit to people prior models excluded*, live in their largest Brazil credit segment, expanding to Mexico/Colombia. Pairing financial inclusion with revenue growth makes this the most broadly relatable example in the talk.

### Slide 1.6 — Industry landscape: who has shipped (level-set slide)
- **Slide:** Logo wall with one-line descriptors:
  - **Stripe** — Payments Foundation Model (the announcement that put TFMs on the map)
  - **Visa** — TransactionGPT
  - **Mastercard** — large tabular foundation model: billions of anonymized transactions today, designed for hundreds of billions of records (fraud, auth, chargeback, merchant location, loyalty); built with NVIDIA NeMo AutoModel + AWS + Databricks
  - **Nubank** — nuFormer, 100M+ customers, trillions of tokens
  - **Revolut** — PRAGMA (+ FinCrime AI agents, ~2M tasks/month)
  - **Adyen** — TFMs in production across ~$1T in payment volume
  - **Plaid** — transaction foundation model
  - **NVIDIA** — the published blueprint that makes this buildable by everyone else
- **Talk track:** Two messages at once: (a) for newcomers — "this is not a research idea, it's the production standard at every major payments company"; (b) for people who already know TFMs — this is the level-set, and NVIDIA's entry is the pivot: "the playbook is now public. Which is why the second half of today is us building one."

### Slide 1.7 — Headline results
- **Slide:** Three stats, big type:
  - Stripe: ~$112B fraud blocked / ~38% fraud-rate reduction
  - Nubank: +1.20% average AUC across benchmark tasks, in production for 100M+ customers
  - NVIDIA blueprint: **~50% lift in Average Precision over a strong XGBoost baseline** on the public IBM TabFormer fraud dataset
- **Talk track:** Explain why "1.2% AUC" is enormous: at 100M-customer scale, a point of AUC is tens of millions of dollars of credit-loss and fraud-loss delta, so foundation models clear the production bar decisively. Then set up the callback: "That third number — the ~50% AP lift — is on a *public dataset* with a *published pipeline*. It's the number we're going to reproduce in the workshop."

### Slide 1.8 — Why now
- **Slide:** Three converging arrows: (1) proven architecture (transformer recipe transferred from LLMs), (2) accelerated data infrastructure (GPU dataframes / distributed compute make trillion-token transaction pipelines tractable), (3) published playbooks (NVIDIA developer example, Nubank/Revolut engineering blogs, papers).
- **Talk track:** "Two years ago this required a Stripe-sized research team. Today the recipe is open. The differentiator is no longer the architecture — it's your data and your ability to run the pipeline at scale." (This is also the honest setup for why the workshop focuses on scaling.)

### Slide 1.9 — What a TFM is NOT (objection pre-empt / level-set)
- **Slide:** Three crossed-out misconceptions: ✗ "ChatGPT on my bank data" ✗ RAG over transactions ✗ a chatbot.  ✓ An **embedding factory**: it turns raw histories into compact behavioral representations your existing models and systems consume.
- **Talk track:** One minute, but it prevents the most common confusion in mixed audiences and cleanly hands off to Segment 2: "So if it's not a chatbot, what is it actually doing to your transactions? That's next."

*(Timing: 1.1–1.2 ≈ 3 min; 1.3–1.4 ≈ 4 min; 1.5–1.7 ≈ 5 min; 1.8–1.9 ≈ 3 min. Trim 1.8 and fold 1.9 into 1.2 for the 5–8 min version.)*

---

## SEGMENT 2 — Technical Context (5–15 min)

### Slide 2.1 — Primitives in 90 seconds: tokens, embeddings, pretraining
- **Slide:** Three mini-panels: text → tokens; tokens → vectors (embeddings, "similar things end up near each other"); next-token prediction as the self-supervised objective ("the model teaches itself by predicting what comes next — no labels required").
- **Talk track:** Pitched for the non-technical audience without losing the engineers — frame it as "vocabulary for the next ten minutes," not a lecture.

### Slide 2.2 — The reframe: a customer history is a document
- **Slide:** The running example rendered two ways: as a database table (rows) and as a token sequence (a "sentence"). Objective: predict the next transaction.
- **Talk track:** "Everything downstream of this slide is consequences of taking this analogy seriously — including the places where it breaks."

### Slide 2.3 — Where the analogy breaks: tokenization is THE design decision
- **Slide:** LLM vs TFM comparison. LLM: tokenization is a solved, trivial front-end (BPE, one fixed vocab, one stream, ~free). TFM: tokenization is the *central modeling decision* and the *dominant pipeline cost*.
- **Talk track:** The input isn't a string — it's a sequence of structured events, each with heterogeneous fields: amount, merchant, MCC, timestamp, geography, device, channel, response codes. Before you can train anything you must answer a question LLMs never face: **what is a token?**
- **Concrete anchor:** the NVIDIA blueprint ships a modular, GPU-accelerated (RAPIDS cuDF/cuML) tokenizer with swappable per-field components — tokenization is so central it's an entire notebook (02) and a whole `src/tokenizer/` module in the code you'll see.

### Slide 2.4 — The granularity decision: three ways to tokenize a payment
- **Slide:** Three diagrams:
  1. **Field-as-token** — every field value is a token; a transaction is a little sentence; a history is a paragraph. (TabFormer lineage; **what the NVIDIA blueprint uses**.)
  2. **Event-as-token** — compress each transaction into one vector; sequence model attends over events. (Stripe's "charge as a token" framing.)
  3. **Hierarchical** — encode fields within an event, then events within the history. (Revolut PRAGMA; FATA-Trans with field-type + time-aware embeddings.)
- **Talk track:** The tradeoff is concrete: field-as-token multiplies sequence length by fields-per-event, and attention cost is quadratic in sequence length — so granularity choice directly sets your compute bill and your usable history window.

### Slide 2.5 — Per-field design problems
- **Slide:** Three quick panels:
  - **Amounts:** continuous numbers → quantile/log binning into discrete tokens (vs. continuous embeddings).
  - **Merchants:** high-cardinality vocab — millions of merchant IDs vs. a ~50K-word LLM vocab; bucketing, hashing, MCC fallback. (Blueprint's whole domain vocab is ~6,251 tokens — a deliberate design choice worth calling out.)
  - **Time:** unlike words, *spacing is signal*. Time-delta tokens and time-aware position embeddings, because "3 transactions in 3 minutes" ≠ "3 transactions in 3 days."
- **Talk track:** Nubank's description is useful here: each attribute (amount, date, description, source, status, denial reason, entry mode) gets its own dedicated pre-processor, and choosing what to include is treated as a hyperparameter search — "as much useful information as possible in as few tokens as possible."

### Slide 2.6 — Brief history: research curiosity → production standard in ~4 years
- **Slide:** Timeline: IBM TabBERT/TabFormer (2021, also the source of the workshop dataset) → Featurespace TallierLTM → FATA-Trans → industrial wave 2024–26 (Stripe PFM, Nubank blog series, Revolut PRAGMA, Visa TransactionGPT, Mastercard) → NVIDIA developer example (2026: the open playbook).
- **Talk track:** One minute. Point out the dataset continuity: "The 2021 academic dataset is the same one NVIDIA's blueprint — and our workshop — trains on."

### Slide 2.7 — Architecture & objective decisions (what you actually choose)
- **Slide:** Decision table:
  - **Encoder (MLM) vs decoder-only (causal/NTP)** — blueprint is decoder-only with causal LM; Nubank uses both NTP and MLM variants.
  - **Context length** — longer history = better long-range behavior capture, but quadratic attention cost and divergence risk (Nubank's published finding).
  - **Model size ↔ latency** — sizing is driven by the serving budget: ~10M-param models for real-time fraud vs ~1B for precision-heavy credit scoring (Revolut/Nebius pattern), against the sub-100ms budget real-time authorization allows.
- **Concrete anchor — the actual model we'll train (blueprint spec):**

  | Parameter | Value |
  |---|---|
  | Architecture | Llama-style decoder-only transformer |
  | Parameters | ~29M |
  | Hidden size / layers | 512 / 8 |
  | Attention | GQA (8 query heads, 2 KV heads) |
  | Context window | 8,192 tokens (RoPE) |
  | Vocabulary | ~6,251 domain-specific tokens |
  | Reference training run | ~3,000 steps on 8× A100 |

- **Talk track:** "Notice how small this is. 29M parameters — the value isn't scale for its own sake, it's the representation. And note the 8× A100 reference checkpoint: even a small TFM requires cluster-scale training — which becomes relevant when we get to infrastructure."

### Slide 2.8 — Using the model downstream (integration patterns)
- **Slide:** Ladder of approaches:
  1. **Frozen embeddings → GBDT** — the obvious first move; Nubank found it often captures limited signal and deltas don't propagate.
  2. **Supervised fine-tuning** — prediction head on the final-token (user) embedding; what Nubank shifted to.
  3. **Fusion with tabular data** — late fusion (LightGBM over embeddings + features) vs joint fusion (end-to-end, DCNv2) — joint fusion beat LightGBM in Nubank's published results.
  4. **LoRA adapters** — cheap task expansion (1–2% of params).
- **Talk track:** Be direct about scope: "The workshop demonstrates rung 1 — embeddings + XGBoost — because it's the cleanest apples-to-apples vs. the baseline. Production systems climb this ladder." Note the blueprint's embedding recipe: 512-dim vectors via last-token pooling, visualized with UMAP.

### Slide 2.9 — The systems reality: TFM training is input-bound
- **Slide:** Pipeline diagram with the bottleneck highlighted: raw tables → **preprocessing + tokenization (the bottleneck)** → training (the easy part, comparatively) → embedding serving.
- **Talk track:** The key, counterintuitive point: unlike LLM training, TFM builds are often *data-pipeline-bound*, not FLOP-bound. Tokenization is schema-coupled, CPU-hungry, and touches every row. Two published answers:
  - **NVIDIA:** move the data work to GPUs — cuDF/cuML, with ~88% data-processing cost savings vs CPU cited in the companion fraud blueprint.
  - **Nubank:** distribute everything — **Ray as their primary scaling framework**, billion-parameter models over complete histories of 100M+ customers, trillions of tokens.
- **This is the bridge:** "Those two answers aren't competitors — the workshop composes them: the blueprint's GPU pipeline, distributed with Ray."
- Also flag serving: embeddings must land inside real-time auth latency budgets (sub-100ms), which drives the small-model tier.

### Slide 2.10 — Open questions and limitations
- **Slide:** Four honest unknowns:
  - **Scaling laws?** Nubank sees continued gains from 5M → 20M → 40M training rows and from longer context — but no one has published a "Chinchilla for transactions."
  - **Schema coupling** — your tokenizer is married to your data model; product changes can mean retokenize + retrain.
  - **Evaluation culture is immature** — no shared benchmark beyond TabFormer; everyone reports lift on private tasks.
  - **Cross-institution transfer unproven** — nobody has shown a TFM that generalizes across companies' data.
- **Talk track:** 90 seconds. Acknowledging the limits openly strengthens everything that came before it.

### Slide 2.11 — Bridge: here is exactly what we'll build
- **Slide:** The blueprint's five stages, verbatim from the repo, with your Ray delta annotated:

  | Blueprint notebook | Stage | Workshop delta (Ray) |
  |---|---|---|
  | 01_dataset_baseline | TabFormer data + temporal splits + GPU XGBoost baseline | Ray Data ingest; baseline unchanged (it's the yardstick) |
  | 02_seq_preproc_tokenization | Custom GPU tokenizer (cuDF/cuML) | **Distribute tokenization across the cluster — the input-bound stage from 2.9** |
  | 03_foundation_model_training | Pretrain ~29M Llama decoder w/ NeMo AutoModel (causal LM) | Multi-node training (Ray Train / distributed NeMo) |
  | 04_inference_embedding_extraction | 512-d embeddings via last-token pooling, UMAP viz | Batch inference with Ray Data |
  | 05_xgboost_fraud_detection | Raw features vs embeddings vs combined | Reproduce the ~50% AP lift from Slide 1.7 |

- **Talk track:** "Every claim from the last 25 minutes maps to a notebook. Segment 1 promised the lift; Segment 2 explained the mechanism; now we run it."

*(Timing: 2.1–2.2 ≈ 2.5 min; 2.3–2.5 ≈ 5 min (core section); 2.6 ≈ 1 min; 2.7–2.8 ≈ 4 min; 2.9 ≈ 2 min; 2.10–2.11 ≈ 2.5 min. For the short version: cut 2.6, compress 2.5 to one panel, keep 2.9 intact — it justifies the workshop.)*

---

## Narrative threads across all three segments
1. **The running example** appears in 1.2, 2.2, and as actual rows in workshop notebook 01/02.
2. **The ~50% AP number** appears three times: promised (1.7), mechanized (2.11), reproduced (workshop nb 05).
3. **"Input-bound, not FLOP-bound"** planted in 2.9 is what makes the Ray Data tokenization section of the workshop the *point* rather than plumbing.
4. **Nubank-uses-Ray** is the strongest third-party validation available — introduce it in 2.9 and reference it again when the Ray cluster starts in the workshop.
5. **Small model, big pipeline** (29M params, 8×A100, trillion-token industrial scale) — recurring justification for why scaling infrastructure, not architecture, is the differentiator.

## Key sources
- NVIDIA developer example repo: github.com/NVIDIA-AI-Blueprints/transaction-foundation-model
- NVIDIA technical blog: developer.nvidia.com/blog/build-your-own-transaction-foundation-model-for-financial-intelligence/
- NVIDIA industry blog: blogs.nvidia.com/blog/financial-institutions-transaction-foundation-models/
- Nubank engineering series: building.nubank.com (foundation models posts, parts 1–3 + Jan 2026 update)
- Nubank/Ray infrastructure: ZenML LLMOps case study; Nu Videocast (June 2026) on nuFormer
- Revolut PRAGMA on Nebius: nebius.com/blog/posts/building-transaction-foundation-models-on-nebius-ai-cloud
- Thoughtworks two-parter on TFMs in banking/payments (business framing you can borrow)
