# How much data does a transaction foundation model need?

Working notes / talk material, companion to `SCALING_TIERS.md`. The recurring question:
*how many transactions before a transaction FM is useful — and could a small business build
one for a new use case?* The honest answer reframes it, because "how much data" depends on
whether you're **building** an FM or **using** one, and the binding constraint usually isn't
raw transaction count.

## What actually drives "useful"

A transaction FM is a *small* model learning a narrow thing — what a normal transaction
sequence looks like for one entity. Ours is **29M params**, four orders of magnitude below an
LLM, so it needs far less data than "foundation model" connotes. Three quantities matter,
roughly in this order:

1. **Number of distinct entities** (cards / customers / accounts). The model generalizes
   across *behavioral diversity*, not raw volume. 10M transactions from 50 entities teaches it
   far less than 2M from 20,000 entities.
2. **History depth per entity.** It's a sequence model — an entity with 3 transactions gives it
   almost nothing to model. You want entities with dozens+ of transactions each.
3. **Downstream positive count.** Separate from pretraining, and it usually binds *first*. At
   0.1% fraud, 1M transactions is only ~1,000 fraud examples. You need enough positives
   (rule of thumb: 1,000+) just to fit and *evaluate* the downstream model stably.

## Rough anchors (heuristics, not thresholds)

- **< ~100K transactions:** don't build an FM. Gradient-boosted trees on hand-engineered
  features will beat it and need a fraction of the data. Deep learning loses to XGBoost at small
  scale.
- **~1M–10M across thousands of entities:** an FM becomes *viable* and may add modest lift.
- **10M–100M+ across many entities:** the regime where it clearly earns its keep.

Grounding in this template's run (real IBM TabFormer): **24.4M transactions**, 0.122% fraud
(~29.7K fraud events total), tokenized into ~66K pretrain sequences, 29M-param Llama decoder.
That's comfortably in the third tier — and it still only bought **+24% PR-AUC** over raw features
(see below).

## The sobering honest bit

Even at 24.4M transactions, our FM added **+24%** PR-AUC over raw features (NVIDIA's reference
adds +42%). The raw transaction fields are *strong* on their own. An FM earns its cost when you
have:

- **scale** (the tiers above),
- **sequential/temporal structure** that's hard to hand-engineer, and especially
- **one representation reused across many downstream tasks** — fraud *and* churn *and* credit
  *and* marketing off the same embedding.

If you have one task and modest data, engineered features + XGBoost is usually the better ROI.
Don't build an FM to prove you can.

## The small-business case

The key move: **a small business is a *consumer* of a foundation model, not a producer of one.**
SMB volume — thousands of transactions a month, tens of thousands a year — is well under the
pretraining floor; a from-scratch FM would memorize noise and lose to simple features. Realistic
options, best first:

1. **Skip the FM — use XGBoost / LightGBM on engineered features.** At that scale this is
   genuinely state of the art, not a consolation prize.
2. **Consume an existing FM** if one exists for the domain — use its embeddings or lightly
   fine-tune. That's the *entire point* of "foundation": one expensive pretrain serves many small
   downstream users.
3. **Pool at the platform level.** The unit with FM-scale data isn't the individual SMB — it's
   the payment processor, the vertical SaaS, the bank serving *thousands* of small businesses.
   They aggregate everyone's transactions, pretrain one FM, and serve each tenant. If you're
   building *for* small businesses (a platform), that's your data story; if you *are* one small
   business, you're a tenant.

## For a genuinely new use case

Pretrain once on a large transaction corpus **if you have the entity diversity for it**, then
reuse the embedding across tasks — that reuse is the foundation-model value proposition. If you
don't have the corpus, the answer isn't "collect 100× more data"; it's "use classical ML, or get
access to a pretrained model / pooled data."

## One-line summary

To *build* a useful transaction FM: millions of transactions across thousands of diverse
entities, with enough downstream positives to train and measure — and even then, only if scale,
hard temporal structure, or multi-task reuse justifies it over boring, excellent gradient-boosted
trees. To *use* one: almost no data — you consume embeddings. A small business is on the "use"
side.
