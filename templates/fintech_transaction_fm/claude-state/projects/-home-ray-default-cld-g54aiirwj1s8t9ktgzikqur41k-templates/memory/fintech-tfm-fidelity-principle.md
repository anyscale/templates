---
name: fintech-tfm-fidelity-principle
description: "The core design principle for fintech_transaction_fm — faithfully reimplement NVIDIA, Ray is ONLY the scaling layer"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6d74f244-a018-4944-a718-2746735b927e
---

The `fintech_transaction_fm` template must be **NVIDIA's transaction-FM approach,
faithfully reimplemented**, with **Ray as the ONLY difference** (the scaling layer).

**Narrative the template must deliver:** a reader can do NVIDIA's tutorial
(github NVIDIA-AI-Blueprints/transaction-foundation-model) exactly as-is → hit the
wall that "this is cool but it doesn't scale" → come to our template → get the
**same results**, now scaled on Ray/Anyscale, **without wading into a mess of
modeling differences.**

**Consequence for debugging:** when our numbers DON'T match NVIDIA's (e.g. our FM
embedding gives AUC 0.62 vs their 0.88), that is a **reimplementation divergence to
ELIMINATE**, not a modeling choice to explore. Do NOT invent modeling variants
(different pooling, different objective, different features) to "fix" a gap — that
defeats the whole point. Match their approach; find and remove the divergence.

**Why:** Zach has said this several times. The template's entire value proposition
is "identical modeling, Ray makes it scale." Modeling drift destroys that story and
makes results incomparable.

**How to apply:** for any modeling component (tokenizer, model arch, pretraining
objective, sequence/corpus construction, embedding extraction, downstream recipe),
the reference is NVIDIA's actual code — mirror it. Only the execution layer (Ray Data
for distributed tokenize/embed, Ray Train for distributed pretrain/XGBoost) is ours.
See [[fintech-tfm-series-state]] and the repo's NVIDIA_BASELINE.md.
