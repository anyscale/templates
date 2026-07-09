---
name: fintech-tfm-extension-candidates
description: "Zach's beyond-the-blueprint improvement candidates for fintech_transaction_fm (history-aware embedding, static/dynamic split)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a07737d-42d1-41f6-8e46-6e24a82d9b87
---

Zach (2026-07-09) wants these two remembered as possible improvements. Both diverge from
NVIDIA's blueprint → per [[fintech-tfm-fidelity-principle]] they must ship as *labeled
extensions* next to the faithful comparison, never silently replace it. Durable copy lives
in the repo: `templates/fintech_transaction_fm/RESUME_HERE.md` ("Beyond-the-blueprint
extension candidates" section).

1. **History-aware embedding** — blueprint embeds each transaction alone (~14 tokens), so
   the downstream classifier can't see multi-transaction fraud patterns. Embedding with
   preceding history in context (model ctx 8192 ≈ 600 txns) showed real promise in the
   2026-07-03 debugging: ~10-txn context → fm AUC 0.62→0.795, embed↔raw corr 0.36→0.13
   (complementary). Also makes the presentation's running example (fraud visible in
   sequence) honest for the demo itself.
2. **Static/dynamic field split** (Visa TREASURE / FATA-Trans lineage) — the OLD design's
   idea, deliberately removed in the faithful rewrite; nb01 prose still wrongly advertises
   it as "our one modeling upgrade". Encoding card-constant fields once instead of in every
   txn's 12 tokens → longer effective history per 4096-token window, cheaper pretrain.
   Old pipeline's weak results are NOT evidence against it (confounded). Needs an A/B vs
   the faithful baseline.
3. **Masked-feature modeling (MLM)** — the OLD design's pretraining objective, replaced by
   causal next-token in the faithful rewrite. Still live: NVIDIA's NB05 excuses weak fm-only
   "as expected for decoder-only models"; bidirectional objectives typically embed better;
   Nubank publishes both NTP and MLM variants. A/B: same corpus/arch, MLM objective,
   compare fm-only AP.
4. **Ray Data across the data stages — a DIRECTIVE, not a candidate (Zach 2026-07-09):**
   "We have to use Ray Data and Ray in general to scale across the whole workload. The whole
   point of Ray's scalability story is that CPU-based workflow steps scale independently of
   GPU-based." The faithful rewrite left NO Ray Data in the series (split/corpus/embed =
   single-GPU Ray tasks on NVIDIA's code; nb02 explore = head-node pandas). Rebuild nb02
   (distributed read/split/explore), nb03 (per-card sharded corpus), nb05 (streaming CPU→GPU
   embed) on Ray Data with distributed-output == single-GPU-reference as the acceptance test
   per stage. Full plan in repo RESUME_HERE.md item 4.

See [[fintech-tfm-series-state]].
