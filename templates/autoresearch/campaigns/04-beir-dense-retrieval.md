# Campaign 04 — BEIR dense retrieval (E5) reproduction, then a throughput/cost beat

> **Status: SEED (not started).** The cheapest gate in the program and the purest Ray Data
> batch-embedding workload — a strong Wave-1 opener to prove the harness on a new domain.

## 1. One-liner

Reproduce a strong open dense retriever's published zero-shot nDCG@10 on BEIR, then beat it on
**cost/throughput** (Ray Data batch-embedding) with retrieval quality held as the invariant —
or on quality via a better readout/pooling.

## 2. Why this is a good autoresearch testbed

- **What makes it clean:** frozen encoder with shipped weights + eval → a baseline is cheap to
  establish; pooling / chunking / embedding-efficiency are concrete improvement levers; nDCG@10
  is the held invariant for the efficiency experiment.
- **Ray substrate it exercises:** corpus embedding as a Ray Data `map_batches` + ActorPoolStrategy
  pipeline (CPU chunk → GPU encode, fractional-GPU packing) vs the reference's single-GPU loop —
  the throughput surface the efficiency experiment optimizes.

## 3. Reference

- **Repo:** BEIR `github.com/beir-cellar/beir` — Apache-2.0 (v2.2.0, 2025). Retriever: E5
  `huggingface.co/intfloat/e5-large-v2` — **MIT**, eval in `github.com/microsoft/unilm/e5`.
  (Contriever is Apache-2.0 but **archived** since 2023 — E5 is the live choice.)
- **Ships:** BEIR = eval harness only (nDCG/MAP/Recall via pytrec_eval). E5 = checkpoints (all
  sizes) + eval script. **E5 pretraining is NOT released** → this is a reproduce-the-**eval**
  campaign (frozen encoder), not reproduce-the-training. No pipeline pretrain gate — the gate is
  matching their eval number through our embedding+retrieval harness.
- **Ray-native?** No — plain Python + faiss + pytrec_eval.
- **Reproduction landscape:** heavily reproduced **if you pin dataset versions**. Footguns:
  Touché-2020 was revised; MTEB/BEIR contamination critique (arXiv:2410.08385) means post-2023
  models may have trained on the corpora — **E5-large-v2 (2022) is a cleaner baseline** for that
  reason.

## 4. The gate

- **Decision metric:** nDCG@10, zero-shot.
- **Published to match (confirm from the shipped script, NOT the paper):** E5-large-v2 ≈ **50.6
  avg** per the UniLM README. ⚠️ The paper (arXiv:2212.03533) reports v1 E5-large = 50.0 over 15
  datasets — **v1 ≠ v2 and the dataset count differs**. Pin the exact subset and trust the
  script output.
- **Guard metrics:** Recall@100, MAP; encode throughput (docs/s) and $/1M-docs (the beat target).
- **Pipeline gate:** reproduce the per-dataset nDCG@10 through the Ray embedding+faiss harness.

## 5. Pinned eval

The exact BEIR dataset **versions** + the subset (13/15/16/18 — "BEIR average" is ambiguous;
pin yours) + query/qrels files + pytrec_eval config + E5's query/passage prefixes ("query:" /
"passage:" — a silent omission tanks nDCG). Content-hash the corpus+qrels. Dense positives —
point estimates stable; still report per-dataset spread.

## 6. Data

- **BEIR 18 datasets.** 14 freely redistributable; **BioASQ, Signal-1M, TREC-NEWS, Robust04 are
  NOT** — self-assemble or exclude (and say so). Sizes: SciFact ~5K docs → MSMARCO 8.8M,
  DBPedia 4.6M. All the redistributable ones are low-friction HF downloads.
- **Input audit:** the E5 prefix scheme, max sequence length / chunking of long docs (truncation
  changes recall), normalization before cosine.

## 7. Rayification

| Stage | What | Ray lib |
|---|---|---|
| ingest + chunk | corpora → passages | Ray Data |
| **encode** | E5 forward pass over corpus + queries | **Ray Data `map_batches` + ActorPoolStrategy, fractional GPU** |
| index + search | faiss / exact, top-k | Ray tasks / Ray Data |
| score | nDCG@10 via pytrec_eval | driver |

The encode stage is the whole Anyscale story: E5-large is ~1.3GB → 0.5 GPU/replica, pack
several per A10G, CPU-read + GPU-encode in one streaming pass, no intermediate disk.

## 8. Fidelity ladder

- **smoke:** SciFact (~5K docs), 1 GPU — the encode→index→score loop runs.
- **proxy:** the small/medium BEIR sets (SciFact, NFCorpus, FiQA, ArguAna, SCIDOCS); rank
  throughput ideas and any pooling change.
- **full:** the pinned 13–15 dataset sweep incl. MSMARCO/DBPedia (the corpus-encoding-dominant
  ones — where the cost beat shows).
- **Proxy axis — TWO distinct proxies, do not conflate:**
  - *Quality proxy* = small BEIR sets (SciFact/NFCorpus/FiQA) — fine for ranking pooling/chunking
    changes.
  - *Throughput proxy* = a **fixed slice of a BIG corpus** (e.g. 5% of MSMARCO). Encode
    throughput on SciFact (5K docs) is all fixed overhead (model load, actor startup) and zero
    steady-state — timing a marathoner over 40 meters. Throughput claims must proxy on a large
    corpus slice, per the template's "reduce the axis that scales cost, preserve the mechanism."
- **Rare-signal trap:** small corpora have different hardness; calibrate that a quality change on
  a small set holds on a large one before trusting the proxy.

## 9. "Beat it" hypotheses

1. **Platform-efficiency demo (primary) — NOT a "beat a published number" claim.** The
   reference published *no* cost/throughput number, so there is nothing to *reproduce-then-beat*
   on cost; racing our Ray pipeline against a strawman we built ourselves would violate the
   program's brand. Frame this honestly: encode BEIR at lower $/1M-docs via Ray Data
   heterogeneous streaming + fractional-GPU packing, benchmarked against a **rigorously-defined
   baseline** (a competent naive single-GPU + faiss loop at the *same* GPU budget) with
   **nDCG@10 held as the correctness invariant**. This is a platform demo (the more useful
   artifact for Anyscale anyway) — label it as such. Flag `encode_pipeline`.
2. **Readout/pooling (quality) — fails the mechanism check; probe before sweeping.** The FM
   readout thesis does *not* obviously port: E5 was contrastively trained end-to-end *through*
   mean-pooling, so mean-pooling is baked into its weights — swapping to last-token / attention
   pooling on frozen weights should be *expected to hurt*, not help. So this earns exactly ONE
   $2 probe (one dataset, one alt pooling) before any sweep; sweep only if the probe surprises.
   Flag `pooling`.
3. **Chunking strategy:** long-doc chunking + max-pooling over chunks vs truncation, on the
   long-doc datasets — this one *does* have a plausible mechanism (truncation loses signal E5
   never saw). Flag `chunking`.

## 10. Budget

- **Wave 1.** Eval-only, no training.
- **GPU-hours:** smoke <0.5 · proxy (small sets, several pipeline variants) ~5 · full (encode
  MSMARCO 8.8M + DBPedia 4.6M + the rest, a couple pipeline variants) ~15–25 · **total ~25–35**.
- **GPU tier:** A10G/L4, spot ON. **~$30 on-demand / ~$12 spot.**
- **Approval:** envelope only.

## 11. Controls

- **BM25 baseline** (the honest cheap floor — many "neural beats" vanish against tuned BM25,
  per the RecSys-2019 / Dacrema literature the FM campaign cites). Beat BM25 convincingly.
- **Contamination guard:** stick to E5-large-v2 (pre-2023) as the clean baseline; note the
  contamination critique for any newer model comparison.
- **Leak-audit trigger:** nDCG@10 implausibly above the published table on any dataset → a
  qrels-version or prefix bug.

## 12. Risks

Dataset-version drift is THE reproducibility trap (pin versions, exclude non-redistributable
sets explicitly). "BEIR average" ambiguity — always report per-dataset + your subset definition.
No training gate means the quality "beat" is bounded by the frozen encoder — the platform demo
is the more durable claim.

**Vector-matrix host-RAM trap (the FM campaign's 128GB-head war story, redux):** MSMARCO is
8.8M docs × 1024-dim × fp32 ≈ **36GB of vectors**; add DBPedia (4.6M) and exact top-k over all of
it and a naive "faiss/exact on the driver" rediscovers the OOM that forced the FM campaign onto a
128GB head for a 7GB matrix. **Mitigation baked into the plan:** vectors stream to disk/object
store (fp16 halves it), search is **sharded** (Ray tasks over index shards → merge top-k) or
faiss-on-GPU; the head node stays small. Budget object-store, not host RAM.
