# Campaign 01 — HSTU generative recommender (Meta) vs SASRec

> **Status: SEED (not started).** The natural second vertical off the FM work — same
> sequential-encoder shape, a public benchmark with a pinned results table, and the repo is
> already cloned at `~/anyscale/generative-recommenders`.

## 1. One-liner

Reproduce Meta's HSTU sequential recommender's published HR@10/NDCG@10 on the public
MovieLens/Amazon benchmarks, then push it — the direct recsys analog of the transaction-FM
readout/context work.

## 2. Vertical & why Anyscale

- **Vertical:** e-commerce / media & streaming recommendation.
- **Why Anyscale:** maps to the e-commerce playbook (ranking/recsys) and echoes **Pinterest's
  30× cost cut (Spark→Ray Data)** and Amazon's 82% cut. Ray Train DDP for the transducer +
  Ray Data for sequence preprocessing + Ray Serve for the M-FALCON online path. HSTU's whole
  thesis is *scaling* sequential recommenders — the ScalingConfig-only laptop→multinode story
  is native here.

## 3. Reference

- **Repo:** github.com/meta-recsys/generative-recommenders (facebookresearch URL redirects) —
  Apache-2.0. Already cloned locally.
- **Paper:** "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
  Generative Recommendations," Zhai et al., ICML 2024.
- **Ships:** training code ✓ (`main.py` + per-dataset `.gin` configs), preprocessing ✓; **no
  checkpoints** (train from scratch — so the gate is the pipeline gate, not an artifact gate).
- **Ray-native?** No — single/multi-GPU torch + torchrec/fbgemm. Rayification is real work
  (wrap the training loop in a `TorchTrainer`, move preprocessing to Ray Data).
- **Reproduction landscape:** README ships a results table "verified as of 04/15/2024."
  **Critique to respect:** Synerise argues MovieLens is pathologically constructed and near
  saturated — beating HSTU on ML-1M/20M is easy but unpersuasive. **Amazon Books is the
  defensible surface.**

## 4. The gate

- **Decision metric:** HR@10 and NDCG@10 (also @50/@200 reported).
- **Published (HSTU-large vs SASRec), confirm from the repo's eval:**
  ML-1M HR@10 **0.3294** / NDCG@10 0.1893; ML-20M HR@10 0.3556 / NDCG@10 0.2098;
  **Amazon Books HR@10 0.0478 / NDCG@10 0.0262** ← the one that matters.
- **Guard metrics:** training throughput (HSTU's efficiency claim), full HR/NDCG@{50,200}.
- **Pipeline gate:** reproduce the ML-1M table first (cheapest), then Amazon Books, through
  the Ray harness before any "beat it" work.

## 5. Pinned eval

The repo's leave-one-out test split per dataset (last interaction held out), fixed candidate
sampling, fixed seed. Freeze the evaluated user set + negative-sampling scheme +
metric-@k implementation as one protocol module. Positives are dense here (unlike fraud) — CIs
still reported, but point estimates are less fragile.

## 6. Data

- **MovieLens** (ML-1M ~6k users, ML-20M ~190MB) — GroupLens, research-use terms. **Amazon
  Reviews (Books)** — open. Both small; a **synthetic ML-3B** (fractal expansion) exists for a
  scaling story. All fully open.
- **Input audit:** item-id vocabulary construction, sequence truncation length (HSTU's whole
  point is long sequences — don't silently truncate), timestamp/position features vs theirs.

## 7. Rayification

| Stage | What | Ray lib | Nearest archetype |
|---|---|---|---|
| ingest + preprocess | interaction logs → per-user sequences, item vocab | Ray Data `groupby`/`map_groups` | `workloads/batch-inference` |
| pretrain | HSTU transducer, sampled-softmax | Ray Train `TorchTrainer` DDP | `workloads/distributed-training` |
| eval | HR@k/NDCG@k, retrieval-style | Ray Data batch scoring | — |
| serve (optional) | M-FALCON candidate generation | Ray Serve | `workloads/online-inference` |

## 8. Fidelity ladder

- **smoke:** ML-1M, tiny model, 1 GPU, few steps — code runs.
- **proxy:** ML-1M / ML-20M full config on 1–4 GPUs (already cheap — the whole dataset fits
  24GB HBM). Rank ideas here.
- **full:** Amazon Books at the paper's HSTU-large config, multi-GPU; the scaling story on
  synthetic ML-3B is the stretch.
- **Proxy axis:** dataset size (ML-1M as proxy for Amazon Books) + model size. **Rare-signal
  trap:** MovieLens saturation — calibrate that a proxy win on ML transfers to Amazon Books
  before trusting the proxy at all (this is exactly the "proxy is an RNG with a GPU bill" risk).

## 9. "Beat it" hypotheses

1. **Readout thesis on recsys** — HSTU already showed a 7× readout swing in the FM campaign's
   own reco task; sweep readout heads (last-position / attention-pooled / MLP+context) on the
   frozen HSTU representation before judging it. Flag `readout`.
2. **Context-length sweet spot** — HSTU's efficiency lets you push sequence length; find the
   per-dataset peak (the FM campaign's "context is a config knob" thesis). Flag `seq_len`.
3. **Frequency-prior blend** — the FM reco win came from blending a memorization prior with the
   model (0.1·model + 0.9·freq beat both). Test whether an HSTU+popularity blend beats HSTU
   alone on the long-tail slice. Flag `freq_blend`.

## 10. Budget

- **Wave 1.** Cheapest real gate in the program after the LLM micro-proxies.
- **GPU-hours:** smoke <0.5 · proxy (ML-1M/20M sweep, ~10 configs) ~10–20 · full (Amazon Books
  HSTU-large, a few seeds) ~15–25 · controls ~5 · **total ~40–55**.
- **GPU tier:** A10G/L4, 24GB sufficient; spot ON. **~$40 on-demand / ~$20 spot.**
- **Approval:** envelope only (Wave 1).

## 11. Controls

- **Shuffled-interaction control** → metrics collapse to popularity floor.
- **Strongest cheap baseline:** a tuned SASRec *and* a most-popular / most-recent frequency
  baseline (the FM campaign learned the frequency baseline is brutal in recsys — beat the
  honest floor, not the weakest quoted).
- **Leak-audit trigger:** any HR@10 implausibly above the published table → temporal leakage
  across the split.

## 12. Risks

MovieLens near-saturation (lead with Amazon Books); torchrec/fbgemm_gpu install friction;
sampled-softmax vs full-softmax eval mismatch is a classic recsys number-inflation bug — pin
the eval negative-sampling exactly. No checkpoints ship, so budget the from-scratch pipeline
gate as real (not free) compute.
