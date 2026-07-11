# Campaign 03 — DLRM-DCNv2 CTR (MLPerf) reproduction, then efficiency/quality push

> **Status: SEED (not started).** The large-data campaign — a terabyte-scale click dataset and
> ~100GB sparse embedding tables. The most externally-audited gate in the program.

## 1. One-liner

Reproduce the MLPerf Training DLRM-DCNv2 gate (validation AUROC 0.80275 within one epoch on
the Criteo click logs) in a Ray Train pipeline, then push either efficiency (GPU-hours to the
gate) or quality (AUC beyond it).

## 2. Vertical & why Anyscale

- **Vertical:** ad tech / e-commerce ranking (CTR prediction).
- **Why Anyscale:** the "large data" story — 3.8TB materialized dataset, ~91–100GiB collective
  embedding tables. Ray Data heterogeneous streaming ingest + Ray Train distributed
  (TorchRec/DDP) with model-parallel embedding sharding. Echoes Amazon's 82% cost cut / 12×
  larger datasets. The reference itself is a from-scratch benchmark, so it's a pure
  reproduce-the-number exercise with a hard, audited target.

## 3. Reference

- **Repo:** github.com/mlcommons/training → `recommendation_v2/torchrec_dlrm` — code Apache-2.0.
  Alive and maintained (not the retired v1 DLRM).
- **Ships:** training code ✓ (TorchRec `dlrm_main.py`, DDP/torchx); **no checkpoints** (the
  benchmark trains from scratch — pipeline gate only).
- **Ray-native?** No — TorchRec + DDP/torchx. Rayification = wrap in Ray Train with the
  embedding-table sharding strategy; Ray Data for the streaming input.
- **Reproduction landscape:** an MLPerf reference with dozens of audited submissions — the
  cleanest, most independently-verified gate in this whole program.

## 4. The gate

- **Decision metric:** validation AUROC.
- **Published target (defined in the MLPerf reference rules):** **≥ 0.80275 within one epoch**
  over 4,195,197,692 samples. The repo's own preliminary runs hit 0.8025–0.8032.
- **Guard metrics:** GPU-hours to the gate, throughput (samples/s), peak host + device memory.
- **Pipeline gate:** hit ≥0.80275 in one epoch through the Ray harness — that IS the
  reproduction (no artifact gate since no checkpoints).

## 5. Pinned eval

The MLPerf-defined validation split + AUROC implementation + the one-epoch sample count, frozen
as the protocol module. Positives are plentiful (CTR) — point estimates are stable; still report
a CI. Env pin: TorchRec + CUDA versions in the job image.

## 6. Data

- **Criteo click logs**, preprocessed multi-hot, **3.8TB materialized**, hosted by MLCommons via
  the R2 Downloader (no raw preprocessing needed — raw path needs 700GB RAM + 1–2 days).
- **⚠️ License: Criteo dataset is CC BY-NC-SA 4.0 — NON-COMMERCIAL.** This blocks using results
  in Anyscale *marketing* without legal sign-off. **PI decision required before starting:**
  either (a) treat this as an internal engineering benchmark only, or (b) substitute a
  commercially-usable CTR dataset (e.g. Avazu, or a synthetic multi-hot generator) and re-anchor
  the gate. Flagged here so it's decided at pre-registration, not discovered mid-campaign.
- **Input audit:** the 26 categorical + 13 dense feature schema, multi-hot cardinalities, the
  hashing/frequency-threshold on rare categories (a silent change here moves AUC).

## 7. Rayification

| Stage | What | Ray lib |
|---|---|---|
| ingest | stream 3.8TB multi-hot parquet, no materialization | Ray Data streaming |
| train | DLRM-DCNv2, sharded embedding tables + dense DDP | Ray Train (TorchRec integration) |
| eval | AUROC over the validation split | Ray Data batch scoring |

The load-bearing Ray story is **model-parallel embedding sharding across nodes** while the dense
arch is data-parallel — the thing that makes 100GiB tables tractable without one giant host.

## 8. Fidelity ladder

- **smoke:** a few million rows, tiny tables, 1 GPU — the sharding + streaming code runs.
- **proxy:** a subsampled day / reduced embedding cardinality, single 8-GPU node; validate the
  pipeline reaches a *proportionate* AUC and rank efficiency ideas.
- **full:** the full one-epoch MLPerf run on 8×A100 (~5 GPU-hr at the fast config; ~43min–4h
  wall depending on batch).
- **Proxy axis:** data fraction + embedding cardinality. **Rare-signal trap:** rare-category
  features carry CTR signal — a naive frequency cutoff at proxy scale changes which features
  matter; hold the cutoff fixed and subsample rows, not categories.

## 9. "Beat it" hypotheses

1. **Efficiency:** fewer GPU-hours to 0.80275 via better sharding / larger batch with a
   compensating optimizer-step budget (heed the FM lesson: a batch-size bump that halves
   optimizer steps degrades quality — ablate it). Flag `shard_strategy` / `batch`.
2. **Quality:** does a readout/interaction-layer change (deeper DCN cross, an attention
   interaction) push AUC past the gate at equal compute? Flag `interaction`.

## 10. Budget

- **Wave 2** (large data + storage, but a single full run is cheap in GPU-hours).
- **GPU-hours:** smoke <1 · proxy (subsampled, several efficiency ideas) ~15–30 · full
  (8×A100, ~5 GPU-hr/attempt × a few attempts + seeds) ~25–40 · **total ~50–80**. Plus **3.8TB
  storage** — a real line item; budget object-store/EBS cost, not just GPU.
- **GPU tier:** A100-80G (embedding memory); spot ON. **~$200–350 on-demand.**
- **Approval:** envelope + each full run + the **dataset-license decision** (§6).

## 11. Controls

- **Shuffled-label control** → AUC collapses to 0.5.
- **Strongest cheap baseline:** logistic regression / a small factorization machine on the same
  features (the "honest floor" — DLRM must clear it convincingly, not marginally).
- **Leak-audit trigger:** AUC materially above 0.803 in one epoch → a preprocessing/leakage bug.

## 12. Risks

The **CC-BY-NC dataset license** is the headline risk (see §6 — decide first). 3.8TB storage and
egress cost; TorchRec-on-Ray integration maturity (verify sharding correctness on the proxy
before the full run); host-RAM for the embedding tables. AUC gains beyond 0.803 are hard and
often illusory — anchor claims to the audited gate.
