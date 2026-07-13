# Seed index — the candidate-campaign menu

Ten candidate reproduce-then-improve campaigns (plus the finished worked example), spanning a
range of ML domains and Ray workload shapes so the loop gets exercised broadly. Each has a
pre-registered plan in `campaigns/`. Ordering here is a **recommendation**, not a commitment —
you pick and sequence.

## Selection criteria

Every seed had to clear four bars, in priority order:
1. **A real baseline gate from shipped code** — a public repo with training/eval code and,
   ideally, checkpoints, and a published number defined in an eval script. No gate, no campaign.
2. **Domain & workload diversity** — different task types and Ray workload shapes (batch-embed,
   distributed-train, big-data streaming, RL, sim-eval), so the harness is stress-tested broadly
   rather than proving the same pattern repeatedly.
3. **A Ray-shaped pipeline** — the rayification is either real leverage (heterogeneous
   Data + Train + Serve) or the reference is already Ray-native (verl).
4. **A portable improvement thesis** — ideally the readout thesis, context-as-a-config-knob, or
   late→joint fusion carries over, so learnings compound across campaigns.

## The menu

| # | Campaign | Domain | Ray workload | Reference (repo) | Decision metric | Wave | Confidence |
|---|---|---|---|---|---|---|---|
| 00 | Transaction-FM fraud **(DONE)** | Tabular / fraud | Data+Train+Serve | NVIDIA TabFormer | AP | — | ✅ done |
| 01 | HSTU generative recsys | Recsys | Train+Data | meta-recsys/generative-recommenders | HR@10 / NDCG@10 | **1** | high |
| 02 | Zero-RL reasoning | LLM post-training | Train+Data+Core (verl) | hkust-nlp/simpleRL-reason (+verl) | pass@1 | 1→3 | high |
| 03 | DLRM-DCNv2 CTR | CTR / big-data tabular | Train+Data (big) | mlcommons/training | AUROC | **2** | very high (audited) |
| 04 | BEIR dense retrieval | Retrieval / embeddings | Data (batch-embed) | beir-cellar/beir + E5 | nDCG@10 | **1** | high |
| 05 | Prithvi geospatial FM | Geospatial / segmentation | Data+Train | NASA-IMPACT (TerraTorch) | mIoU | **2** | medium |
| 06 | OpenVLA robotics | Robotics / VLA | Train + sim-eval fleet | openvla/openvla | success rate | **3** | medium |
| 07 | Chronos time-series FM | Time-series | Data (batch-inf)+Train | amazon-science/chronos-forecasting | MASE / WQL | 1→2 | high |
| 08 | ESM-2 protein readout | Protein / embeddings | Data (batch-embed)+probe | facebookresearch/esm + FLIP | Spearman / acc | **1** | high |
| 09 | ASR fine-tune | Speech | Train + Data (batch) | Whisper + LibriSpeech | WER | **2** | high |
| 10 | Pathology FM readout | Digital pathology / embeddings | Data (batch-embed)+probe | Phikon / Virchow + public benchmarks | AUROC | 1→2 | high |

## Recommended first campaigns (cheapest, highest-confidence)

Start with the campaigns that are cheapest, highest-confidence, and exercise the harness on
distinct workload shapes — get a clean, credible first result before committing to a heavy run:

1. **08 ESM-2 protein readout** — the **purest test of the readout thesis**: frozen embeddings +
   a linear/MLP probe ladder on a FLIP split. Cheap (embed + probe, no pretraining), and the FLIP
   literature already documents a large readout swing to reproduce.
2. **04 BEIR retrieval** — the cheapest baseline gate: pure Ray Data batch-embedding, frozen
   encoder, no training to reproduce. Fastest path to "the loop holds together end to end."
3. **01 HSTU recsys** — the direct sequential-encoder analog; carries the readout + context-length
   + frequency-blend theses over verbatim.

All three are Wave 1, together ~120–160 GPU-hours (~$35–80 spot), and each exercises a different
Ray workload (embed+probe, batch-embed, distributed-train) — enough to shake out the harness
(`REQUIREMENTS.md` R1/R2) before a heavier campaign depends on it.

## Workload-shape coverage

The menu deliberately spans distinct Ray/compute patterns so the harness is validated on more
than one shape: **batch-embed** (04, 08, 10), **embed+cheap-probe** (08, 10), **distributed
train** (01, 03, 05), **big-data streaming** (03), **RL co-location** (02), and **parallel
sim-eval** (06). Breadth of *workload*, not domain prestige, is the point.

## How the waves gate spend

Waves are cost tiers in A10G-equivalent hours (`BUDGET_POLICY.md`): Wave 1 (≤60) runs largely
unattended within an approved envelope; Wave 2 (60–400) needs sign-off on each full run; Wave 3
(400–2,000, e.g. the 7B-RL full run at ~660 A10G-eq, or OpenVLA's fine-tune + robustness sweep)
needs sign-off on the envelope, each full run, and any multi-node request.
