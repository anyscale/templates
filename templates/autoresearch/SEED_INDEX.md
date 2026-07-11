# Seed index — the candidate-campaign menu

Nine candidate reproduce-then-beat campaigns plus the finished worked example, spanning
Anyscale's five service verticals and four adjacent good-fit domains. Each has a
pre-registered plan in `campaigns/`. Ordering here is a **recommendation**, not a commitment —
the PI picks and sequences (`BUDGET_POLICY.md`, `CLAUDE_WITH_ANYSCALE.md` §8).

## Selection criteria (why these nine)

Every seed had to clear four bars, in priority order:
1. **A real reproduction gate from shipped code** — a public repo with training/eval code and,
   ideally, checkpoints, and a published number defined in an eval script (Iron Rule #1). No
   gate, no campaign.
2. **Vertical fit** — covers a vertical Anyscale services (fintech, e-commerce, AI-natives,
   autonomy/physical-AI, creatives) or an adjacent domain that's a natural fit (ad tech, life
   sciences, time-series, speech).
3. **A Ray-shaped pipeline** — the rayification is either real leverage (heterogeneous
   Data+Train+Serve) or the reference is already Ray-native (verl).
4. **A portable "beat it" thesis** — ideally the readout thesis, context-as-a-config-knob, or
   late→joint fusion carries over, so the program compounds learning across campaigns.

## The menu

| # | Campaign | Vertical | Ray workload | Reference (repo) | Decision metric | Wave | Confidence |
|---|---|---|---|---|---|---|---|
| 00 | Transaction-FM fraud **(DONE)** | Fintech | Data+Train+Serve | NVIDIA TabFormer | AP | — | ✅ won |
| 01 | HSTU generative recsys | E-commerce/media | Train+Data | meta-recsys/generative-recommenders | HR@10 / NDCG@10 | **1** | high |
| 02 | Zero-RL reasoning | AI-natives/agentic | Train+Data+Core (verl) | hkust-nlp/simpleRL-reason (+verl) | pass@1 | 1→2 | high |
| 03 | DLRM-DCNv2 CTR | Ad tech | Train+Data (big) | mlcommons/training | AUROC | **2** | very high (audited) |
| 04 | BEIR dense retrieval | Enterprise search/creatives | Data (batch-embed) | beir-cellar/beir + E5 | nDCG@10 | **1** | high |
| 05 | Prithvi geospatial FM | Autonomy/climate | Data+Train | NASA-IMPACT/hls-foundation-os | mIoU | **2** | medium |
| 06 | OpenVLA robotics | Physical AI | Train + sim-eval fleet | openvla/openvla | success rate | **3** | medium |
| 07 | Chronos time-series FM | Retail/energy/supply | Data (batch-inf)+Train | amazon-science/chronos-forecasting | MASE / WQL | 1→2 | high |
| 08 | ESM-2 protein readout | Life sciences | Data (batch-embed)+probe | facebookresearch/esm + FLIP | Spearman / acc | **1** | high |
| 09 | ASR fine-tune | Contact center/media | Train + Data (batch) | Whisper / wav2vec2 + LibriSpeech | WER | **2** | high |

## Recommended first wave (ship a cheap, credible result fast)

Start with the campaigns that are cheapest, highest-confidence, and reuse the FM machinery most
directly — get a second vertical on the board before committing to an OpenVLA-scale run:

1. **04 BEIR retrieval** — the cheapest gate in the program. Pure Ray Data batch-embedding
   (Anyscale's most-winnable workload, echoes Canva/Mirakl), frozen-encoder eval, no training to
   reproduce. Fastest path to "the harness works on a new vertical."
2. **08 ESM-2 protein readout** — the **purest test of the readout thesis** on a new domain:
   frozen embeddings + a linear/MLP probe ladder on a FLIP split, exactly the FM reco pattern.
   Cheap (embed + probe, no pretraining), and life-sciences is a high-value vertical.
3. **01 HSTU recsys** — the direct sequential-encoder analog; repo already cloned; carries the
   readout + context-length + frequency-blend theses over verbatim.

These three are all Wave 1, together ~120–160 GPU-hours (~$35–80 spot at the A10G spot rate),
and each exercises a
different Ray workload (batch-embed, embed+probe, distributed-train). They also stress the
harness (`REQUIREMENTS.md` R1/R2) enough to shake out its design before a Wave-2/3 campaign
depends on it.

## Vertical coverage check

| Anyscale vertical | Covered by |
|---|---|
| Fintech | 00 (done) |
| E-commerce / delivery | 01 (recsys), 03 (CTR ranking) |
| AI-natives | 02 (RL), 04 (retrieval) |
| Autonomy / physical AI | 05 (geospatial), 06 (VLA) |
| Creatives / productivity | 04 (embedding/search) |
| **Adjacent good-fit** | 03 (ad tech), 07 (time-series: retail/energy), 08 (life sciences), 09 (speech: contact center) |

All five Anyscale verticals are covered; the four adjacents extend into domains Anyscale
doesn't yet playbook but where the same Ray patterns (batch-embed, distributed-train,
batch-inference) obviously fit.

## How the waves gate spend

Per `BUDGET_POLICY.md`: Wave 1 campaigns (≤60 GPU-hr each) run largely unattended within a
PI-approved envelope. Wave 2 (60–400, e.g. the 7B RL full run at ~120 GPU-hr) needs PI sign-off
on each full run. Wave 3 (400–2,000, e.g. OpenVLA's fine-tune + robustness sweep) needs sign-off
on the envelope, each full run, and any
multi-node request. The program-level cap is the sum of active envelopes against the quarterly
allocation.
