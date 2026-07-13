# Campaign 10 — Computational-pathology FM: reproduce a probe number, then beat the readout + $/slide

> **Status: SEED (not started).** A digital-pathology testbed: image-based (whole-slide images),
> so a different modality + workload from the protein/sequence and text campaigns. Numbers below
> are *targets to confirm from code*, not trusted from papers.

## 1. One-liner

Reproduce a published pathology-FM downstream number (frozen tile embeddings + a probe/MIL head
on a public benchmark), then improve it two ways: a better **readout** (MIL aggregator swept on
the frozen embeddings) at held backbone, and a lower **$/slide** for the embedding pass via Ray
Data — tile-embedding throughput being the recurring cost this workload pays.

## 2. Why this is a good autoresearch testbed

- **What makes it clean:** frozen pathology-FM tile embeddings + a probe/MIL head reproduce a
  public benchmark number, so a baseline is cheap to establish; the MIL aggregator is a genuine
  readout lever; public patch-level tasks give a cheap gate before the slide-level pipeline.
- **Ray substrate it exercises:** the batch-embedding pattern at scale — a gigapixel slide is
  10k–100k tiles, so embedding a cohort is a large Ray Data `map_batches` + ActorPoolStrategy pass
  (CPU tile/normalize → GPU ViT embed, fractional-GPU, scale-to-zero); extract once, sweep readouts.

## 3. Reference

- **Backbone (frozen):** pick by **license, per the `commercial_use` gate** (if your intended
  use is commercial, research-only weights are disqualified up front):
  - **Phikon / Phikon-v2** (Owkin) — **Apache-2.0**, ships weights + eval. ✅ safe default.
  - **Virchow-1** (Paige) — **Apache-2.0**, ships weights.
  - **UNI / UNI2** (Mahmood Lab) — **CC-BY-NC-ND, research-only, gated**. ⚠️ research use only.
  - **Virchow2** (Paige) — **CC-BY-NC-ND**. ⚠️ research use only.
  - **CONCH** — vision-language; a Dec-2025 benchmark finds it out-generalizes UNI/Virchow-2 on
    some tasks (check its license against your intended use).
- **`commercial_use`: `yes`** on the Apache-2.0 path (Phikon/Virchow-1); **`no`** for UNI/Virchow2.
  This is the `REQUIREMENTS.md` #7 gate — decide the backbone by license *before* spend.
- **Ships:** backbone weights ✓; the downstream probe/MIL + eval is standard (HF + public
  benchmark loaders). No pretraining to reproduce → this is a reproduce-the-**eval** campaign
  (frozen encoder), like ESM-2 and BEIR.
- **Ray-native?** No — PyTorch + openslide/timm. The tiling+embedding rayification is pure upside.
- **Reproduction landscape:** heavily benchmarked (PathBench, PathOrchestra). **Contamination is
  the headline trap:** several FMs pretrained on TCGA/CAMELYON/PANDA, so evaluating them on those
  same slides is leaky. UNI deliberately excluded public benchmarks from pretraining precisely to
  be a clean baseline — mirror that discipline: prefer an eval the chosen backbone did *not* see.

## 4. The gate

- **Decision metric:** balanced accuracy / AUROC on a downstream task (patch-level for the cheap
  gate; slide-level MIL for the full run).
- **Published to match (confirm from the model card / PathBench eval code):** e.g. PatchCamelyon
  (PCam) linear-probe AUROC ≈ high-0.9s for a strong FM; TCGA-NSCLC subtyping MIL AUROC per the
  backbone's paper. Pin the exact split + probe protocol; trust the eval script, not the abstract.
- **Guard metrics:** per-class balanced acc; embedding throughput (tiles/s) and **$/slide** (the
  efficiency beat); AUROC on a second task (transfer breadth).
- **Artifact gate:** run the backbone's shipped eval on its shipped weights → match its number.
- **Pipeline gate:** reproduce it through the Ray tile-embed + probe harness (one protocol module).

## 5. Pinned eval

The exact benchmark split (PCam test / a TCGA fold) + the frozen layer read + tile size (224px) +
magnification (20× vs 40×) + stain-normalization setting + the probe/MIL protocol + metric impl.
Content-hash the tile/label manifest. Slide-level tasks have few slides → **MIL AUROC needs a CI
over slides + seeds** (few-positives territory — see `harness/metrics.positives_warning`).

## 6. Data

- **Cheap gate (patch-level, low friction, public):** PatchCamelyon (~327k 96px patches, binary
  tumor), NCT-CRC-HE-100K / CRC-100K (colorectal tissue types). HF downloads.
- **Full (slide-level, gigapixel — the Anyscale story):** CAMELYON16 (lymph-node metastasis WSIs),
  TCGA-NSCLC / TCGA-RCC subtyping, PANDA (prostate grading). Public but **large** (WSIs are GBs
  each; storage + egress is a real line item).
- **Input audit:** stain normalization (Macenko/Vahadane), magnification/MPP, tile size + overlap,
  background/tissue masking, and **which hidden layer** is read — all silently move the number.

## 7. Rayification — stage → Ray library

| Stage | What | Ray lib |
|---|---|---|
| ingest | WSI (openslide) → tissue-masked tiles | Ray Data (binary/large-file) |
| normalize | stain normalization per tile | Ray Data `map_batches` (CPU) |
| **embed** | frozen pathology-FM forward over tiles | **Ray Data `map_batches` + ActorPoolStrategy, fractional GPU** |
| aggregate | tiles → slide embedding (MIL) | Ray tasks / driver |
| probe/eval | linear/MIL head → AUROC | driver |

The embed stage is where the substrate earns its keep: millions of tiles, CPU-read + GPU-embed in one streaming
pass, scale-to-8+-GPU then to zero. Extract once, sweep many readouts (probe ladder).

## 8. Fidelity ladder

- **smoke:** a few slides / a PCam subset, 1 GPU — tile→embed→probe→score runs.
- **proxy:** full PCam or NCT-CRC (patch-level, cheap); rank MIL/readout ideas + pipeline variants.
- **full:** TCGA/CAMELYON slide-level MIL (the gigapixel embedding pipeline) — where the $/slide
  beat shows and the slide-level claim lives.
- **Proxy axis:** patch-level tasks proxy the slide-level pipeline for *readout* ranking; a fixed
  slice of the big WSI cohort proxies *throughput*. **Rare-signal trap:** slide-level tasks have
  few slides and class imbalance — hold out whole slides (never tiles from the same slide → leak),
  keep positive slides in the subsample.

## 9. "Beat it" hypotheses (each: mechanism + cheapest validating run)

1. **MIL readout sweep (primary, the readout thesis).** Frozen backbone; sweep the aggregator:
   mean-pool → attention-MIL (ABMIL) → CLAM → TransMIL. Mechanism check: the aggregator is *not*
   jointly trained into the frozen backbone, so it's a genuine free lever (unlike a contrastively-
   baked pooling) — the thesis should hold here. Cheapest: extract tile embeddings once, sweep
   heads on patch/small-slide sets. Flag `mil_readout`.
2. **$/slide embedding-efficiency demo.** Materially lower $/slide for the tile-embedding pass via
   Ray Data heterogeneous streaming + fractional-GPU packing, AUROC held within CI. Baseline =
   a competent single-GPU tile-embed loop at the same budget (`REQUIREMENTS.md` #8) — an
   efficiency result, not "beat the paper." The Anyscale-native win. Flag `embed_pipeline`.
3. **Backbone × task fit.** Which frozen backbone (Phikon vs Virchow vs CONCH) probes best per
   task — the Dec-2025 finding says it varies. A cheap, decision-useful comparison. Flag `backbone`.

## 10. Budget envelope

- **Wave 1** for the patch-level probe (PCam/NCT-CRC, A10G) → **$100 is a GO** (embed a public
  patch set once, sweep readouts ~free). **Wave 2** for the slide-level TCGA/CAMELYON MIL + the
  gigapixel embedding + storage.
- **Est A10G-equivalent hours:** smoke <0.5 · proxy (PCam embed + readout sweep) ~5–8 · full
  (slide-level MIL + WSI embedding, a couple variants) ~30–60 · controls ~5 · **total ~40–75**.
- **GPU tier:** A10G/L4 (ViT tile embedding packs fractionally); spot ON. **~$15–30 spot** for the
  wave-1 patch probe; wave-2 more (WSI volume). Run `harness/feasibility.py` before committing.
- **Approval:** envelope (wave 1); + each full run + the WSI storage line (wave 2).

## 11. Controls

- **Shuffled-label control** → AUROC collapses to 0.5 (`harness/canary.shuffled_label_control`).
- **Strongest cheap baseline:** ImageNet-pretrained ViT/ResNet tile embeddings + the same probe —
  the pathology FM must clear the non-pathology-pretrained floor convincingly.
- **Contamination guard:** prefer a backbone/eval pair with no pretraining overlap; flag any
  model that pretrained on the eval's public slides. (This is the FM-contamination gap noted in
  `critiques.md`, made concrete here.)
- **Leak-audit trigger:** slide-level AUROC implausibly high → same-slide tile leakage across the
  split, or a magnification/normalization mismatch.

## 12. Risks / repro gotchas

- **License** is the #1 risk if the intended use is commercial — anchor on Apache-2.0
  (Phikon/Virchow-1), not the research-only UNI/Virchow2. Decide before spend (the `commercial_use` gate).
- **Contamination** (backbone pretrained on the public eval slides) — pick the pair carefully.
- **WSI data volume** (gigabytes/slide) — storage + egress is a real line item, not just GPU.
- **Stain/scanner domain shift** and magnification are the silent-failure surface (input audit).
- **Few slides** at the slide level → MIL AUROC has wide CIs; CI-mandatory, hold out whole slides.
