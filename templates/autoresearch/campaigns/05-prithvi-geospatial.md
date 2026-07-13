# Campaign 05 — Prithvi geospatial FM (IBM-NASA) reproduction, then a downstream beat

> **Status: SEED (not started).** A foundation-model-in-a-new-domain campaign (the
> `AUTORESEARCH.md` "satellite imagery" case) — geospatial segmentation. Medium confidence —
> the reference's margin over tuned baselines is contested.

## 1. One-liner

Reproduce Prithvi-EO-2.0's published burn-scar segmentation mIoU via TerraTorch, then test
whether the geospatial FM's advantage over a tuned U-Net is real and where a better
readout/fine-tune closes or widens it.

## 2. Why this is a good autoresearch testbed

- **What makes it clean:** ships fine-tune + eval + downstream checkpoints (artifact gate
  available); a contested margin over a tuned baseline makes the honest-reckoning finding valuable.
- **Ray substrate it exercises:** Ray Data heterogeneous ingest of large multi-band raster tiles
  (binary-file ingestion, no materialization) + Ray Train (porting a single-GPU Lightning fine-tune
  → TorchTrainer) + Ray Data batch inference over a large area of interest.

## 3. Reference

- **Repo:** `github.com/NASA-IMPACT/Prithvi-EO-2.0` (MIT) on `github.com/IBM/terratorch`
  (Apache-2.0, active). Weights `huggingface.co/ibm-nasa-geospatial` (all Apache-2.0).
  **⚠️ Do NOT start from `hls-foundation-os` — it is deprecated** (redirects to TerraTorch,
  pins stale MMSegmentation/MMCV).
- **Ships:** fine-tune + eval via TerraTorch CLI (`terratorch fit/test`, YAML-driven);
  backbones (300M/600M ±temporal); **fine-tuned downstream checkpoints also ship**
  (`Prithvi-EO-2.0-300M-BurnScars`, `-TL-Sen1Floods11`) → an artifact gate is available.
  Backbone pretraining is not the reproduction surface.
- **Ray-native?** No — PyTorch Lightning + TorchGeo (DDP).
- **Reproduction landscape:** **the margin is contested** — third-party work (arXiv:2504.17397;
  U-Prithvi, GIScience 2025) finds Prithvi beats a tuned U-Net by only small, task-dependent
  margins on burn scars and loses on biomass. That's both the vulnerability to probe and
  evidence the headline is a genuinely tuned number.

## 4. The gate

- **Decision metric:** mIoU on HLS Burn Scars.
- **Published — PICK ONE (they differ by protocol):** HF model card **93.00 test mIoU** (single
  `terratorch fit -c burn_scars_config.yaml`) vs paper (arXiv:2412.02732) **88.6** for 300M
  (10-seed × 10-trial HPO). Pre-register which is the gate; the card number is the cheap
  artifact gate, the paper number needs ~20+ runs.
- **Guard metrics:** burned-class IoU (87.52 on the card), per-class IoU, boundary quality.
- **Artifact gate:** run `terratorch test` on the shipped BurnScars checkpoint → match the card.
- **Pipeline gate:** reproduce it through the Ray Train fine-tune from the backbone.

## 5. Pinned eval

The Burn Scars test split the config points at (⚠️ the card's split description is muddled —
verify what the config actually loads), the mIoU implementation, fixed seed. Content-hash the
tile set. Segmentation mIoU is stable per-tile but scene-count is small — report a CI over tiles
and over seeds (the paper's own averaging is why its number is lower).

## 6. Data

- **HLS Burn Scars** — CC-BY-4.0, 2.65 GB, ~804 chips, **no Earthdata login, plain HF**.
  Optional second task: Sen1Floods11 (446 chips, public GCS) — weaker gate, paper-only mIoU.
- **Input audit (critical for imagery):** the 6 HLS bands used, their order, normalization
  stats, tile size/overlap, temporal-vs-single-frame input. This is the imagery analog of the
  FM's deleted-geography botch — a wrong band order or normalization silently craters mIoU.

## 7. Rayification

| Stage | What | Ray lib |
|---|---|---|
| ingest | multi-band GeoTIFF tiles → normalized chips | Ray Data (binary-file ingestion) |
| fine-tune | Prithvi backbone + segmentation head | Ray Train (`TorchTrainer`, port from Lightning) |
| eval | mIoU over the test tiles | Ray Data batch inference |
| (scale story) | inference over a large AOI raster | Ray Data batch inference, scale-to-zero |

## 8. Fidelity ladder

- **smoke:** a handful of chips, tiny crop, 1 GPU — ingest→fine-tune→eval runs.
- **proxy:** the full Burn Scars set at reduced resolution / fewer epochs; rank readout &
  augmentation ideas.
- **full:** the pinned Burn Scars config (and the seed-averaged protocol if chasing the paper #).
- **Proxy axis:** resolution + epochs (+ 300M vs 600M backbone). **Rare-signal trap:**
  site/scene domain shift — hold out whole scenes, never random tiles from the same scene (the
  imagery analog of temporal leakage); class rarity (burned pixels are rare) must survive the
  subsample.

## 9. "Beat it" hypotheses

1. **Readout/decoder swap** — the readout thesis for segmentation: does a stronger decoder head
   (UPerNet vs the shipped head) on the frozen Prithvi backbone lift mIoU? Flag `decoder`.
2. **Honest-baseline reckoning** — reproduce the contested comparison: a *properly tuned* U-Net
   on the same split. The durable claim is "FM beats a tuned baseline by X, holding data fixed"
   — measure X honestly (it may be small). Flag `baseline_unet`.
3. **Temporal vs single-frame** — does the temporal variant actually help on burn scars, or is
   the extra input dead (a la the FM's velocity features)? Flag `temporal`.

## 10. Budget

- **Wave 2.** A single fine-tune is cheap, but the honest protocol needs multiple seeds/runs.
- **GPU-hours:** smoke <0.5 · proxy (several decoder/aug ideas) ~10 · full (Burn Scars, seed
  sweep + U-Net baseline) ~20–40 · **total ~35–55**.
- **GPU tier:** A100 or A10G (300M backbone fits 24GB); spot ON. **~$120–200 on-demand.**
- **Approval:** envelope + each full run.

## 11. Controls

- **Shuffled-mask control** → mIoU collapses to class-prior.
- **Strongest cheap baseline:** a tuned U-Net / DeepLab on the same split (the contested
  comparison — this IS the point of the campaign, not an afterthought).
- **Leak-audit trigger:** mIoU above the paper's HPO number from a single run → a train/test tile
  overlap (same-scene leakage).

## 12. Risks

Starting from the deprecated `hls-foundation-os` (use TerraTorch). The card-vs-paper number
divergence (decide the gate up front). Contested margin means the "beat" may be modest —
frame it as an honest tuned-baseline reckoning, not a hype number. GeoTIFF/rasterio ingest and
band-normalization correctness are the silent-failure surface (input audit is load-bearing here).
