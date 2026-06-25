---
name: fintech-tfm-series-state
description: State of the fintech_transaction_fm notebook series (Parts 1-9) and known open issues
metadata: 
  node_type: memory
  type: project
  originSessionId: 27bfccd3-8165-4406-85f6-09e28afd43a0
---

`templates/templates/fintech_transaction_fm` is being split from the monolith `README.ipynb` into a 9-part series (01–09), each verified runnable at `mini` (CPU) via papermill. The scale name was renamed `smoke`→`mini` everywhere.

Parts: 1 Setup, 2 Load/explore, 3 Tokenize, 4 Pretrain, 5 Embed, 6 Downstream fraud, 7 Serving, 8 Run-on-Anyscale (fused `run_pipeline.py` + Job), 9 Scaling-up (4 bottlenecks: data>node, GPU-starved-by-CPU, stage-boundary I/O, serve latency).

`src/` additions made for the notebooks: `generate_data.generate_dataset_distributed`/`generate_cards` (distributed synthetic gen), `pretrain.save_checkpoint`, `tokenizer.decode_static_fields`/`decode_dynamic_window`, `embed` actor pool fix (`num_cpus=1` + `min_size=1,max_size=N` — the CPU-path inf-bundle autoscaler assertion). `scripts/generate_big.py` added.

Open issues to resolve in review (as of 2026-06-25):
- **CI (`tests/fintech_transaction_fm/tests.sh`) still runs `README.ipynb`** (monolith), not the series — and README sets `SCALE="small"` (GPU) while its prose says it runs at `mini`. Decide whether to point CI at the series.
- **Part 8 `run_pipeline.py` is destructive** (wipes all stage outputs, streams tokenized through memory so no tokenized Parquet remains). Part 9 has a guard that regenerates tokenized if missing. Decide if that UX is right or reorder/soften.
- **Bottleneck 2 (GPU idle) has no inline measurement** (no GPU at mini) — config + arithmetic + referred-out Job only.
- **Distributed generator is slow** (~150s/10k cards; per-card Python loop) — fine for demo, could vectorize.

Verified green via papermill this session: Parts 3,4,5,6,7,9. Part 8 verified via headless `run_pipeline.py` (its only executable cell). See [[fintech-tfm-working-style]].
