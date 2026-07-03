---
name: fintech-tfm-series-state
description: State of the fintech_transaction_fm notebook series (Parts 1-9) and known open issues
metadata: 
  node_type: memory
  type: project
  originSessionId: 27bfccd3-8165-4406-85f6-09e28afd43a0
---

`templates/templates/fintech_transaction_fm` is a 9-part notebook series (01–09), each runnable at `mini` (CPU) via papermill. The scale name was renamed `smoke`→`mini` everywhere.

**2026-07-03 — big re-architecture to NVIDIA's transaction-FM blueprint (committed, branch `zgarner_transaction_foundation_model`):** flat tokenizer (12 tokens/txn, shared vocab 6259) + Llama causal decoder (29M params: hidden 512, 8 layers, GQA 8/2, SwiGLU 1408, rope 5e5) + next-token pretrain + last-token embedding. Replaced the old field-split/MLM design. See CHANGES.md (authoritative RESUME-HERE at top). Goal: beat NVIDIA fusion **PR-AUC 0.1755** on real IBM TabFormer (24.4M txns, 0.12% fraud); old arch best was 0.0301. Metric = PR-AUC (AP) at ~0.1% prevalence.

**GPU problem SOLVED (2026-07-03):** last session was stuck on T4s (A10G capacity-failing, ~16h/epoch). After Zach rebooted, the workspace launches **A10G (23.7GB)** cleanly, up to 8 for pretrain. Uncommitted fixes on branch: `job_config.yaml` (g4dn/T4→g5/A10G), `requirements.txt` (+transformers/accelerate).

**IMPORTANT — `/mnt/cluster_storage` PERSISTS across cluster restarts** (workspace-scoped, NOT ephemeral as CHANGES.md feared). On resume, `source/` CSV, `raw/full`, `tokenized/full` (6.7G, 64,561 pretrain seqs, ~5M eval) all survived → skip nb 01/02/03, run only 04→06. Memory levers present: bf16 autocast (`pretrain.py`), `gradient_checkpointing_enable()` (`model.py`), expandable_segments.

**Ray perf/scale storylines** — a core PURPOSE of this series is to SHOW Ray addressing performance/scale issues, so capture real ones as they arise. `PERFORMANCE.md` (new, 2026-07-03) catalogs them (symptom→cause→fix→what-Ray-did→which-notebook), source material esp. for nb 09 "scaling up". Captured so far: T4-can't-fit-seq4096 (→A10G+bf16+grad-ckpt), Ray Train worker-startup-timeout auto-retry, embed fp32 OOM (→bf16+batch 256→64), **stranded GPUs (24 provisioned / 16 used = 67% — the "match resources to workload" story)**, open item: per-GPU SM saturation of the 16 busy GPUs (batch 64 maybe too small; test 128; needs nvidia-smi measurement), Ray Data CPU→GPU streaming backpressure (a win), embed throughput-bound (linear scale-out), driver-OOM-on-scoring + CPU-path autoscaler-assertion (both already fixed).

**In progress (2026-07-03 ~11:53):** first full run of new arch — headless chain `scripts/03_pretrain.py→04_extract_embeddings.py→05_train_downstream.py --scale full` on 8×A10G, launched from scratchpad `run_full_tail.sh`. ~2-4h pretrain + embed + downstream. Then read 06 fusion PR-AUC vs 0.1755. Stage scripts read persisted disk artifacts and DON'T wipe (only `run_pipeline.py` wipes).

Parts: 1 Setup, 2 Load/explore, 3 Tokenize, 4 Pretrain, 5 Embed, 6 Downstream fraud, 7 Serving, 8 Run-on-Anyscale (fused `run_pipeline.py` + Job), 9 Scaling-up (4 bottlenecks: data>node, GPU-starved-by-CPU, stage-boundary I/O, serve latency).

`src/` additions made for the notebooks: `generate_data.generate_dataset_distributed`/`generate_cards` (distributed synthetic gen), `pretrain.save_checkpoint`, `tokenizer.decode_static_fields`/`decode_dynamic_window`, `embed` actor pool fix (`num_cpus=1` + `min_size=1,max_size=N` — the CPU-path inf-bundle autoscaler assertion). `scripts/generate_big.py` added.

Open issues to resolve in review (as of 2026-06-25):
- **CI (`tests/fintech_transaction_fm/tests.sh`) still runs `README.ipynb`** (monolith), not the series — and README sets `SCALE="small"` (GPU) while its prose says it runs at `mini`. Decide whether to point CI at the series.
- **Part 8 `run_pipeline.py` is destructive** (wipes all stage outputs, streams tokenized through memory so no tokenized Parquet remains). Part 9 has a guard that regenerates tokenized if missing. Decide if that UX is right or reorder/soften.
- **Bottleneck 2 (GPU idle) has no inline measurement** (no GPU at mini) — config + arithmetic + referred-out Job only.
- **Distributed generator is slow** (~150s/10k cards; per-card Python loop) — fine for demo, could vectorize.

Verified green via papermill this session: Parts 3,4,5,6,7,9. Part 8 verified via headless `run_pipeline.py` (its only executable cell). See [[fintech-tfm-working-style]].
