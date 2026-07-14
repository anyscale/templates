# Stepping through the pipeline in a debugger

Launch configs live in the repo-root `.vscode/launch.json` (Cursor and VS Code
both read it). Two environments:

## On the workspace (real data, one GPU) — the main path

1. Connect Cursor to the workspace over SSH (`anyscale workspace_v2 ssh config`
   once, then Cursor → Remote-SSH → the workspace host) and open the repo.
2. Pick any `fm ws:` launch config. They all use `--scale dbg1gpu`
   (`configs/dbg1gpu.yaml`): small's real-TabFormer recipe with **one** Ray
   Train worker, **one** embedding actor, one whole GPU, 4 epochs — so there is
   exactly one worker process per stage and a debug loop is minutes.
3. Run stages in order (01 → 02 → 03 → 04 → 05 → 06); each reads the previous
   stage's Parquet from `/mnt/cluster_storage/transaction-fm`, so you can
   re-step any single stage without rerunning the rest.

**The one caveat:** breakpoints you click in the gutter only bind in the
*driver* process. Stage 01, stage 05, and all orchestration code are
driver-side — plain stepping works. But the tokenizer's `map_groups` UDF
(stage 02), the Ray Train loop (stage 03), and the embedding actor
(stage 04) execute in a Ray **worker** process. To step inside those:

- put a literal `breakpoint()` on the line you care about (e.g. inside
  `tokenize_group` in `src/tokenizer.py`, or `train_func` in
  `src/pretrain.py`),
- run the stage with `RAY_DEBUG=1` (the `fm ws:` configs already set it),
- when it hits, the **Ray Distributed Debugger** panel (preinstalled in
  Anyscale's VS Code/Cursor setup; else install the `anyscale.ray-distributed-debugger`
  extension) lists the paused task — click it to attach and step.

Because `dbg1gpu` runs one worker per stage, there's exactly one task to
attach to — this is why the single-GPU config exists. Post-mortem on a worker
exception: also `RAY_DEBUG=1`, plus `RAY_DEBUG_POST_MORTEM=1`.

To understand tensor *shapes* end-to-end, the cheapest path is stage 03 with a
`breakpoint()` at the top of the inner loop in `src/pretrain.py`, then inspect
`batch` — every `d_*` field is `[B, seq_len]`, statics `s_*` are `[B]`, and
`model._embed` fuses them to `[B, seq_len, d_model]`.

## Locally (laptop, synthetic smoke data)

`fm local:` configs use the template venv at
`templates/fintech_transaction_fm/.venv`. To recreate it:

```bash
cd templates/fintech_transaction_fm
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
    'ray[data,train,serve]==2.55.0'   # >=2.56 breaks the hash shuffle; <=2.49 lacks expr filters
brew install libomp                    # xgboost's OpenMP runtime
```

macOS gotcha: stage 05 segfaults when xgboost's OpenMP meets torch's
(`OMP: Error #179`). The launch configs set `OMP_NUM_THREADS=1`, which fixes
it; do the same for manual runs.
