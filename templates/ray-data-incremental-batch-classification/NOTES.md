# Notes on the measurements

The evidence behind the README's figures, on synthetic data. The model is public, MIT-licensed and ungated (checked 2026-09-24).

## What ran where

- **AWS config** (head m5.2xlarge, one g5.2xlarge GPU worker with an A10G, Ray 2.58.0), 2026-09-24: two CI runs, each through all 23 cells the notebook then had. The second ran the current code. The first predates the join warmup, the label rewording and the input split; only its one-block packing row and its memory reading appear below.
- **14-CPU macOS laptop** (Ray 2.58.0, torch 2.13.0, transformers 5.17.0), 2026-09-24: three papermill runs as the code changed, with `CLASSIFY_ROWS=2000`, the lock install skipped and no `pip` in `ray.init`, since the lock holds Linux CUDA wheels. The classifier uses only CUDA, so every laptop pass ran on CPU.
- **Same laptop:** `run_pipeline.py` once at `NUM_ROWS=2000`, with the same substitution; its two classifier actors took 518 and 492 of the 1,010 new rows, with no "can launch at most" warning. `ZeroShotClassifier` outside Ray reproduced both wordings' label counts on the notebook's 2,000 rows.
- **Unmeasured:** the probe under worker loss, `job.yaml` and `run_pipeline.py` on a cluster, and the GCP config.

## Anti-join, 20,000 rows

| Where | Path | Untimed run | Timed, warm | Aggregators |
|---|---|---|---|---|
| AWS config | Hash join | 189.6 s | 6.0 s | 24 |
| AWS config | Probe | 4.0 s | 2.3 s | 0 |
| 14-CPU laptop | Hash join | 4.4 s | 3.8 s | 14 |
| 14-CPU laptop | Probe | 1.9 s | 2.6 s | 0 |

Every run kept the same 10,081 rows, 371 of them with a NULL key. The untimed join's 189.6 s covers the cluster growing from the GPU worker's 8 CPUs to 16: in the first CI run, Ray Data reported 24.0 GiB of memory active and requested during the join against the cluster's 18.9 GiB, and the autoscaler added a CPU worker, which then installed the lock. The cluster kept that size through the timed runs. The laptop's timed probe ran slower than its untimed one: at this size the seconds are noise.

## GPU packing, A10G, bfloat16, 10,081 rows

| Classifier input | One actor per GPU | Two per GPU | Ratio |
|---|---|---|---|
| 8 blocks, split as in the notebook | 267 rows/s | 398 rows/s | 1.49x |
| 1 block, first CI run, before the split | 260 rows/s | 287 rows/s | 1.10x |

With one block, Ray Data warned that the operator "can launch at most 1 task(s)" and showed 0.5 of 1 GPU in use for the whole two-actor pass; with 8 blocks, 1 of 1 once both actors were up. Both timings include actor start-up; the ratio over a longer run is unmeasured.

## Labels

| Wording | Run | positive | negative | neutral |
|---|---|---|---|---|
| `LABEL_PHRASES` | AWS config, A10G, bfloat16, 10,081 rows | 3,356 of 3,356 | 3,401 of 3,401 | 3,324 of 3,324 |
| `LABEL_PHRASES` | Laptop CPU, float32, 2,000 rows | 656 of 656 | 701 of 701 | 643 of 643 |
| Bare names in "The sentiment of this text is {}." | Laptop CPU, float32, 2,000 rows | 656 of 656 | 564 of 701 | 0 of 643 |

Leaving out `hypothesis_template` changed the top label on 681 of those 2,000 rows (34.1%) with the bare names, and on none with `LABEL_PHRASES`.

## Probe cost

The digests are 128-bit because a new row falsely matches a prior key with probability about (prior keys) / 2^bits: at 10^9 prior keys and 10^9 new rows a day, a 64-bit digest silently drops about one row every 18 days, a 128-bit digest about 3 x 10^-21 rows a day. Both figures are arithmetic.

Measured with numpy 2.2.6 and pandas 2.3.3 outside Ray, on the laptop, 2026-09-25, on a 10,000-row batch of the synthetic rows (3.3 MiB in pandas). Times are medians of 7; memory is tracemalloc's peak above where each call started.

| Prior keys (set) | `ProbeFilter` per batch | With `np.isin` in place of `np.searchsorted` |
|---|---|---|
| 10^6 (15 MiB) | 0.027 s, 2.4 MiB | 0.043 s, 57 MiB |
| 10^7 (153 MiB) | 0.037 s, 2.4 MiB | 0.34 s, 563 MiB |

The lookup itself took 0.007 s at 10^6 keys and 0.022 s at 10^7, and allocated 25 bytes per batch row at both; `np.isin` copies and sorts the set on every call, using 3.7 times its size. Over the timed calls at 10^7 keys, the process's peak RSS rose about 1 MiB, against about 630 MiB with `np.isin` (three runs). Each probe actor checks the set's order once as it starts: 0.03 s and 1 MiB at 10^7 keys. In a local Ray 2.58.0 run, both probe actors held the 10^7-key set as a read-only view of the object store's copy. Collecting and deduplicating 10^7 digests as `prior_key_hashes` does peaked at 4.1 times the set. The README's 10^8 figures scale these linearly.

## From the originating workload

Not reproduced here: 1.73x and 2.35x packing for two and four actors per A10G, HTTP 429 at a few hundred simultaneous model downloads, the hash join as the stage that stalled at production scale, and tens of minutes spent writing per-partition markers on object storage.
