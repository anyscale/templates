# Incremental batch classification with Ray Data

A daily Ray Data job that classifies only the rows it hasn't classified before, on GPUs, writes them as partitioned Parquet with one completion marker, and ends at a time bound. The notebook runs each step on synthetic data, three of them beside the usual approach; `run_pipeline.py` and `job.yaml` run the same pipeline as a job.

| Step | Usual approach | This template | What it buys |
|---|---|---|---|
| Skip rows already classified | Hash `left_anti` join against the prior output | Broadcast probe: hash each prior key to 16 bytes, broadcast the set, filter in a streaming `map_batches` | No shuffle aggregators, which hold CPU and memory on the GPU nodes and fail the join if one is lost; the classifier starts on the first filtered block |
| Load the model | Every actor downloads it from the Hugging Face Hub | Download once to shared storage, or bake it into the image | No burst of Hub requests, which the Hub rate-limits with HTTP 429 |
| Pack the GPU | One actor per GPU | Two per GPU: `gpu_fraction=0.5` | More rows/s from a GPU one actor underuses |
| Mark the output complete | `_SUCCESS` in every partition, written from the driver | One `_SUCCESS` at the output root | One request instead of one per partition |
| Bound the run | None | Job `timeout_s`, the hanging-execution detector, one driver stack dump | A stalled run ends and leaves a trace |
| Pin pyarrow | Unpinned | `pyarrow==23.0.1` | Avoids two known deadlocks: concurrent dataset writes before 22 ([apache/arrow#47124](https://github.com/apache/arrow/issues/47124)), and interpreter exit with a live `S3FileSystem` in 24.x ([apache/arrow#50188](https://github.com/apache/arrow/issues/50188)) |

Measured on the included AWS compute config (one g5.2xlarge GPU worker with an A10G, Ray 2.58.0) on 2026-09-24:

| Measure | Result |
|---|---|
| Anti-join, 20,000 rows, warm cluster | Hash join 6.0 s with 24 aggregators; probe 2.3 s with none; same 10,081 rows kept |
| GPU packing, 10,081 rows | 267 rows/s with one actor, 398 with two: 1.49x, actor start-up included |
| Labels, bfloat16 | All 10,081 rows labelled as written |
| CI test runtime | 11 min 41 s, workspace start to teardown, including about 3 minutes of untimed first join while the autoscaler adds a CPU worker |

The data is synthetic and the model public. [NOTES.md](https://github.com/anyscale/templates/blob/main/templates/ray-data-incremental-batch-classification/NOTES.md) has the per-step figures, the laptop runs and what ran where.

## Set up

This reads the demo's settings from environment variables and prints them; everything the notebook writes goes under `STORAGE_DIR`.


```python
import os
import time

NUM_ROWS = int(os.getenv("NUM_ROWS", "20000"))
PRIOR_FRACTION = float(os.getenv("PRIOR_FRACTION", "0.5"))  # share of today's keys already classified
NUM_SHARDS = int(os.getenv("NUM_SHARDS", "200"))  # partitions in the output
JOIN_PARTITIONS = int(os.getenv("JOIN_PARTITIONS", "64"))
CLASSIFY_ROWS = int(os.getenv("CLASSIFY_ROWS", "0")) or None  # cap rows sent to the model
STORAGE = os.getenv("STORAGE_DIR") or (
    "/mnt/cluster_storage/incremental-demo" if os.path.isdir("/mnt/cluster_storage") else "/tmp/incremental-demo"
)
print(f"{NUM_ROWS=} {PRIOR_FRACTION=} {NUM_SHARDS=} {JOIN_PARTITIONS=} {STORAGE=}")
```

This installs `python_depset.lock` on the driver: torch built for CUDA 12.9, transformers, huggingface-hub and pyarrow, pinned on top of the image.


```python
!uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
```

This starts Ray with the lock and `incremental.py` shipped to the workers, turns on Ray Data's hanging-execution detector and arms one driver stack dump for 45 minutes in; `HangingExecutionIssueDetector` should be in the printed list.


```python
import ray

import incremental as inc

ray.init(runtime_env={"pip": os.path.abspath("python_depset.lock"), "py_modules": [inc.__file__]})
inc.enable_hang_detection()
inc.arm_driver_stack_dump(after_s=45 * 60)
print([d.__name__ for d in ray.data.DataContext.get_current().issue_detectors_config.detectors])
```

## Skip rows already classified

This writes a synthetic `today` and `prior`, with about 2% of today's rows NULL in the key column `lang`, and prints the counts; NULL never equals NULL in a join, so both paths below must keep those rows.


```python
today_df, prior_df = inc.make_synthetic_frames(NUM_ROWS, prior_fraction=PRIOR_FRACTION, num_shards=NUM_SHARDS)
ray.data.from_pandas(today_df).write_parquet(f"{STORAGE}/today", mode=ray.data.SaveMode.OVERWRITE)
ray.data.from_pandas(prior_df).write_parquet(f"{STORAGE}/prior", mode=ray.data.SaveMode.OVERWRITE)

today = ray.data.read_parquet(f"{STORAGE}/today")
prior = ray.data.read_parquet(f"{STORAGE}/prior")
null_key_rows = int(today_df["lang"].isna().sum())
print(f"today: {len(today_df)} rows, prior: {len(prior_df)} keys, NULL-key rows today: {null_key_rows}")
today_df.head()
```

This runs the hash join and the probe once untimed, so a node the autoscaler adds isn't charged to either, then times both; look for equal row and NULL-key counts, aggregators for the join only, and no `NOT A FAIR COMPARISON` line.


```python
# Untimed warm-up, so a node the autoscaler adds isn't charged to the join.
cpus_cold = ray.cluster_resources().get("CPU", 0)
_, warmup_join_s, warmup_join_aggregators = inc.timed_materialize(inc.anti_join_hash, today, prior, num_partitions=JOIN_PARTITIONS)
_, warmup_probe_s, warmup_probe_aggregators = inc.timed_materialize(inc.anti_join_probe, today, prior)
cpus_warm = ray.cluster_resources().get("CPU", 0)
print(f"warmup, untimed: join {warmup_join_s:.1f}s, probe {warmup_probe_s:.1f}s, cluster CPUs {cpus_cold:.0f} -> {cpus_warm:.0f}")

# Timed, on the warmed cluster.
joined, t_join, join_aggregators = inc.timed_materialize(inc.anti_join_hash, today, prior, num_partitions=JOIN_PARTITIONS)
new_rows, t_probe, probe_aggregators = inc.timed_materialize(inc.anti_join_probe, today, prior)
cpus_after = ray.cluster_resources().get("CPU", 0)

join_keys, probe_keys = inc.key_set(joined), inc.key_set(new_rows)
print(f"hash left_anti : {joined.count():>8} rows  {t_join:6.1f}s  aggregator actors started: {join_aggregators}")
print(f"broadcast probe: {new_rows.count():>8} rows  {t_probe:6.1f}s  aggregator actors started: {probe_aggregators}")
print(f"NULL-key rows kept: join={sum(None in k for k in join_keys)} probe={sum(None in k for k in probe_keys)} expected={null_key_rows}")
if cpus_after != cpus_warm:
    print(f"NOT A FAIR COMPARISON: the cluster went from {cpus_warm:.0f} to {cpus_after:.0f} CPUs during the timed runs")

assert join_keys == probe_keys, "the probe and the join disagree"
assert sum(None in k for k in probe_keys) == null_key_rows, "a NULL-key row was dropped"
assert join_aggregators > 0 and warmup_join_aggregators > 0, "the join started no shuffle aggregators"
assert probe_aggregators == warmup_probe_aggregators == 0, "the probe started shuffle aggregators"
```

## Classify with fractional GPUs

This downloads [`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7), a multilingual NLI model (MIT) for zero-shot classification, to shared storage once, only the 578 MB of 2.6 GB that inference needs, and lists the files.


```python
model_dir = inc.download_model_once(inc.MODEL_ID, f"{STORAGE}/models/mdeberta")
sorted(os.listdir(model_dir))
```

This classifies the new rows with one actor per GPU, then two, on the same blocks; look for the rows/s ratio and a label table with every count on the diagonal (written label down, the model's across).


```python
to_classify = new_rows.limit(CLASSIFY_ROWS) if CLASSIFY_ROWS else new_rows
has_gpu = ray.cluster_resources().get("GPU", 0) >= 1
fractions = [1.0, 0.5] if has_gpu else [1.0]

# Split once, so every actor has blocks and both passes classify the same ones.
largest_pool = max(inc.classify_actors(fraction) for fraction in fractions)
to_classify = inc.split_for_actors(to_classify, largest_pool).materialize()
n_rows = to_classify.count()
print(f"{n_rows} rows in {to_classify.num_blocks()} blocks for up to {largest_pool} actors")

rates = {}
for fraction in fractions:
    t0 = time.perf_counter()
    classified = inc.classify(to_classify, model_dir, gpu_fraction=fraction).materialize()
    elapsed = time.perf_counter() - t0
    rates[fraction] = n_rows / elapsed
    print(f"gpu_fraction={fraction}: {inc.classify_actors(fraction)} actor(s), {n_rows} rows in {elapsed:.1f}s -> {rates[fraction]:.0f} rows/s")

if len(rates) == 2:
    print(f"two actors per GPU vs one: {rates[0.5] / rates[1.0]:.2f}x")
out = classified.to_pandas()
print(inc.label_confusion(out))
out[["company", "lang", "text", "label", "score"]].head()
```

## Write partitioned output with one marker

This writes the classified rows to `out-per-partition/`, with a `_SUCCESS` in every partition, and to `out/`, with one at the root; look for 200 markers against 1, and each marker time.


```python
per_partition = inc.write_partitioned(classified, f"{STORAGE}/out-per-partition", "shard", marker="per_partition")
root_only = inc.write_partitioned(classified, f"{STORAGE}/out", "shard", marker="root")
partitions = classified.to_pandas()["shard"].nunique()

print(f"per-partition markers: {inc.count_markers(f'{STORAGE}/out-per-partition'):>5}  marker time {per_partition['marker_s']}s")
print(f"root marker          : {inc.count_markers(f'{STORAGE}/out'):>5}  marker time {root_only['marker_s']}s")
assert inc.count_markers(f"{STORAGE}/out-per-partition") == partitions
assert inc.count_markers(f"{STORAGE}/out") == 1
```

## Run it as a job on your data

`run_pipeline.py` is this pipeline without the comparisons, `anyscale job submit --config-file job.yaml` submits it, and the cell prints `job.yaml`.


```python
print(open("job.yaml").read())
```

To run it on your data, set `TODAY_PATH`, `PRIOR_PATH` and `OUTPUT_PATH` under `env_vars`. Today's rows need a `text` column, the key columns in `KEY_COLUMNS` (`doc_id`, `company`, `lang`) and a `shard` column to partition by, and the prior output needs the keys; edit `KEY_COLUMNS` in `incremental.py` and the partition column in `run_pipeline.py` to match your table. The script only reads `PRIOR_PATH`: adding each run's keys to it is up to you.

Two overwrites can destroy data:

- Every run deletes everything under `OUTPUT_PATH` before it writes.
- Unless `os.path.isdir` finds both `TODAY_PATH` and `PRIOR_PATH`, the script writes synthetic data over both. That includes a first run with no prior output yet, and every `s3://` or `gs://` path, which `os.path.isdir` can't see. Remove that block from `run_pipeline.py` before a real run.

In `job.yaml`:

- `timeout_s` is the job's only bound: job clusters don't idle-terminate, and Ray Data's no-progress timeout is off for any plan with an all-to-all operation, such as the script's `repartition`. Set it about 30% above your longest healthy run; 3600 is a placeholder, since the job hasn't run on a cluster.
- `max_retries` reruns the whole job and `timeout_s` applies per attempt, so the worst case is `timeout_s * (max_retries + 1)`. Use 0 until the pipeline is stable.
- The compute config is the one CI runs the notebook on. On GCP, use n2-standard-8 for the head and CPU workers and g2-standard-8-nvidia-l4-1 for the GPU worker.

Bake the model into your image before you grow the actor pool.

## What to watch

- **Prior-key count.** The probe's memory grows with the prior output. Its set is 16 bytes per prior key, and the driver needs about 4 times that to build and sort it. Each node running a probe actor then holds the sorted set once, shared by its actors, and each batch adds about its own size: against 10^7 prior keys, a 10,000-row batch of 3.3 MiB took 0.04 s and 2.4 MiB (measured with numpy on a laptop). An `np.isin` lookup would copy and sort the whole set for every batch instead: 0.34 s and 563 MiB on the same batch. At 10^8 prior keys the set is 1.6 GB per node and the driver needs about 6.5 GB.
- **Blocks.** If Ray Data warns that the classifier "can launch at most 1 task(s)", give it more blocks: with all its rows in one block, one of two actors did all the work, for 1.10x (first CI run).
- **Labels.** The model reads each label inside the hypothesis template, so word the labels for it, pass `hypothesis_template` explicitly, and check both against rows you know. With the bare names in "The sentiment of this text is {}.", 0 of 643 neutral rows came out `neutral` (laptop, CPU); `LABEL_PHRASES` was chosen against this template's synthetic rows.
- **Small-run timings.** At 20,000 rows the join and probe seconds are mostly noise; what grows with scale is the join's aggregator pool, one actor per partition up to the cluster's maximum CPU count.
