# Incremental batch classification with Ray Data

A daily batch job that classifies only the rows it has not classified before, writes the results as partitioned Parquet with a completion marker, and cannot run forever.

Each step compares the version most first drafts reach for with the one this template recommends, on synthetic data, so you can see the difference on your own cluster:

| Step | Common first version | This template | Why |
|---|---|---|---|
| Skip rows already classified | `Dataset.join(join_type="left_anti")` against the prior output | Broadcast 16-byte key hashes and filter in a streaming `map_batches` | The hash join shuffles both sides through stateful `HashShuffleAggregator` actors: losing one fails the dataset, and they keep GPU nodes busy while using no GPU. The probe has no shuffle and streams straight into the GPU stage. |
| Model weights | Each actor downloads from the Hugging Face Hub when it starts | Download once to shared storage, or bake into the image | In the source engagement, a few hundred simultaneous downloads of one file drew HTTP 429 and the actors died in their constructors. |
| GPU packing | One actor per GPU | `num_gpus=0.5`: two actors per GPU | A base-size model leaves most of an A10G or L4 idle. In the source engagement, on A10G GPUs, two actors per GPU measured 1.73x the rows/s of one and four measured 2.35x. The classify cell below measures this notebook's own ratio. |
| Completion marker | `_SUCCESS` in every partition directory, written from the driver | One `_SUCCESS` at the output root | Thousands of partitions means thousands of sequential requests after the data is already written. In the source engagement that was tens of minutes on object storage, and throttling on the prefix. |
| Failure bounds | None | Job `timeout_s`, Ray Data's hanging-execution detector | A stalled Ray Data job raises nothing. Its no-progress guard turns itself off when the plan contains a shuffle, and job clusters do not idle-terminate. |
| pyarrow | Whatever the lockfile resolves | `pyarrow==23.0.1`, inside the 22–23 window | 21 and older deadlock on concurrent partitioned writes under S3 throttling ([apache/arrow#47124](https://github.com/apache/arrow/issues/47124), fixed in 22). 24.x can deadlock at interpreter exit with a live `S3FileSystem` ([apache/arrow#50188](https://github.com/apache/arrow/issues/50188)). |

The ratios and failure modes in this table come from a customer engagement whose data is private, on Ray 2.55.1 in September 2026. None is reproduced here unless a cell below measures it. Treat the ratios as directions: they do not transfer as absolutes. **Provenance**, at the end, says what this notebook has measured and where.

**Runtime:** estimated at 15 minutes at the default scale on the included compute config (one GPU worker); not yet measured on a cluster.

## Configure

Every knob reads from an environment variable, so the same notebook runs as a quick demo or at a scale where the differences are large. `NUM_ROWS` is the number of rows in today's batch; `PRIOR_FRACTION` of them were already classified by an earlier run.


```python
import os
import time

NUM_ROWS = int(os.getenv("NUM_ROWS", "20000"))
PRIOR_FRACTION = float(os.getenv("PRIOR_FRACTION", "0.5"))
NUM_SHARDS = int(os.getenv("NUM_SHARDS", "200"))  # partitions in the output
JOIN_PARTITIONS = int(os.getenv("JOIN_PARTITIONS", "64"))
CLASSIFY_ROWS = int(os.getenv("CLASSIFY_ROWS", "0")) or None  # cap rows sent to the model
STORAGE = os.getenv("STORAGE_DIR") or (
    "/mnt/cluster_storage/incremental-demo" if os.path.isdir("/mnt/cluster_storage") else "/tmp/incremental-demo"
)
print(f"{NUM_ROWS=} {PRIOR_FRACTION=} {NUM_SHARDS=} {JOIN_PARTITIONS=} {STORAGE=}")
```

## Install the dependencies

`python_depset.lock` pins everything this template adds to the image: torch built for CUDA 12.9, transformers and huggingface-hub, and pyarrow held at 23.0.1. This cell installs it on the driver only. The workers get the same file from `ray.init` below, through `runtime_env`.


```python
!uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
```

## Start Ray with the guardrails on

`incremental.py` holds every function this notebook uses. Shipping it with `py_modules` makes it importable on every worker, and `pip` installs the lock there. It goes as a file path, not as the module object: when this code runs as a job, Ray deep-copies the runtime env into the job's, and a module object cannot be copied.

Two guardrails go on before any work runs:

- **Ray Data's hanging-execution detector** ships commented out of the default detector list. Enabled, it logs a warning naming the operator and the stuck task when one task runs far longer than its peers. It only warns.
- **A one-shot stack dump of the driver.** On clusters where nothing can be attached with a debugger, an in-process dump is the only record of where a silent driver was blocked. Use `repeat=False`: a repeating dump reads Python frames without holding the GIL and can crash the driver it is meant to observe.


```python
import ray

import incremental as inc

ray.init(runtime_env={"pip": os.path.abspath("python_depset.lock"), "py_modules": [inc.__file__]})
inc.enable_hang_detection()
inc.arm_driver_stack_dump(after_s=45 * 60)
print([d.__name__ for d in ray.data.DataContext.get_current().issue_detectors_config.detectors])
```

## Generate today's batch and the prior output

`today` has one row per `(doc_id, company, lang)` with the text to classify and a `shard` column that becomes the output partition. `prior` holds the keys an earlier run already classified. A small share of rows have a NULL `lang`; `prior` carries NULL-`lang` twins of them. NULL never equals NULL in a join, so every one of those rows must be classified again.


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

## Skip what was already classified: hash join vs broadcast probe

**The hash join** is correct and simple. It also hash-shuffles both sides through `HashShuffleAggregator` actors sized by `num_partitions`. Those actors hold the shuffle state in memory: one lost actor fails the whole dataset, and while they wait they keep their nodes busy, so GPU nodes they land on cannot be reclaimed. In the source engagement, at production scale, this was the stage that stalled.

**The broadcast probe** hashes the prior keys on the workers into 16-byte digests, broadcasts the sorted set, and filters today's rows in a streaming `map_batches`. No shuffle, no aggregators, and the GPU stage downstream starts on the first filtered block. It costs about 16 bytes per prior key in each filter actor. 128-bit hashes, not 64. A new row is wrongly matched with probability of about (prior keys) / 2^bits, so at 10^9 prior keys and 10^9 new rows a day a 64-bit hash expects one silently dropped row every 18 days, and a 128-bit hash about 3 × 10^-21 per day. That is arithmetic, not a measurement.

Both must return exactly the same rows, NULL-key rows included.


```python
aggregators_before = inc.count_actors("HashShuffleAggregator")

t0 = time.perf_counter()
joined = inc.anti_join_hash(today, prior, num_partitions=JOIN_PARTITIONS).materialize()
t_join = time.perf_counter() - t0
aggregators_after_join = inc.count_actors("HashShuffleAggregator")

t0 = time.perf_counter()
new_rows = inc.anti_join_probe(today, prior).materialize()
t_probe = time.perf_counter() - t0
aggregators_after_probe = inc.count_actors("HashShuffleAggregator")

join_keys, probe_keys = inc.key_set(joined), inc.key_set(new_rows)
print(f"hash left_anti : {joined.count():>8} rows  {t_join:6.1f}s  aggregator actors started: {aggregators_after_join - aggregators_before}")
print(f"broadcast probe: {new_rows.count():>8} rows  {t_probe:6.1f}s  aggregator actors started: {aggregators_after_probe - aggregators_after_join}")
print(f"NULL-key rows kept: join={sum(None in k for k in join_keys)} probe={sum(None in k for k in probe_keys)} expected={null_key_rows}")

assert join_keys == probe_keys, "the probe and the join disagree"
assert sum(None in k for k in probe_keys) == null_key_rows, "a NULL-key row was dropped"
assert aggregators_after_probe == aggregators_after_join, "the probe started shuffle aggregators"
```

On a 4-CPU macOS laptop with Ray 2.58.0 at the default 20,000 rows, on 2026-09-23, the join started 14 aggregator actors and took 7.2 s; the probe started none and took 2.5 s. At this size the timings are noise-sensitive. The structural difference is the one that grows: the join's aggregator count scales with `num_partitions` and each aggregator reserves CPU, while the probe adds one filter to a stream that was already running.

## Download the model once

The classifier is [`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) (MIT licence), a multilingual NLI model used for zero-shot classification. Download it from the driver to shared storage, once, and point every actor at the local copy. Only the files inference needs are fetched: about 578 MB instead of the repository's several GB.

Letting every actor download on start-up works on one GPU. In the source engagement it failed at a few hundred: the hub rate-limited the burst with HTTP 429 and the actors died before their first batch. That is not reproduced here. For production, bake the weights into the image.


```python
model_dir = inc.download_model_once(inc.MODEL_ID, f"{STORAGE}/models/mdeberta")
sorted(os.listdir(model_dir))
```

## Classify with fractional GPUs

Each actor loads the model once and classifies batches of text against three labels. Two details matter:

- **Pass `hypothesis_template` explicitly.** Without it the pipeline wraps each label in `"This example is {}."`, which in the source engagement changed the winning label on a noticeable share of rows. Not measured here.
- **Load in bfloat16 on GPU, not float16.** The [model card](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) says mDeBERTa does not support FP16. This notebook did not try float16.

The comparison runs the same rows with one actor per GPU and then with two (`gpu_fraction=0.5`). The actor count is sized to the GPUs in the cluster, never above: a pool larger than the GPU count leaves the extra actors pending for the whole run. On a cluster with no GPU the cell runs once on CPU actors.


```python
to_classify = new_rows.limit(CLASSIFY_ROWS) if CLASSIFY_ROWS else new_rows
n_rows = to_classify.count()
has_gpu = ray.cluster_resources().get("GPU", 0) >= 1

rates = {}
for fraction in ([1.0, 0.5] if has_gpu else [1.0]):
    t0 = time.perf_counter()
    classified = inc.classify(to_classify, model_dir, gpu_fraction=fraction).materialize()
    elapsed = time.perf_counter() - t0
    rates[fraction] = n_rows / elapsed
    print(f"gpu_fraction={fraction}: {n_rows} rows in {elapsed:.1f}s -> {rates[fraction]:.0f} rows/s")

if len(rates) == 2:
    print(f"two actors per GPU vs one: {rates[0.5] / rates[1.0]:.2f}x")
classified.to_pandas()[["company", "lang", "text", "label", "score"]].head()
```

The ratio is printed, not asserted. Both timings include actor start-up, which is a large share of a short run, so at the default scale the gain looks smaller than it is. Raise `NUM_ROWS` to see it grow.

## Write partitioned output with one completion marker

Downstream jobs usually wait for a `_SUCCESS` object before reading. Writing one per partition, from the driver, one request at a time, is a habit carried over from Spark. In the source engagement, with thousands of partitions on object storage, it added minutes of wall clock after the data itself was written, and the request burst on a single prefix drew throttling. Writing one marker at the root after the write returns carries the same signal.

`write_parquet` blocks until every file is written, so a marker written after it returns never announces a partial output.


```python
per_partition = inc.write_partitioned(classified, f"{STORAGE}/out-per-partition", "shard", marker="per_partition")
root_only = inc.write_partitioned(classified, f"{STORAGE}/out", "shard", marker="root")
partitions = classified.to_pandas()["shard"].nunique()

print(f"per-partition markers: {inc.count_markers(f'{STORAGE}/out-per-partition'):>5}  marker time {per_partition['marker_s']}s")
print(f"root marker          : {inc.count_markers(f'{STORAGE}/out'):>5}  marker time {root_only['marker_s']}s")
assert inc.count_markers(f"{STORAGE}/out-per-partition") == partitions
assert inc.count_markers(f"{STORAGE}/out") == 1
```

## Run it as a job, with a bound

`run_pipeline.py` is this notebook's pipeline without the comparisons, and `job.yaml` submits it:

```bash
anyscale job submit --config-file job.yaml
```

Three settings in `job.yaml` are the difference between a failed run and a run that holds a cluster until someone notices:

- **`timeout_s`** is the only bound a job has. Set it about 30% above your longest healthy run. It applies per attempt.
- **`max_retries`** re-runs the whole job, so the worst case is `timeout_s * (max_retries + 1)`. Use 0 while a pipeline is still unstable.
- **pyarrow** stays pinned in `requirements.txt` so a rebuilt image cannot float into a version with a known write or exit deadlock. Check the version the job actually ran with, not the one the lockfile names: a mutable image tag can serve a different build.


```python
print(open("job.yaml").read())
```

## Summary

- The broadcast probe returned the same rows as the hash `left_anti` join, NULL-key rows included, without starting any shuffle aggregators.
- The model downloaded once, and every actor loaded it from shared storage.
- Packing two actors per GPU raised throughput on the same hardware.
- One root `_SUCCESS` replaced one marker per partition.
- The job runs under a timeout, with the hanging-execution detector on and a one-shot driver stack dump armed.

Next steps: point `TODAY_PATH`, `PRIOR_PATH` and `OUTPUT_PATH` at your own data, set `timeout_s` from your own run times, and bake the model into your image before scaling the actor pool.

## Provenance

This template comes from a customer engagement whose data is private. The pipeline shape and the levers are the engagement's. The data is synthetic and the model is public: [`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7), MIT, ungated, checked on 2026-09-24.

**Measured on a 4-CPU macOS laptop, Ray 2.58.0.** On 2026-09-23, every cell but the model download and the classify cell, with the classifier stubbed out: the numbers in the text after the anti-join cell. On 2026-09-24, the anti-join logic again at 5,000 and 20,000 rows, the same rows from both paths and every NULL-key row kept, and the classifier on its own on CPU with torch 2.13.0 and transformers 5.17.0: 30 rows labelled, no NaN scores.

**From the source engagement, on a different cluster and a different dataset.** Not reproduced here: the 1.73x and 2.35x packing ratios on A10G; HTTP 429 at a few hundred simultaneous model downloads; the hash join as the stage that stalled at production scale; tens of minutes of per-partition markers on object storage; the hypothesis template changing labels.

**Unmeasured.** Whether the probe survives losing a worker: a single-node join starts no aggregators to kill, so this is untested here and was untested in the source engagement. `run_pipeline.py` and `job.yaml` have not been submitted.
