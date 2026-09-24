# Incremental batch classification with Ray Data

A daily batch job that classifies only the rows it has not classified before, writes the results as partitioned Parquet with a completion marker, and cannot run forever.

Each step compares the version most first drafts reach for with the one this template recommends, on synthetic data, so you can see the difference on your own cluster:

| Step | Common first version | This template | Why |
|---|---|---|---|
| Skip rows already classified | `Dataset.join(join_type="left_anti")` against the prior output | Broadcast 16-byte key hashes and filter in a streaming `map_batches` | The hash join shuffles both sides through stateful `HashShuffleAggregator` actors: losing one fails the dataset, and they keep GPU nodes busy while using no GPU. The probe has no shuffle and streams straight into the GPU stage. |
| Model weights | Each actor downloads from the Hugging Face Hub when it starts | Download once to shared storage, or bake into the image | In the source engagement, a few hundred simultaneous downloads of one file drew HTTP 429 and the actors died in their constructors. |
| GPU packing | One actor per GPU | `num_gpus=0.5`: two actors per GPU | Measured here, on the included AWS config's one A10G, with the input split so both actors had work: 398 rows/s with two actors against 267 with one, 1.49x. The source engagement's 1.73x for two actors per GPU and 2.35x for four are that workload's figures, on its own data, and are not reproduced here. |
| Completion marker | `_SUCCESS` in every partition directory, written from the driver | One `_SUCCESS` at the output root | Thousands of partitions means thousands of sequential requests after the data is already written. In the source engagement that was tens of minutes on object storage, and throttling on the prefix. |
| Failure bounds | None | Job `timeout_s`, Ray Data's hanging-execution detector | A stalled Ray Data job raises nothing. Its no-progress guard turns itself off when the plan contains a shuffle, and job clusters do not idle-terminate. |
| pyarrow | Whatever the lockfile resolves | `pyarrow==23.0.1`, inside the 22–23 window | 21 and older deadlock on concurrent partitioned writes under S3 throttling ([apache/arrow#47124](https://github.com/apache/arrow/issues/47124), fixed in 22). 24.x can deadlock at interpreter exit with a live `S3FileSystem` ([apache/arrow#50188](https://github.com/apache/arrow/issues/50188)). |

The failure modes in this table, and every ratio credited to the source engagement, come from a customer engagement whose data is private, on Ray 2.55.1 in September 2026. None is reproduced here unless a cell below measures it. Treat the ratios as directions: they do not transfer as absolutes. **Provenance**, at the end, says what this notebook has measured and where.

**Runtime:** 11 min 41 s for the CI test, workspace start to teardown, on the included AWS compute config (one g5.2xlarge GPU worker), 2026-09-24. The untimed first join takes about 3 of those minutes, while the autoscaler adds a CPU worker.

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

**Time them on a warm cluster.** Each path runs once, untimed, before the timed runs. On the included AWS config the first join is what made the autoscaler add a CPU worker, and a new node installs the lock through `runtime_env` before it runs anything; timing that first run charges the node launch to the join, and hands the probe that runs after it a bigger cluster. The warmup line prints both untimed runs and the cluster's CPU count before and after. If the cluster changes size during the timed runs anyway, the cell says so.


```python
# Warm up: run each path once, untimed. The first join can make an autoscaling cluster add
# a CPU worker, and a new node installs the lock through runtime_env before it runs
# anything. Timing that run charges the node launch to the join.
cpus_cold = ray.cluster_resources().get("CPU", 0)
_, warmup_join_s, warmup_join_aggregators = inc.timed_materialize(inc.anti_join_hash, today, prior, num_partitions=JOIN_PARTITIONS)
_, warmup_probe_s, warmup_probe_aggregators = inc.timed_materialize(inc.anti_join_probe, today, prior)
cpus_warm = ray.cluster_resources().get("CPU", 0)
print(f"warmup, untimed: join {warmup_join_s:.1f}s, probe {warmup_probe_s:.1f}s, cluster CPUs {cpus_cold:.0f} -> {cpus_warm:.0f}")

# The comparison: the same two paths again, on the warmed cluster.
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

On a 14-CPU macOS laptop with Ray 2.58.0 at the default 20,000 rows, on 2026-09-24, after the warmup: the join started 14 aggregator actors and took 3.8 s; the probe started none and took 2.6 s. Both returned the same 10,081 rows and kept all 371 NULL-key rows. The untimed warmup runs took 4.4 s and 1.9 s, and with no autoscaler the laptop stayed at 14 CPUs. An earlier version of this cell, with no warmup and one run of each, measured 5.8 s against 1.9 s on the same laptop. At this size the timings are noise-sensitive: the probe's timed run was slower than its warmup. The structural difference is the one that grows: the join's aggregator count scales with `num_partitions` and each aggregator reserves CPU, while the probe adds one filter to a stream that was already running.

On the included AWS config, on 2026-09-24, the version without the warmup printed 187.0 s for the join, with 24 aggregators, and 4.0 s for the probe, with none. Those two numbers are not a comparison. The cluster started with 8 CPUs, all on the GPU worker, because the head has `CPU: 0`. During the join Ray Data reported 24.0 GiB of memory active and requested against the cluster's 18.9 GiB, the autoscaler launched an m5.2xlarge CPU worker, and the cluster reached 16 CPUs about a minute into the join. The join finished almost two minutes after that, most likely while the new node installed the lock through `runtime_env`, and the probe then ran on the larger, warm cluster. The warmup takes that cost out of the comparison.

With the warmup, on the same config later that day: the untimed runs took 189.6 s for the join and 4.0 s for the probe, while the cluster went from 8 to 16 CPUs. On the warm cluster the join then took 6.0 s and started 24 aggregators; the probe took 2.3 s and started none. Both returned the same 10,081 rows and kept all 371 NULL-key rows, and the cluster did not change size during the timed runs.

## Download the model once

The classifier is [`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) (MIT licence), a multilingual NLI model used for zero-shot classification. Download it from the driver to shared storage, once, and point every actor at the local copy. Only the files inference needs are fetched: about 578 MB instead of the repository's several GB.

Letting every actor download on start-up works on one GPU. In the source engagement it failed at a few hundred: the hub rate-limited the burst with HTTP 429 and the actors died before their first batch. That is not reproduced here. For production, bake the weights into the image.


```python
model_dir = inc.download_model_once(inc.MODEL_ID, f"{STORAGE}/models/mdeberta")
sorted(os.listdir(model_dir))
```

## Classify with fractional GPUs

Each actor loads the model once and classifies batches of text against three labels. Three details matter:

- **Word the labels for the model, then check them against rows you know.** The model reads each label inside the hypothesis template. `LABEL_PHRASES` in `incremental.py` has it score "This text is good news.", "This text is bad news." and "This text is routine news.", and the `label` column keeps the short names `positive`, `negative` and `neutral`. The first wording, the bare names inside "The sentiment of this text is {}.", never labelled a neutral sentence `neutral`; the counts are after the cell. In a sweep of 14 templates over 180 of the sentences (laptop, Apple GPU, float32), no label set that used the bare word `neutral` got more than 30% of the neutral sentences.
- **Pass `hypothesis_template` explicitly.** The pipeline's default is `"This example is {}."`. With the first wording, leaving it out changed the winning label on 681 of 2,000 of this notebook's synthetic rows (34.1%); with the current wording it changed none. Which template works depends on the labels, so choose the two together. Both measured on a laptop CPU in float32 on 2026-09-24.
- **Load in bfloat16 on GPU, not float16.** The [model card](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) says mDeBERTa does not support FP16. This notebook did not try float16.

The comparison runs the same rows with one actor per GPU and then with two (`gpu_fraction=0.5`). Every actor needs blocks to work on. Ray Data gives an actor one block per task, and an actor that is ready first takes up to two tasks without waiting for the rest of the pool, so input with one block per actor can leave the later actors idle, and input with one block always does. The cell splits the rows once, into `BLOCKS_PER_ACTOR` (4) blocks per actor of the larger pool, and both passes classify those same blocks. The actor count is sized to the GPUs in the cluster, never above: a pool larger than the GPU count leaves the extra actors pending for the whole run. On a cluster with no GPU the cell runs once on CPU actors.


```python
to_classify = new_rows.limit(CLASSIFY_ROWS) if CLASSIFY_ROWS else new_rows
has_gpu = ray.cluster_resources().get("GPU", 0) >= 1
fractions = [1.0, 0.5] if has_gpu else [1.0]

# Split the rows once, into BLOCKS_PER_ACTOR blocks per actor of the largest pool below, so
# every actor has blocks to work on and both passes classify the same blocks.
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

The ratio is printed, not asserted. Both timings include actor start-up, which is a large share of a short run.

On the included AWS config, one A10G on a g5.2xlarge, on 2026-09-24, before the cell split its input: 260 rows/s with one actor per GPU and 287 with `gpu_fraction=0.5`, 1.10x, over 10,081 rows. That is not a packing ratio. `new_rows` reached the classifier as one block, so Ray Data could run one task at a time: it warned that the operator "can launch at most 1 task(s)", and it reported 0.5 of 1 GPU in use at every progress line of the second pass. One actor did all the work in both passes.

With the split, on the same config later that day: 10,081 rows in 8 blocks for up to 2 actors; 267 rows/s with one actor and 398 with two, 1.49x. Once its actors had started, Ray Data reported 1 of 1 GPU in use for the two-actor pass, and it printed no "can launch at most" warning for the classifier. Both timings include actor start-up; whether a longer run gives a different ratio is unmeasured.

On the laptop, CPU only, after the split, on 2026-09-24: the cell printed 2,000 rows in 8 blocks and no "can launch at most" warning for the classifier. A separate check that tagged every classified row with the process that classified it, over 800 of the same rows, found one actor classifying all 800 in the one-actor pass and two actors classifying 400 each in the two-actor pass. The cell's timings, 36 and 63 rows/s (1.74x), are two CPU actors against one on a 14-CPU laptop: not a GPU packing ratio, and not comparable with the figures below.

The source engagement measured 1.73x for two actors per A10G and 2.35x for four. Those are the originating workload's figures, on its own data; this notebook has not reproduced them.

The labels are not this template's lesson, but read them before you trust them. The table the cell prints counts rows by the label each synthetic sentence was written to carry (rows) against the label the model gave it (columns). Measured on a laptop CPU in float32 over 2,000 rows on 2026-09-24, with the current wording: 656 of 656 positive sentences came out `positive`, 701 of 701 negative ones `negative` and 643 of 643 neutral ones `neutral`. With the first wording, on the same rows: 656 of 656 `positive`; 564 of 701 `negative` and the other 137 `neutral`, every one of them the Spanish sentence; 642 of 643 neutral ones `negative` and 1 `positive`.

The current wording was chosen by trying wordings against these same rows, so a clean table here says it fits this data, not yours. On the included config's A10G, in bfloat16, over all 10,081 new rows on 2026-09-24, the current wording labelled every row as written too: 3,356 of 3,356 positive, 3,401 of 3,401 negative and 3,324 of 3,324 neutral. The first cluster run used the first wording, and the two neutral sentences among the five rows it printed both came out `negative`. Choose the labels and the hypothesis template against your own data.

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

`run_pipeline.py` splits today's rows into blocks for the classifier's actors before the probe, not after it: `repartition(num_blocks)` waits for its whole input, and between the probe and the classifier it would hold the GPU stage until the probe finished.

Three settings in `job.yaml` are the difference between a failed run and a run that holds a cluster until someone notices:

- **`timeout_s`** is the only bound a job has. Set it about 30% above your longest healthy run. It applies per attempt.
- **`max_retries`** re-runs the whole job, so the worst case is `timeout_s * (max_retries + 1)`. Use 0 while a pipeline is still unstable.
- **pyarrow** stays pinned in `requirements.txt` so a rebuilt image cannot float into a version with a known write or exit deadlock. Check the version the job actually ran with, not the one the lockfile names: a mutable image tag can serve a different build.

`job.yaml` also carries its own compute config, the cluster this notebook runs on in CI: a head with `CPU: 0`, one g5.2xlarge GPU worker and up to two m5.2xlarge CPU workers. Without one, a job lands on the cloud's default compute config. The instance types are AWS's; on GCP use n2-standard-8 for the head and the CPU workers and g2-standard-8-nvidia-l4-1 for the GPU worker.


```python
print(open("job.yaml").read())
```

## Summary

- The broadcast probe returned the same rows as the hash `left_anti` join, NULL-key rows included, without starting any shuffle aggregators.
- The model downloads once, from the driver, and every actor loads that copy.
- The classify cell splits its input so every actor has blocks, then prints what two actors per GPU buy on your GPU: 1.49x on the included config's A10G.
- One root `_SUCCESS` replaced one marker per partition.
- `job.yaml` bounds the job with `timeout_s`; the notebook runs with the hanging-execution detector on and a one-shot driver stack dump armed.

Next steps: point `TODAY_PATH`, `PRIOR_PATH` and `OUTPUT_PATH` at your own data, set `timeout_s` from your own run times, and bake the model into your image before scaling the actor pool.

## Provenance

This template comes from a customer engagement whose data is private. The pipeline shape and the levers are the engagement's. The data is synthetic and the model is public: [`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7), MIT, ungated, checked on 2026-09-24.

**Measured on a 14-CPU macOS laptop, Ray 2.58.0, torch 2.13.0 on CPU, transformers 5.17.0, 2026-09-24.** This notebook end to end through papermill, three times, 23 of 23 cells each time, with two substitutions: the lock install was skipped and `ray.init` carried no `pip`, because the lock holds Linux CUDA wheels. `CLASSIFY_ROWS=2000` capped the classifier. The first run, 3 min 17 s, before the join warmup and the label rewording, gave the unwarmed 5.8 s and 1.9 s, 200 per-partition markers against 1 root marker, the first wording's label counts and its hypothesis-template count. The second, 2 min 58 s, gave the warm anti-join numbers and the current wording's label counts. The two runs' classify cells printed 30 and 33 rows/s (1.11x), then 30 and 34 (1.13x): **not** GPU packing ratios. On Apple Silicon Ray reports one GPU that torch cannot use, so both passes ran on CPU, in float32, and the input was one block, so one actor had work in each pass. A third run, 2 min 9 s, after the classify cell began splitting its input, gave the CPU timings and block count in the classify section and the same label counts. `run_pipeline.py` ran once, at `NUM_ROWS=2000` with the same `pip` substitution: its two classifier actors classified 518 and 492 of the 1,010 new rows, and Ray Data printed no "can launch at most" warning for the probe or the classifier. The same 2,000 rows classified with `ZeroShotClassifier` outside Ray, on CPU in float32, reproduced both wordings' label counts exactly and gave the current wording's hypothesis-template count.

**Measured on the included AWS compute config, 2026-09-24.** Two CI runs: head m5.2xlarge, one g5.2xlarge (A10G) GPU worker, Ray 2.58.0; 23 of 23 cells each. The first, 11 min 50 s workspace start to teardown, showed the lock reaching the workers through `runtime_env`, the actors loading the model from shared storage, and the classifier running on the GPU in bfloat16. It came before the join warmup, the label rewording and the classify split: its classify cell printed 260 and 287 rows/s, 1.10x, with one actor working in both passes, and the join section says why its anti-join timings are not a comparison. The second, 11 min 41 s, ran this notebook's code and gave the warm anti-join numbers, the 1.49x packing ratio and the bfloat16 label counts above.

**From the source engagement, on a different cluster and a different dataset.** Not reproduced here: the 1.73x (two actors per GPU) and 2.35x (four) packing ratios on A10G; HTTP 429 at a few hundred simultaneous model downloads; the hash join as the stage that stalled at production scale; tens of minutes of per-partition markers on object storage.

**Unmeasured.** Whether the probe survives losing a worker: a single-node join starts no aggregators to kill, so this is untested here and was untested in the source engagement. `job.yaml` has not been submitted, and `run_pipeline.py` has not run on a cluster.
