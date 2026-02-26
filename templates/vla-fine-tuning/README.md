## Distributed Vision-Language-Action (VLA) Model Fine-tuning with Ray


![VLA Fine-tuning Pipeline](images/vla-fine-tuning-pipeline.png)

This notebook implements an end-to-end distributed fine-tuning pipeline for the **PI0.5 Vision-Language-Action (VLA)** model on the **Droid v1.0.1** robot manipulation dataset

It leverages Ray's ability to independently scale two distinct compute tiers.
- **CPU nodes** — Orchestrated by Ray Data, this tier handles all data-intensive work:
    - streaming HDF5 robot state and action files,
    - reading MP4 camera videos directly from Google Cloud Storage,
    - decoding video frames,
    - assembling per-timestep rows, and
    - running batched normalization preprocessing to produce camera images, robot state vectors, and action tensors
- **GPU nodes** — Managed by Ray Train, these are dedicated exclusively to the PI0.5 forward and backward passes and DDP gradient synchronization across A100 accelerators.

Because the CPU preprocessing pool and the GPU training pool are independently scaled and communicate asynchronously through the Ray object store, adding more CPU nodes accelerates distributed video decoding and data loading without touching the GPU fleet, and adding more GPU nodes increases training throughput without requiring additional preprocessing capacity. Neither side sits idle, and GPU stalls waiting for data are eliminated.

---

- Run this entire tutorial on [Anyscale](https://anyscale.com) for free: https://console.anyscale.com/template-preview/vla-fine-tuning

- This tutorial focuses on the **systems challenges** of distributed VLA fine-tuning at scale - data loading, preprocessing, scaling across nodes, orchestration and moving from development to production.

- We treat the model itself as a **black box**. The goal is not to explore the model architecture or learning algorithm, but to show how to reliably and efficiently train large multimodal models in distributed environments.

- The tutorial uses [PI0.5](https://www.physicalintelligence.company/blog/pi05) — Physical Intelligence's vision-language-action model — and the [Droid v1.0.1 raw dataset](https://droid-dataset.github.io/).


## 0. Setup

<span style="background-color: #fff3b0; padding: 2px 4px; border-radius: 3px;">
<strong>IMPORTANT: BEFORE YOU START - DEPENDENCIES WITH UV</strong>
</span>

1. Open a terminal and run <code>uv sync</code>  
2. After this completes, if prompted for a kernel, select the existing Python environment named vla (.venv/bin/python)

<span style="background-color: #fff3b0; padding: 2px 4px; border-radius: 3px;">
<strong>IMPORTANT: BEFORE YOU START - GOOGLE CLOUD AUTHENTICATION</strong>
</span>

Run the following command in a terminal window. Note: this will require google cloud authentication and then stores user google cloud credentials in a shared location for access by all workers

```
gcloud auth application-default login && \
install -m 600 ~/.config/gcloud/application_default_credentials.json \
/mnt/cluster_storage/gcp_adc.json
```

<span style="background-color: #fff3b0; padding: 2px 4px; border-radius: 3px;">
<strong>IMPORTANT: BEFORE YOU START - HUGGINGFACE TOKEN</strong>
</span>

PI0.5 depends on [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) as a backbone. Although the model weights are publicly available, Google requires you to **accept the model license** before they can be downloaded.

To do this:
1. Log in to [huggingface.co](https://huggingface.co) and navigate to the [google/paligemma-3b-pt-224 model page](https://huggingface.co/google/paligemma-3b-pt-224).
2. Accept the license agreement on that page.
3. Generate a HuggingFace access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and set it in the cell below.

Without completing both steps, the model download will fail with a 401/403 error even if a valid token is provided.

#### Init Ray

First we use a custom utility function to initialize Ray so that workers can access:
1. A HuggingFace token (for access to the pi0.5 VLA and the paligemma VLM backbone)
2. Google Cloud credentials (for reading the Droid dataset)


```python
# Start a Ray cluster session configured for this tutorial.

# Reads GCS credentials from the shared path written by the setup step,
# then calls ray.init() with a runtime environment that:
#     - uses uv as the Python executable on every worker
#     - bundles the current working directory (so vla/ is importable)
#     - propagates HF_TOKEN and GCP credentials to all remote workers

# Requires the HF_TOKEN environment variable to be set before calling this
# function.  Export it in your shell or set it in the notebook kernel:

#     export HF_TOKEN=hf_...

import json
import os
import ray

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise EnvironmentError(
        "HF_TOKEN is not set. Please export it before running this notebook:\n"
        "    export HF_TOKEN=hf_..."
    )

os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)

credentials_path = "/mnt/cluster_storage/gcp_adc.json"
with open(credentials_path) as f:
    project_id = json.load(f).get("quota_project_id")

ray.init(
    runtime_env={
        "py_executable": "uv run",
        "working_dir": ".",
        "env_vars": {
            "HF_TOKEN": hf_token,
            "GOOGLE_APPLICATION_CREDENTIALS": credentials_path,
            "GOOGLE_CLOUD_PROJECT": project_id,
        },
    },
    ignore_reinit_error=True,
)
```

## 1. Configuration


```python
STATS_PATH             = "/mnt/cluster_storage/stats_droid.json"
EPISODE_INDEX_PATH     = "./data/episodes_droid_v1.0.1.parquet"
NUM_EPISODES_TO_PROCESS = 10

RUN_NAME         = "pi05-droid-finetune"
RUN_STORAGE_PATH = "/mnt/cluster_storage/ray_train_runs/pi05_droid"
```

## 2. Normalization statistics

Before training, the preprocessor needs per-feature **mean and standard deviation** for robot actions and state so it can normalize inputs to zero mean / unit variance.

`compute_or_load_stats` runs a Ray Data pipeline that streams through the DROID HDF5 files, computes partial stats per episode in parallel, then aggregates them into a single JSON file. On subsequent runs it loads the cached file immediately.


```python
"""
Return normalization statistics for the DROID dataset.

If stats_path already exists the file is loaded and returned immediately,
skipping all I/O.  Otherwise a Ray Data pipeline streams through the HDF5
files, computes per-feature mean and standard deviation for robot actions
and state, writes the result to stats_path as JSON, and returns the dict.

Pipeline:
    read_parquet                  — loads the episode index (one row per episode)
    map(harmonize_episode_paths)  — normalizes truncated GCS paths to canonical form
    filter                        — drops episodes missing an hdf5_path
    map(extract)                  — streams each HDF5 from GCS, pulls action/state
    map(episode_stats)            — computes partial mean/std per episode
    aggregate                     — reduces partials into final per-feature stats
"""
from vla.data import harmonize_episode_paths
from vla.stats import (
    compute_episode_stats,
    compute_stats,
    extract_episode_action_and_state,
)

if os.path.exists(STATS_PATH):
    with open(STATS_PATH) as f:
        stats=json.load(f)
else:
    stats_ds = (
        ray.data
        .read_parquet(episode_index_path)
        .limit(num_episodes)
        .map(harmonize_episode_paths)
        .filter(lambda ep: bool(ep.get("hdf5_path")))
        .map(extract_episode_action_and_state)
        .map(compute_episode_stats)
    )

    stats = compute_stats(stats_ds)

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

```

## 3. Dataset pipeline

With the Ray Data pipeline defined, we have a fully lazy, streaming dataset ready to feed the training loop. Here's how data flows at runtime:

**GCS (source)** — HDF5 files (robot state and actions) and MP4 files (wrist camera video) are stored in Google Cloud Storage. Each episode is an HDF5/MP4 pair; the Parquet index tells Ray where to find them. Data is never copied to a local disk — workers stream directly from GCS using `smart_open`.

**Ray Data CPU workers (preprocessing)** — CPU actors read files from GCS, decode wrist camera video with PyAV, assemble per-timestep rows, and run `preprocess_batch` to produce normalized `float32` tensors in PI0.5 format: `observation.images.base_0_rgb` (CHW), `observation.state` (7-dim), and `action` (7-dim). Completed batches are written into the Ray object store and held there until a GPU worker requests them.

**Ray Train GPU workers (training)** — each GPU worker calls `ray.train.get_dataset_shard("train")` to receive its partition of the stream. Workers run the PI0.5 forward pass, compute the loss, and synchronize gradients via DDP. CPU preprocessing and GPU training **overlap in time** — while one batch is being computed on the GPU, CPU workers are already assembling the next batch.

Neither side sits idle: GPUs are never stalled waiting for data to decode, and CPU workers are never blocked waiting for the GPU to consume their output.

![VLA Fine-tuning Pipeline](images/vla-fine-tuning-pipeline.png)


```python
import ray
from vla.data import episode_to_training_rows, harmonize_episode_paths, preprocess_batch

ds = (
    ray.data
    .read_parquet(EPISODE_INDEX_PATH)
    .limit(NUM_EPISODES_TO_PROCESS)
    .map(harmonize_episode_paths)                  # normalize truncated GCS paths to canonical form
    .flat_map(episode_to_training_rows)            # episode  → timestep rows
    .map_batches(preprocess_batch, batch_size=32)  # rows     → PI0.5 tensors
)
```

## 4. Distributed training


```python
from ray.train import FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from vla.train_worker import train_loop_per_worker

scaling_config = ScalingConfig(
    num_workers=2,
    use_gpu=True,
    accelerator_type="A100",
)

run_config = RunConfig(
        name=RUN_NAME,
        storage_path=RUN_STORAGE_PATH,
        failure_config=FailureConfig(max_failures=1),
)

train_loop_config = {
    "stats_path": STATS_PATH,
    "num_epochs": 2,
    "batch_size": 1,
    "grad_accum": 1,
    "lr":         1e-4,
    "max_len":    512,   # truncate token sequences to fit smaller GPUs
}

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=scaling_config,
    run_config=run_config,
    datasets={"train": ds},
)

result = trainer.fit()
```

## Summary

This notebook demonstrates distributed fine-tuning of the **PI0.5 Vision-Language-Action (VLA) model** on the **Droid v1.0.1 robot manipulation dataset** using Ray on Anyscale.

**What was covered:**

1. **Environment setup** — Ray is initialized with a `uv`-managed runtime environment, and a HuggingFace token is configured to access gated model weights.

2. **Configuration** — Key paths and hyperparameters are defined: episode index (Parquet), normalization stats cache, training run name/storage, and training settings (epochs, batch size, learning rate, sequence length).

3. **Normalization statistics (Ray Data)** — A one-time streaming scan over the Droid HDF5 files computes per-feature mean and standard deviation for robot actions and state. Ray Data's parallel CPU workers stream directly from GCS, extract episode data, compute partial stats, and aggregate them into a `stats.json` file reused across runs.

4. **Dataset pipeline (Ray Data)** — A fully lazy, streaming pipeline reads the Parquet episode index, expands each episode into per-timestep rows (`episode_to_training_rows`), and applies batched preprocessing (`preprocess_batch`) to produce normalized PI0.5-format tensors: camera images (CHW), robot state (7-dim), and actions (7-dim). CPU preprocessing and GPU training overlap in time via the Ray object store.

5. **Distributed training (Ray Train)** — A `TorchTrainer` runs the PI0.5 forward/backward pass across 2 A100 GPU workers using DDP. Fault tolerance (`max_failures=1`), run checkpointing to shared storage, and dataset sharding are handled automatically by Ray Train.
