# Distributed VLA Fine-Tuning with Ray

**⏱️ Time to complete:** 30 min

This notebook fine-tunes the **PI0.5 Vision-Language-Action (VLA)** model on a
[LeRobot](https://github.com/huggingface/lerobot) robotics dataset stored in S3.

## Why Ray?

Fine-tuning a VLA model involves two fundamentally different workloads:

1. **DATA** — Decoding mp4 video, renaming/transposing images, normalizing
   actions/states. This is CPU-heavy, I/O-heavy, and embarrassingly parallel.

2. **TRAINING** — Forward/backward passes on a large transformer with mixed
   precision, gradient accumulation, and distributed data parallelism.
   This is GPU-heavy.

Without Ray, you'd stitch these together yourself: write a PyTorch Dataset
that blocks GPU workers while they decode video, or build a separate
preprocessing pipeline and serialize to disk. Either way, the GPUs sit idle
waiting for data, and you own all the plumbing.

Ray splits this cleanly:

- **[Ray Data](https://docs.ray.io/en/latest/data/data.html)** handles (1): it streams, decodes, and preprocesses data on
  auto-scaled CPU workers, feeding batches to GPU workers through a
  pipelined, backpressure-aware channel. No GPU ever stalls waiting for
  a video decode.

- **[Ray Train](https://docs.ray.io/en/latest/train/train.html)** handles (2): it launches distributed PyTorch workers across
  GPUs, manages DDP synchronization, checkpointing, and fault recovery.
  If a worker dies, training resumes from the last checkpoint — you
  don't re-run from scratch.

The result: you write a short, single-file script that scales from 1 GPU
on a laptop to 32 GPUs across a cluster, with zero changes to your code.

## Architecture Overview

```
+-----------+       +------------+       +-----------+
| S3 Bucket | ----> | Ray Data   | ----> | Ray Train |
| (LeRobot  |       | (CPU pool) |       | (N GPUs)  |
| mp4+pqt)  |       |            |       |           |
+-----------+       +------------+       +-----------+
                     |                    |
                     | - read parquet     | - load PI0.5
                     | - decode mp4       | - freeze backbone
                     | - rename cameras   | - train action heads
                     | - transpose HWC    | - mixed-precision
                     |   -> CHW float32   | - gradient accum
                     | - stream batches   | - checkpoint & resume
                     +--------------------+---------------------
```

## Files

| File | Description |
|------|-------------|
| `README.ipynb` (this notebook) | Interactive walkthrough — open in Jupyter and run cells top-to-bottom |
| `vla.py` | Job script — same pipeline, submittable with `uv run python vla.py` |
| `util.py` | Training utilities — model loading, checkpointing, collation, training step helpers |
| `lerobot_datasource.py` | Custom Ray Data datasource for LeRobot v3 datasets |

## 0. Setup

### Dependencies

This template uses [uv](https://docs.astral.sh/uv/) for dependency management.
Open a terminal and run:

```bash
uv sync
```

When prompted for a Jupyter kernel, select the Python environment named **vla** (`.venv/bin/python`).

### HuggingFace Token

PI0.5 depends on [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) as a vision backbone.
Google requires you to **accept the model license** before the weights can be downloaded.

1. Navigate to the [model page](https://huggingface.co/google/paligemma-3b-pt-224) and accept the license.
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Export it before running:

```bash
export HF_TOKEN=hf_...
```

## GPU Requirements

This template supports both **A100** and **L4** GPUs. Pick one and set
`accelerator_type` plus matching hyperparameters in
[`ScalingConfig`](https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html)
and `train_loop_config` (see [Section 5](#5-launch-distributed-training)).

|               | A100 (80 GB) | L4 (24 GB) |
|---------------|--------------|------------|
| `batch_size`  | 4            | 1          |
| `grad_accum`  | 2            | 8          |
| `num_workers` | 4            | 4          |

Pick instances with **≥ 64 GB RAM** to avoid OOM during model checkpointing.

**A100s** have enough VRAM to run larger batch sizes with minimal gradient
accumulation. This keeps GPU utilization high and the data pipeline straightforward.

**L4s** require `batch_size=1` to fit in 24 GB VRAM. To compensate, increase
`grad_accum` so the effective batch size stays reasonable. Smaller per-step batches
mean faster consumption, which can cause the Ray Data pipeline to fall behind and
spill to disk. If you see frequent object store spillage, reduce the data pipeline's
throughput to match training speed — for example, lower `map_batches` concurrency
or decrease the number of CPU data workers so the producer and consumer stay in balance.

## 1. Configuration


```python
import logging
import os

import numpy as np
import util

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

RUN_NAME         = "pi05-xvla-soft-fold-finetune"
RUN_STORAGE_PATH = "/mnt/cluster_storage/ray_train_runs/pi05_xvla_soft_fold"

# Dataset location -- a LeRobot v3 dataset with parquet + mp4 files in S3.
DATASET_PATH = "s3://anyscale-public-robotics-datasets/lerobot/lerobot/xvla-soft-fold"

# The dataset's camera column names don't match what PI0.5 expects.
# This map renames them during preprocessing so the model receives the
# feature names it was pretrained on.
CAMERA_RENAME = {
    "observation.images.cam_high":        "observation.images.base_0_rgb",
    "observation.images.cam_left_wrist":  "observation.images.left_wrist_0_rgb",
    "observation.images.cam_right_wrist": "observation.images.right_wrist_0_rgb",
}

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("Set HF_TOKEN before running:  export HF_TOKEN=hf_...")
```

## 2. Connect to Ray


```python
# The Anyscale base image registers a RAY_RUNTIME_ENV_HOOK that imports a
# package only available in the system Python. With `py_executable="uv run"`,
# the driver runs from the project venv (no `uv run` ancestor process), so
# Ray falls through to the hook and crashes on the missing import. Disable it.
import os
os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)

import ray

ray.init(
    runtime_env={
        "py_executable": "uv run",
        "working_dir": ".",
        "env_vars": {"HF_TOKEN": HF_TOKEN},
    },
    ignore_reinit_error=True,
)
```

## 3. Ray Data — Build the preprocessing pipeline

These functions run on Ray Data's **CPU worker pool** — NOT on the GPU training
workers. Ray Data calls them automatically as it streams data from S3,
keeping GPU workers fed without blocking them on I/O or image decoding.

This separation is the key scaling advantage: you can scale CPU and GPU
resources independently. Need faster preprocessing? Add CPU workers.
Need more training throughput? Add GPUs. Ray handles the plumbing.

Docs: [Transforming Data](https://docs.ray.io/en/latest/data/transforming-data.html)


```python
def rename_columns(row: dict, rename: dict[str, str]) -> dict:
    """Rename dataset camera columns to match PI0.5's expected feature names."""
    return {rename.get(k, k): v for k, v in row.items()}


def transpose_images(batch: dict, camera_keys: list[str]) -> dict:
    """Convert camera images from HWC uint8 to CHW float32.

    PI0.5 (like most vision models) expects (batch, channels, height, width).
    The raw dataset stores images as (height, width, channels) uint8.
    """
    result = dict(batch)
    for key in camera_keys:
        result[key] = np.transpose(np.stack(batch[key]), (0, 3, 1, 2)).astype(np.float32)
    return result
```

### Build the pipeline

This is **lazy** — no data is read until training starts.
Ray Data will stream from S3, decode mp4 video frames, rename camera
columns, and transpose images — all on CPU workers, in parallel, with
backpressure so it never overwhelms GPU memory.

Docs: [Loading Data](https://docs.ray.io/en/latest/data/loading-data.html)


```python
from lerobot_datasource import LeRobotDatasource

source = LeRobotDatasource(DATASET_PATH)
stats = util.extract_stats(source)                                         # normalization stats for the training loop
image_keys = util.renamed_image_keys(source, CAMERA_RENAME)                # camera columns after renaming

ds = (
    ray.data
    .read_datasource(source)                                               # Stream from S3
    .map(rename_columns, fn_args=(CAMERA_RENAME,))                         # Rename camera columns
    .map_batches(transpose_images, batch_size=32, fn_args=(image_keys,))   # HWC -> CHW
)
```

## 4. Ray Train — Distributed training loop

`train_loop_per_worker` runs inside each GPU worker. Ray Train launches N
copies of it (one per GPU), each receiving its own shard of the streaming
data.

Under the hood, Ray Train:
- Sets up `torch.distributed` (NCCL) across all workers
- Wraps the model in DDP via `prepare_model()`
- Streams data shards via Ray Data (no manual `DistributedSampler` needed)
- Coordinates checkpointing so only rank 0 writes, but all ranks sync
- Handles fault tolerance: if a worker dies, training restarts from the
  last checkpoint, not from scratch

All of this happens transparently. The code below reads like **single-GPU
training code** — Ray handles the distribution.

Docs: [Getting Started with PyTorch](https://docs.ray.io/en/latest/train/getting-started-pytorch.html)


```python
import torch


def train_loop_per_worker(config: dict):
    """Per-GPU training entry point. Ray Train calls this on each worker."""

    import ray.train
    import ray.train.torch

    device = torch.device("cuda")

    # -- Load model and freeze backbone ----------------------------------------
    #
    # load_pi05_policy() loads the pretrained PI0.5 model, applies the
    # attention mask patch, and freezes the backbone -- only the action/time
    # projection heads are trainable (see util.py for details).
    #
    # ray.train.torch.prepare_model() wraps the model in DistributedDataParallel.
    # Ray Train has already called torch.distributed.init_process_group() for us
    # with the NCCL backend -- prepare_model() just applies the DDP wrapper.
    # Docs: https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.prepare_model.html

    policy = util.load_pi05_policy()
    policy = ray.train.torch.prepare_model(policy)                         # <-- RAY TRAIN: wrap in DDP

    # AdamW optimizer -- only updates the unfrozen action/time projection heads.
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=config.get("lr", 1e-4),
    )

    # GradScaler for mixed-precision training (fp16 forward, fp32 gradients).
    scaler = torch.amp.GradScaler("cuda")

    # -- Resume from checkpoint (if any) ---------------------------------------
    #
    # ray.train.get_checkpoint() returns the last checkpoint saved by
    # ray.train.report() -- if this is a fresh run, it returns None.
    # On failure recovery, Ray automatically passes the most recent checkpoint
    # back here, so training resumes from where it left off.
    # Docs: https://docs.ray.io/en/latest/train/user-guides/checkpoints.html

    checkpoint = ray.train.get_checkpoint()                                # <-- RAY TRAIN: fault tolerance
    if checkpoint:
        start_epoch, step = util.load_checkpoint(checkpoint, policy, optimizer, scaler)
    else:
        start_epoch, step = 0, 0

    # -- Build preprocessor with normalization stats ---------------------------

    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, _ = make_pre_post_processors(
        policy.module.config, pretrained_path="lerobot/pi05_base", dataset_stats=config["stats"],
    )

    # -- Hyperparameters and LR schedule ----------------------------------------

    batch_size      = int(config.get("batch_size", 1))
    grad_accum      = int(config.get("grad_accum", 1))
    num_epochs      = int(config.get("num_epochs", 1))
    max_len         = int(config.get("max_len", 512))
    max_train_steps = config.get("max_train_steps")                        # CI smoke-test cap; None = no cap

    num_workers = ray.train.get_context().get_world_size()                 # <-- RAY: how many GPU workers?
    scheduler = util.build_lr_scheduler(optimizer, config, num_workers, last_step=step)

    # =========================================================================
    # Training loop
    # =========================================================================
    #
    # Notice what's NOT here:
    #   - No DistributedSampler or manual sharding
    #   - No torch.distributed.init_process_group()
    #   - No data loading threads or prefetch queues
    #   - No checkpoint path management
    #
    # To scale from 1 to N GPUs: change ScalingConfig.num_workers. Done.
    # =========================================================================

    shard = ray.train.get_dataset_shard("train")                           # <-- RAY: get this worker's data shard

    for epoch in range(start_epoch, num_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum, epoch_loss_count = 0.0, 0
        accum_count = 0

        # iter_torch_batches() streams pre-processed batches from the Ray Data
        # CPU worker pool directly into GPU memory.
        # Docs: https://docs.ray.io/en/latest/data/api/doc/ray.data.DataIterator.iter_torch_batches.html
        for batch in shard.iter_torch_batches(                             # <-- RAY DATA: streaming batches
            batch_size=batch_size,
            collate_fn=util.NumpyToTorchCollate(device),
        ):
            # Standard PyTorch: forward + scaled backward (see util.py)
            loss_val = util.train_step(policy, batch, preprocessor, max_len, grad_accum, scaler)
            step += 1
            accum_count += 1
            epoch_loss_sum += loss_val
            epoch_loss_count += 1

            if accum_count % grad_accum == 0:
                # Standard PyTorch: unscale, clip, step, zero_grad (see util.py)
                util.optimizer_step(policy, optimizer, scaler, scheduler)
                accum_count = 0

            # Log every 10 steps so you can watch training progress.
            if step % 10 == 0:
                log.info("epoch=%d  step=%d  loss=%.4f  lr=%.2e", epoch, step, loss_val, scheduler.get_last_lr()[0])

            # smoke-test cap (unset MAX_TRAIN_STEPS for full training)
            if max_train_steps and step >= max_train_steps:
                break

        # Flush any leftover accumulated gradients at epoch end.
        if accum_count > 0:
            util.optimizer_step(policy, optimizer, scaler, scheduler)

        # -- End of epoch: report metrics and checkpoint -----------------------
        #
        # ray.train.report() is a synchronization barrier -- every worker must
        # call it. Only rank 0 creates the actual checkpoint.
        # Docs: https://docs.ray.io/en/latest/train/api/doc/ray.train.report.html
        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        metrics = {"epoch": epoch, "steps": step, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]}

        if ray.train.get_context().get_world_rank() == 0:                  # <-- RAY TRAIN
            checkpoint = util.make_checkpoint(policy, optimizer, scaler, epoch, step)
            ray.train.report(metrics, checkpoint=checkpoint)               # <-- RAY TRAIN
        else:
            ray.train.report(metrics)                                      # <-- RAY TRAIN

        # smoke-test cap (unset MAX_TRAIN_STEPS for full training)
        if max_train_steps and step >= max_train_steps:
            break
```

## 5. Launch distributed training

[`TorchTrainer`](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html)
is the single entry point for distributed PyTorch on Ray.
To scale to 8, 16, or 32 GPUs: just change `num_workers`. The training
code, data pipeline, and checkpointing all adapt automatically.


```python
import ray.train
import ray.train.torch

# GPU requirements:
#   A100 (80 GB) -- batch_size=4, grad_accum=2 works well.
#   L4   (24 GB) -- use batch_size=1, grad_accum=8 to avoid OOM. Smaller
#                   batches consume data faster, which can cause object store
#                   spillage. If this happens, reduce data pipeline concurrency
#                   to keep producers and consumers in balance.
#
# smoke-test cap (unset MAX_TRAIN_STEPS for full training)
# training at N micro-batches per worker (None = full training).
train_loop_config = {
    "stats": stats,
    "total_rows": source.meta.total_frames,
    "num_epochs": 2,
    "batch_size": 1,
    "grad_accum": 8,
    "lr":         1e-4,
    "warmup_frac": 0.1,
    "max_len":    512,
    "max_train_steps": int(os.environ["MAX_TRAIN_STEPS"]) if os.environ.get("MAX_TRAIN_STEPS") else None,
}

# ScalingConfig controls parallelism:
#   num_workers=4       -> 4 GPU workers, each running train_loop_per_worker
#   use_gpu=True        -> each worker gets 1 GPU
#   accelerator_type    -> request a specific GPU type (e.g. "L4", "A100")
# Docs: https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html
scaling_config = ray.train.ScalingConfig(num_workers=4, use_gpu=True, accelerator_type="L4")

# RunConfig controls experiment tracking and fault tolerance:
#   storage_path -> shared storage for checkpoints (cluster NFS or S3)
#   max_failures=1 -> auto-restart once on failure, resuming from checkpoint
# Docs: https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html
run_config = ray.train.RunConfig(
    name=RUN_NAME,
    storage_path=RUN_STORAGE_PATH,
    failure_config=ray.train.FailureConfig(max_failures=1),
)

trainer = ray.train.torch.TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=scaling_config,
    run_config=run_config,
    datasets={"train": ds},
)

result = trainer.fit()
print(f"Training complete: {result}")
```

## Summary

This notebook demonstrated distributed fine-tuning of the **PI0.5 VLA model** on a
**LeRobot robotics dataset** using Ray.

**What was covered:**

1. **Configuration** — Dataset path, camera column rename map, and HuggingFace token.

2. **Ray Data preprocessing** — A lazy, streaming pipeline that reads from S3,
   decodes mp4 video, renames camera columns, and transposes images from HWC to CHW —
   all on CPU workers, in parallel, with backpressure.

3. **Ray Train distributed training** — `train_loop_per_worker` runs on each GPU
   with automatic DDP wrapping (`prepare_model`), data sharding (`get_dataset_shard`),
   and fault-tolerant checkpointing (`get_checkpoint` / `report`).

4. **Launch** — `TorchTrainer` ties it all together. To scale from 1 to N GPUs,
   change `ScalingConfig.num_workers`. Everything else adapts automatically.

### Job submission

This notebook has a companion script, **`vla.py`**, that mirrors the same pipeline
and can be submitted as a standalone job:

```bash
export HF_TOKEN=hf_...
uv run python vla.py
```

