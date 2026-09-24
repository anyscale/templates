"""
Distributed VLA Fine-Tuning with Ray
=====================================

This script fine-tunes the PI0.5 Vision-Language-Action (VLA) model on a
LeRobot robotics dataset stored in S3.

WHY RAY?
--------
Fine-tuning a VLA model involves two fundamentally different workloads:

  1. DATA -- Decoding mp4 video, renaming/transposing images, normalizing
     actions/states. This is CPU-heavy, I/O-heavy, and embarrassingly parallel.

  2. TRAINING -- Forward/backward passes on a large transformer with mixed
     precision, gradient accumulation, and distributed data parallelism.
     This is GPU-heavy.

Without Ray, you'd stitch these together yourself: write a PyTorch Dataset
that blocks GPU workers while they decode video, or build a separate
preprocessing pipeline and serialize to disk. Either way, the GPUs sit idle
waiting for data, and you own all the plumbing.

Ray splits this cleanly:

  * Ray Data handles (1): it streams, decodes, and preprocesses data on
    auto-scaled CPU workers, feeding batches to GPU workers through a
    pipelined, backpressure-aware channel. No GPU ever stalls waiting for
    a video decode.
    Docs: https://docs.ray.io/en/latest/data/data.html

  * Ray Train handles (2): it launches distributed PyTorch workers across
    GPUs, manages DDP synchronization, checkpointing, and fault recovery.
    If a worker dies, training restarts from the last *completed epoch* --
    not from scratch, but not from the point of failure either. This
    template checkpoints once per epoch and its input stream cannot be
    fast-forwarded; see "CHECKPOINT AND RESUME -- WHAT IS AND ISN'T
    RESTORED" below.
    Docs: https://docs.ray.io/en/latest/train/train.html

The result: you write a short, single-file script that scales from 1 GPU
on a laptop to 32 GPUs across a cluster, with zero changes to your code.

ARCHITECTURE OVERVIEW
---------------------

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
                         | - stream batches   | - checkpoint per epoch
                         +--------------------+---------------------

CHECKPOINT AND RESUME -- WHAT IS AND ISN'T RESTORED
---------------------------------------------------
Read this before relying on restart behaviour for a long run.

RESTORED: model weights, optimizer state, gradient scaler state, and the
epoch/step counters (util.make_checkpoint / util.load_checkpoint).

NOT RESTORED: the position of the Ray Data input stream. lerobot_datasource.py
has no concept of an offset -- LeRobotDatasource.get_read_tasks() always plans
the same row ranges starting at row 0 -- and nothing in the checkpoint could
tell it otherwise. Ray Data rebuilds the whole pipeline from the datasource on
every restart.

What that means in practice:

  * Resume is epoch-granular, not step-granular. A checkpoint is only written
    at an epoch boundary, so on restart the loop re-enters at `epoch + 1` with
    the stream back at the beginning. The data stays consistent -- re-reading
    the dataset is what an epoch is -- but every batch processed in the
    interrupted epoch is thrown away and re-consumed. With num_epochs=2 below,
    a failure late in the second epoch costs close to half the run.
  * FailureConfig(max_failures=1) allows one such restart after a worker
    error; a second worker error ends the run. Node preemptions are budgeted
    separately (max_preemption_failures, unlimited by default on Ray 2.58),
    and each one costs the same epoch replay.
  * `step` is a count of micro-batches consumed, not a row offset into the
    dataset. Don't read it as one.

Step-level resume would mean teaching LeRobotDatasource to start at a given
row, recording that row in the checkpoint, and threading it back into
read_datasource() on restart -- a change to the read path, not to this file.

Usage:
    export HF_TOKEN=hf_...
    uv run python vla.py
"""

import logging
import os

import numpy as np
import util

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

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


# ============================================================================
# Connect to Ray
# ============================================================================

import ray

ray.init(
    runtime_env={
        "pip": os.path.join(os.getcwd(), "python_depset.lock"),
        "working_dir": ".",
        "env_vars": {"HF_TOKEN": HF_TOKEN},
    },
    ignore_reinit_error=True,
)


# ============================================================================
# RAY DATA -- Build the preprocessing pipeline
# ============================================================================
#
# These functions run on Ray Data's CPU worker pool -- NOT on the GPU training
# workers. Ray Data calls them automatically as it streams data from S3,
# keeping GPU workers fed without blocking them on I/O or image decoding.
#
# This separation is the key scaling advantage: you can scale CPU and GPU
# resources independently. Need faster preprocessing? Add CPU workers.
# Need more training throughput? Add GPUs. Ray handles the plumbing.
#
# See: https://docs.ray.io/en/latest/data/transforming-data.html
# ============================================================================

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


# -- Build the pipeline (lazy -- no data is read until training starts) --------
#
# Ray Data will stream from S3, decode mp4 video frames, rename camera
# columns, and transpose images -- all on CPU workers, in parallel, with
# backpressure so it never overwhelms GPU memory.
#
# See: https://docs.ray.io/en/latest/data/loading-data.html

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


# ============================================================================
# RAY TRAIN -- Distributed training loop (runs on GPU workers)
# ============================================================================
#
# train_loop_per_worker runs inside each GPU worker. Ray Train launches N
# copies of it (one per GPU), each receiving its own shard of the streaming
# data.
#
# Under the hood, Ray Train:
#   - Sets up torch.distributed (NCCL) across all workers
#   - Wraps the model in DDP via prepare_model()
#   - Streams data shards via Ray Data (no manual DistributedSampler needed)
#   - Coordinates checkpointing so only rank 0 writes, but all ranks sync
#   - Handles fault tolerance: if a worker dies, training restarts from the
#     last checkpoint, not from scratch -- which here means the last completed
#     epoch, since that is the only place this loop checkpoints (see the
#     "CHECKPOINT AND RESUME" section at the top of this file)
#
# All of this happens transparently. The code below reads like single-GPU
# training code -- Ray handles the distribution.
#
# See: https://docs.ray.io/en/latest/train/getting-started-pytorch.html
# ============================================================================

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
    # See: https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.prepare_model.html

    policy = util.load_pi05_policy()
    policy = ray.train.torch.prepare_model(policy)                         # <-- RAY TRAIN: wrap in DDP

    # AdamW optimizer -- only updates the unfrozen action/time projection heads.
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=config.get("lr", 1e-4),
    )

    # GradScaler for mixed-precision training (fp16 forward, fp32 gradients).
    # Scales the loss before backward() to prevent fp16 underflow, then
    # unscales gradients before the optimizer step. Works transparently
    # with DDP -- Ray doesn't change how PyTorch mixed precision works.
    scaler = torch.amp.GradScaler("cuda")

    # -- Resume from checkpoint (if any) ---------------------------------------
    #
    # ray.train.get_checkpoint() returns the last checkpoint saved by
    # ray.train.report() -- if this is a fresh run, it returns None.
    # On failure recovery, Ray automatically passes the most recent checkpoint
    # back here, so training resumes from the start of the epoch after the last
    # one that completed. Not from where it left off: this loop checkpoints
    # once per epoch, and the input stream restarts at row 0 regardless (see
    # "CHECKPOINT AND RESUME" at the top of this file).
    # See: https://docs.ray.io/en/latest/train/user-guides/checkpoints.html

    checkpoint = ray.train.get_checkpoint()                                # <-- RAY TRAIN: fault tolerance
    if checkpoint:
        start_epoch, step = util.load_checkpoint(checkpoint, policy, optimizer, scaler)
    else:
        start_epoch, step = 0, 0

    # -- Build preprocessor with normalization stats ---------------------------

    from lerobot.policies.factory import make_pre_post_processors
    from huggingface_hub import snapshot_download
    preprocessor, _ = make_pre_post_processors(
        policy.module.config, pretrained_path=snapshot_download("lerobot/pi05_base", revision="a538eb273274"), dataset_stats=config["stats"],
    )

    # -- Hyperparameters and LR schedule ----------------------------------------

    batch_size  = int(config.get("batch_size", 1))
    grad_accum  = int(config.get("grad_accum", 1))
    num_epochs  = int(config.get("num_epochs", 1))
    max_len     = int(config.get("max_len", 512))

    num_workers = ray.train.get_context().get_world_size()                 # <-- RAY: how many GPU workers?
    scheduler = util.build_lr_scheduler(optimizer, config, num_workers, last_step=step)

    # =========================================================================
    # Training loop
    # =========================================================================
    #
    # Read this loop and notice what's NOT here:
    #   - No DistributedSampler or manual sharding
    #   - No torch.distributed.init_process_group()
    #   - No data loading threads or prefetch queues
    #   - No checkpoint path management
    #
    # Ray handles all of that. What remains is essentially single-GPU code:
    #   1. Get a data shard from Ray Data       (get_dataset_shard)
    #   2. Stream batches to GPU                (iter_torch_batches)
    #   3. Forward + backward                   (train_step)
    #   4. Optimizer step                       (optimizer_step)
    #   5. Report metrics & checkpoint via Ray   (ray.train.report)
    #
    # To scale from 1 to N GPUs: change ScalingConfig.num_workers -- that is all
    # the *code* needs. Whether it goes faster is a hardware question; see the
    # SCALING CAVEAT above ScalingConfig below.
    # =========================================================================

    # ray.train.get_dataset_shard() returns this worker's slice of the dataset.
    # Ray Data automatically partitions the data across workers -- no manual
    # sharding or DistributedSampler needed.
    # See: https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html
    shard = ray.train.get_dataset_shard("train")                           # <-- RAY: get this worker's data shard

    for epoch in range(start_epoch, num_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum, epoch_loss_count = 0.0, 0
        accum_count = 0

        # iter_torch_batches() streams pre-processed batches from the Ray Data
        # CPU worker pool directly into GPU memory. The data was already decoded
        # (mp4), renamed, and transposed on CPU workers -- zero GPU time spent
        # on data loading. Backpressure ensures we never OOM.
        # See: https://docs.ray.io/en/latest/data/api/doc/ray.data.DataIterator.iter_torch_batches.html
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

        # Flush any leftover accumulated gradients at epoch end.
        if accum_count > 0:
            util.optimizer_step(policy, optimizer, scaler, scheduler)

        # -- End of epoch: report metrics and checkpoint -----------------------
        #
        # ray.train.report() is a synchronization barrier -- every worker must
        # call it. Only rank 0 creates the actual checkpoint (model weights +
        # optimizer + scaler state); the others just report metrics.
        #
        # On failure, Ray restarts workers and feeds the checkpoint back to
        # ray.train.get_checkpoint() above -- automatic resume, zero user code.
        #
        # LIMITATION -- this is the only place the run checkpoints, so it is
        # also the only place it can resume from. make_checkpoint() stores
        # model/optimizer/scaler/epoch/step but cannot store where the input
        # stream had got to: lerobot_datasource.py has no offset, so Ray Data
        # replans from row 0 on every restart. A crash at 90% of an epoch
        # therefore replays that whole epoch. If your epochs are long, add
        # mid-epoch checkpointing *and* datasource resumption together --
        # adding the first alone would resume the model mid-epoch while the
        # stream restarts from the beginning, which silently re-trains on data
        # the model has already seen and makes `step` disagree with how far
        # through the data the run actually is.
        # See: https://docs.ray.io/en/latest/train/api/doc/ray.train.report.html
        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        metrics = {"epoch": epoch, "steps": step, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]}

        if ray.train.get_context().get_world_rank() == 0:                  # <-- RAY TRAIN
            checkpoint = util.make_checkpoint(policy, optimizer, scaler, epoch, step)
            ray.train.report(metrics, checkpoint=checkpoint)               # <-- RAY TRAIN
        else:
            ray.train.report(metrics)                                      # <-- RAY TRAIN


# ============================================================================
# Launch distributed training
# ============================================================================
#
# TorchTrainer is the single entry point for distributed PyTorch on Ray.
# To scale to 8, 16, or 32 GPUs: change num_workers, and give the compute
# config the GPUs to match -- the shipped one tops out at 4. The training code,
# data pipeline, and checkpointing all adapt automatically -- but see the
# caveat below before treating throughput as a function of num_workers.
# See: https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html
#
# SCALING CAVEAT -- num_workers is not the binding constraint on multi-node
# ---------------------------------------------------------------------------
# The compute configs shipped with this template (configs/vla-fine-tuning/)
# request four single-GPU nodes: one L4 each, min_nodes=max_nodes=4. That is
# the arrangement in which *every* DDP gradient allreduce crosses the network,
# because no two GPUs share a host. At num_workers=4 that config is already
# full, so scaling out on this shape means raising max_nodes as well, and every
# node added brings one GPU and one more host for each allreduce to cross --
# measured throughput can flatten or regress while the config looks like it
# scaled.
#
# Two things to do before scaling out:
#
#   1. Verify the interconnect is actually being used. High-bandwidth fabrics
#      (EFA, InfiniBand, RDMA) do not engage by default -- they need the right
#      instance type, driver, and NCCL configuration, and NCCL will quietly
#      fall back to TCP when any of that is missing. The fallback works; it is
#      just one to two orders of magnitude slower on an allreduce, and nothing
#      in the training logs says so. Run a multi-node NCCL allreduce benchmark
#      first and confirm the bus bandwidth is what the hardware advertises.
#      (Field observation from other hardware, not measured for this template:
#      the same 16-GPU allreduce ran ~25x apart -- tens of GB/s versus
#      hundreds -- purely on whether RDMA was in use.)
#   2. Prefer packing GPUs into fewer nodes over spreading them across more.
#      Eight GPUs on one host allreduce over NVLink/PCIe; eight GPUs on eight
#      hosts allreduce over the network. Choosing one multi-GPU instance type
#      instead of N single-GPU ones is usually a bigger win than any change to
#      num_workers.
# ============================================================================

import ray.train
import ray.train.torch

# GPU requirements:
#   A100 (80 GB) -- batch_size=4, grad_accum=2 works well.
#   L4   (24 GB) -- use batch_size=1, grad_accum=8 to avoid OOM. Smaller
#                   batches consume data faster, which can cause object store
#                   spillage. If this happens, reduce data pipeline concurrency
#                   to keep producers and consumers in balance.
train_loop_config = {
    "stats": stats,
    "total_rows": source.meta.total_frames,
    "num_epochs": 2,
    "batch_size": 1,
    "grad_accum": 8,
    "lr":         1e-4,
    "warmup_frac": 0.1,
    "max_len":    512,
}

# ScalingConfig controls parallelism:
#   num_workers=4  -> 4 GPU workers, each running train_loop_per_worker
#   use_gpu=True   -> each worker gets 1 GPU
# With the shipped compute config that is 4 single-GPU nodes -- every allreduce
# goes over the network. Read the SCALING CAVEAT above before raising this.
# See: https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html
scaling_config = ray.train.ScalingConfig(num_workers=4, use_gpu=True)

# RunConfig controls experiment tracking and fault tolerance:
#   storage_path -> shared storage for checkpoints (cluster NFS or S3)
#   max_failures=1 -> auto-restart once on failure, resuming from checkpoint
# See: https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html
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
log.info("Training complete: %s", result)
