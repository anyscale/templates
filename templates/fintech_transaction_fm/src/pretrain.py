"""Distributed masked-feature-modeling pretraining with Ray Train.

The training function is plain PyTorch. Ray Train handles the distributed parts:
worker setup, dataset sharding, DDP/FSDP wrapping, checkpointing, and fault
tolerance — the same code runs on 1 CPU worker (CI mini) or N GPU workers
(the real distributed story) by changing only ``ScalingConfig``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import defaultdict

import torch

import ray
import ray.train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from .model import build_model, mask_batch


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def train_func(config: dict):
    # Same init on every rank (DDP would broadcast anyway; this also makes
    # *runs* reproducible for A/B comparisons), then per-rank reseed below so
    # MLM masking differs across workers deterministically.
    torch.manual_seed(config.get("seed", 0))

    vocab_path = config["vocab_path"]
    with open(vocab_path) as f:
        vocab = json.load(f)
    dynamic_fields = vocab["dynamic_fields"]
    weighting = config.get("loss_weighting", "uncertainty")

    # Per-column dtypes: tokens are int64, the soft-bin amount weight is float32.
    dtypes = {"d_amount_frac": torch.float32} if vocab.get("amount_mode") == "soft" else None

    model = build_model(vocab_path, arch=config["arch"], max_len=config["max_len"])

    use_fsdp = config.get("use_fsdp", False) and torch.cuda.is_available()
    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        model = model.to(ray.train.torch.get_device())
        model = FSDP(model)
    else:
        model = ray.train.torch.prepare_model(model)
    base = _unwrap(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    train_shard = ray.train.get_dataset_shard("train")
    mask_prob = config.get("mask_prob", 0.15)
    torch.manual_seed(config.get("seed", 0) + ray.train.get_context().get_world_rank())

    for epoch in range(config["epochs"]):
        model.train()
        running, n_batches = 0.0, 0
        ce_sum, acc_sum, tot_n = defaultdict(float), defaultdict(float), 0
        for batch in train_shard.iter_torch_batches(
            batch_size=config["batch_size"], dtypes=dtypes
        ):
            corrupted, targets, masked = mask_batch(batch, dynamic_fields, mask_prob)
            n = int(masked.sum())
            if n == 0:
                continue
            # Heads + loss run inside forward so DDP all-reduces every param.
            loss, stats = model(corrupted, targets=targets, masked=masked, weighting=weighting)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
            tot_n += n
            for f, d in stats.items():
                ce_sum[f] += d["ce"] * n   # weight per-field means by #masked
                acc_sum[f] += d["acc"] * n

        # The weighted total drifts as the log-variances learn — watch the
        # per-field accuracy and perplexity instead. Perplexity vs. the field's
        # vocab size tells you whether it learned structure (ppl << vocab = good).
        metrics = {"epoch": epoch, "mlm_loss": running / max(n_batches, 1)}
        macro_acc = 0.0
        for f in dynamic_fields:
            mean_ce = ce_sum[f] / max(tot_n, 1)
            acc = acc_sum[f] / max(tot_n, 1)
            metrics[f"acc_{f}"] = acc
            metrics[f"ppl_{f}"] = math.exp(min(mean_ce, 20.0))
            macro_acc += acc
        metrics["acc_macro"] = macro_acc / len(dynamic_fields)

        # Checkpoint on the last epoch (rank 0 writes the weights).
        checkpoint = None
        if epoch == config["epochs"] - 1 and ray.train.get_context().get_world_rank() == 0:
            tmp = tempfile.mkdtemp()
            torch.save(base.state_dict(), os.path.join(tmp, "model.pt"))
            shutil.copy(vocab_path, os.path.join(tmp, "vocab.json"))
            with open(os.path.join(tmp, "model_config.json"), "w") as f:
                json.dump(
                    {
                        "size": config["size"],
                        "max_len": config["max_len"],
                        "arch": config["arch"],
                    },
                    f,
                )
            checkpoint = Checkpoint.from_directory(tmp)
        ray.train.report(metrics, checkpoint=checkpoint)


def pretrain(
    tokenized_path: str | None = None,
    vocab_path: str = "",
    checkpoint_out: str = "",
    train_ds=None,
    size: str = "small",
    max_len: int = 64,
    arch: dict | None = None,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 3e-4,
    num_workers: int = 2,
    use_gpu: bool = False,
    use_fsdp: bool = False,
    loss_weighting: str = "uncertainty",
    storage_base: str | None = None,
    seed: int = 0,
) -> dict:
    """Run distributed pretraining and persist the final checkpoint.

    Training data comes from ``train_ds`` (a live Ray Dataset — e.g. tokenized
    windows materialized in the object store, no Parquet round-trip) or, when
    that's absent, from ``tokenized_path``. Ray Train re-executes the dataset
    every epoch, so a materialized in-memory dataset also avoids re-reading
    Parquet from shared storage once per epoch.

    ``arch`` is the `model:` block of a scale config (d_model / n_heads /
    n_layers / dim_ff); when omitted it is resolved from configs/<size>.yaml.

    ``storage_base`` must be a path every node can read/write (e.g.
    ``/mnt/cluster_storage/...``) — Ray Train persists checkpoints there from
    the worker nodes, so a head-node-local default like ``~/ray_results``
    breaks on multi-node clusters.
    """
    if arch is None:
        from .scale_config import load_scale

        arch = load_scale(size)["model"]

    ds = train_ds if train_ds is not None else ray.data.read_parquet(tokenized_path)
    storage_path = os.path.join(storage_base, "ray_results") if storage_base else None

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "vocab_path": vocab_path,
            "size": size,
            "max_len": max_len,
            "arch": arch,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "use_fsdp": use_fsdp,
            "loss_weighting": loss_weighting,
            "seed": seed,
        },
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        datasets={"train": ds},
        run_config=RunConfig(name="transaction_fm_pretrain", storage_path=storage_path),
    )
    result = trainer.fit()

    os.makedirs(checkpoint_out, exist_ok=True)
    # result.checkpoint.as_directory() is a context manager; copy its contents out
    # to the canonical location so downstream stages can find the weights.
    with result.checkpoint.as_directory() as ckpt_dir:
        for fn in os.listdir(ckpt_dir):
            shutil.copy(os.path.join(ckpt_dir, fn), os.path.join(checkpoint_out, fn))
    m = result.metrics
    print(
        f"[pretrain] final mlm_loss={m.get('mlm_loss', float('nan')):.4f} "
        f"macro_acc={m.get('acc_macro', float('nan')):.3f} -> {checkpoint_out}"
    )
    return m
