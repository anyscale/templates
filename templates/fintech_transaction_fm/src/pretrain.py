"""Distributed masked-feature-modeling pretraining with Ray Train.

The training function is plain PyTorch. Ray Train handles the distributed parts:
worker setup, dataset sharding, DDP/FSDP wrapping, checkpointing, and fault
tolerance — the same code runs on 1 CPU worker (CI smoke) or N GPU workers
(the real distributed story) by changing only ``ScalingConfig``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import torch

import ray
import ray.train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from .model import build_model, mask_batch


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def train_func(config: dict):
    vocab_path = config["vocab_path"]
    with open(vocab_path) as f:
        vocab = json.load(f)
    dynamic_fields = vocab["dynamic_fields"]
    weighting = config.get("loss_weighting", "uncertainty")

    # Per-column dtypes: tokens are int64, the soft-bin amount weight is float32.
    dtypes = {"d_amount_frac": torch.float32} if vocab.get("amount_mode") == "soft" else None

    model = build_model(vocab_path, size=config["size"], max_len=config["max_len"])

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

    for epoch in range(config["epochs"]):
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_shard.iter_torch_batches(
            batch_size=config["batch_size"], dtypes=dtypes
        ):
            corrupted, targets, masked = mask_batch(batch, dynamic_fields, mask_prob)
            if masked.sum() == 0:
                continue
            # Heads + loss run inside forward so DDP all-reduces every param.
            loss, _ = model(corrupted, targets=targets, masked=masked, weighting=weighting)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1

        avg = running / max(n_batches, 1)
        metrics = {"epoch": epoch, "mlm_loss": avg}

        # Checkpoint on the last epoch (rank 0 writes the weights).
        checkpoint = None
        if epoch == config["epochs"] - 1 and ray.train.get_context().get_world_rank() == 0:
            tmp = tempfile.mkdtemp()
            torch.save(base.state_dict(), os.path.join(tmp, "model.pt"))
            shutil.copy(vocab_path, os.path.join(tmp, "vocab.json"))
            with open(os.path.join(tmp, "model_config.json"), "w") as f:
                json.dump({"size": config["size"], "max_len": config["max_len"]}, f)
            checkpoint = Checkpoint.from_directory(tmp)
        ray.train.report(metrics, checkpoint=checkpoint)


def pretrain(
    tokenized_path: str,
    vocab_path: str,
    checkpoint_out: str,
    size: str = "small",
    max_len: int = 64,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 3e-4,
    num_workers: int = 2,
    use_gpu: bool = False,
    use_fsdp: bool = False,
    loss_weighting: str = "uncertainty",
) -> dict:
    """Run distributed pretraining and persist the final checkpoint."""
    ds = ray.data.read_parquet(tokenized_path)

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "vocab_path": vocab_path,
            "size": size,
            "max_len": max_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "use_fsdp": use_fsdp,
            "loss_weighting": loss_weighting,
        },
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        datasets={"train": ds},
        run_config=RunConfig(name="transaction_fm_pretrain"),
    )
    result = trainer.fit()

    os.makedirs(checkpoint_out, exist_ok=True)
    # result.checkpoint.as_directory() is a context manager; copy its contents out
    # to the canonical location so downstream stages can find the weights.
    with result.checkpoint.as_directory() as ckpt_dir:
        for fn in os.listdir(ckpt_dir):
            shutil.copy(os.path.join(ckpt_dir, fn), os.path.join(checkpoint_out, fn))
    print(f"[pretrain] final mlm_loss={result.metrics.get('mlm_loss'):.4f} -> {checkpoint_out}")
    return result.metrics
