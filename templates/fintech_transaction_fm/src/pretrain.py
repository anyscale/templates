"""Distributed causal-LM pretraining with Ray Train.

Pretrains the Llama decoder (src/model.py) by next-token prediction over the flat
token stream from src/flat_tokenizer.py — NVIDIA's transaction-FM recipe. The
training function is plain PyTorch; Ray Train handles worker setup, dataset
sharding, DDP/FSDP wrapping, and checkpointing — the same code runs on 1 CPU
worker (CI mini) or N GPU workers by changing only ``ScalingConfig``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile

import torch

import ray
import ray.train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from .model import build_model


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def train_func(config: dict):
    # Reduce CUDA fragmentation on long-sequence training (set before first alloc).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Same init on every rank (DDP broadcasts anyway; also makes runs reproducible).
    torch.manual_seed(config.get("seed", 0))

    vocab_path = config["vocab_path"]
    model = build_model(vocab_path, arch=config["arch"], max_len=config["max_len"])

    use_fsdp = config.get("use_fsdp", False) and torch.cuda.is_available()
    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        model = model.to(ray.train.torch.get_device())
        model = FSDP(model)
    else:
        model = ray.train.torch.prepare_model(model)
    base = _unwrap(model)

    # AdamW betas/weight-decay (transaction-FM recipe: beta2 0.95, wd 0.077).
    betas = tuple(config.get("betas", (0.9, 0.999)))
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], betas=betas, weight_decay=weight_decay
    )

    # Warmup + cosine-decay LR schedule, stepped every optimizer step (needs
    # total_steps = (sequences / workers / batch) * epochs, passed in).
    scheduler = None
    if config.get("lr_schedule") == "cosine" and int(config.get("total_steps", 0)) > 0:
        total_steps = int(config["total_steps"])
        warmup_steps = int(config.get("warmup_steps", 0))
        min_lr_ratio = float(config.get("min_lr_ratio", 0.0))

        def _lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    train_shard = ray.train.get_dataset_shard("train")
    # Token ids/mask are int32 on disk; the model needs long indices. Single dtype
    # (not a per-column dict) — the pretrain dataset is only input_ids/attention_mask.
    dtypes = torch.long

    for epoch in range(config["epochs"]):
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_shard.iter_torch_batches(
            batch_size=config["batch_size"], dtypes=dtypes
        ):
            input_ids, attn = batch["input_ids"], batch["attention_mask"]
            # Next-token labels: predict every real position; ignore pads. The
            # LlamaForCausalLM head shifts internally.
            labels = input_ids.clone()
            labels[attn == 0] = -100
            # bf16 autocast: halves activation/attention memory and speeds A10G.
            # (bf16 needs no GradScaler.) The full-vocab logits are the memory
            # driver — keep per-worker batch small (see configs) to bound them.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                loss, _ = model({"input_ids": input_ids, "attention_mask": attn, "labels": labels})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += float(loss.item())
            n_batches += 1

        avg = running / max(n_batches, 1)
        metrics = {
            "epoch": epoch,
            "lm_loss": avg,
            "perplexity": math.exp(min(avg, 20.0)),
            "lr": optimizer.param_groups[0]["lr"],
        }
        # Per-epoch progress (rank 0 only). report() ships metrics to the driver
        # but doesn't surface them live, so this makes the curve visible in logs.
        if ray.train.get_context().get_world_rank() == 0:
            print(
                f"[pretrain] epoch {epoch + 1}/{config['epochs']}  "
                f"lm_loss={avg:.3f}  ppl={metrics['perplexity']:.1f}  lr={metrics['lr']:.2e}",
                flush=True,
            )

        # Checkpoint on the last epoch (rank 0 writes the weights).
        checkpoint = None
        if epoch == config["epochs"] - 1 and ray.train.get_context().get_world_rank() == 0:
            tmp = tempfile.mkdtemp()
            torch.save(base.state_dict(), os.path.join(tmp, "model.pt"))
            shutil.copy(vocab_path, os.path.join(tmp, "vocab.json"))
            with open(os.path.join(tmp, "model_config.json"), "w") as f:
                json.dump(
                    {"size": config["size"], "max_len": config["max_len"], "arch": config["arch"]},
                    f,
                )
            checkpoint = Checkpoint.from_directory(tmp)
        ray.train.report(metrics, checkpoint=checkpoint)


def save_checkpoint(result, checkpoint_out: str) -> None:
    """Copy a ``TorchTrainer`` result's checkpoint to a canonical, all-nodes
    path so downstream stages (embedding, serving) can load the weights.

    ``result.checkpoint.as_directory()`` is a context manager pointing at
    Ray-managed storage; we copy its contents (``model.pt`` + ``vocab.json`` +
    ``model_config.json``) out to ``checkpoint_out``.
    """
    os.makedirs(checkpoint_out, exist_ok=True)
    with result.checkpoint.as_directory() as ckpt_dir:
        for fn in os.listdir(ckpt_dir):
            shutil.copy(os.path.join(ckpt_dir, fn), os.path.join(checkpoint_out, fn))


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
    objective: str = "mlm",
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999),
    lr_schedule: str | None = None,
    warmup_ratio: float = 0.0,
    min_lr_ratio: float = 0.0,
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

    # Total optimizer steps for the LR schedule: each worker steps once per batch
    # of its shard, so (windows / workers / batch) * epochs. Counted once here.
    n_windows = int(ds.count())
    steps_per_epoch = max(1, math.ceil(n_windows / max(num_workers, 1) / batch_size))
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(warmup_ratio * total_steps)

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
            "objective": objective,
            "weight_decay": weight_decay,
            "betas": tuple(betas),
            "lr_schedule": lr_schedule,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "min_lr_ratio": min_lr_ratio,
            "seed": seed,
        },
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        datasets={"train": ds},
        run_config=RunConfig(name="transaction_fm_pretrain", storage_path=storage_path),
    )
    result = trainer.fit()

    save_checkpoint(result, checkpoint_out)
    m = result.metrics
    print(
        f"[pretrain] final lm_loss={m.get('lm_loss', float('nan')):.4f} "
        f"perplexity={m.get('perplexity', float('nan')):.2f} -> {checkpoint_out}"
    )
    return m
