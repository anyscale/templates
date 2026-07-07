"""Distributed masked-feature-modeling pretraining with Ray Train.

The training function is plain PyTorch. Ray Train handles the distributed parts:
worker setup, dataset sharding, DDP/FSDP wrapping, checkpointing, and fault
tolerance — the same code runs on 1 CPU worker (CI smoke) or N GPU workers
(the real distributed story) by changing only ``ScalingConfig``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from collections import defaultdict

import torch

import ray
import ray.train
from ray.train import Checkpoint, CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from .model import build_model, mask_batch
from .paths import tensorboard_root


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _contrastive_views(batch: dict):
    """Split each (left-padded) window into two disjoint temporal halves.

    Two sub-sequences of the same card = a CoLES positive pair, built from the
    window itself so there is NO dataset constraint (works at any seq_len and
    for short cards; windows with <4 txns are excluded via pair_valid). Views
    are half-length tensors (S/2), so the two extra encodes cost ~0.5x the
    main forward, not 2x. Static fields are PAD'd out — matching halves via
    the user/card id embedding would be a trivial shortcut.
    """
    am = batch["attention_mask"]
    B, S = am.shape
    V = S // 2
    L = am.sum(dim=1).long()  # valid txns per window
    h1 = L // 2
    j = torch.arange(V, device=am.device).unsqueeze(0)  # (1, V)
    start = (S - L).unsqueeze(1)  # first valid position (left-padded)

    def build(h, base):
        off = j - (V - h).unsqueeze(1)  # (B, V) target-aligned offsets
        m = off >= 0
        src = (base + off).clamp(0, S - 1)
        view = {}
        for k, t in batch.items():
            if k.startswith("s_"):
                view[k] = torch.zeros_like(t)  # statics -> PAD
            elif t.dim() == 2 and t.shape[1] == S and not k.startswith("y_"):
                g = t.gather(1, src)
                view[k] = torch.where(m, g, torch.zeros_like(g))
        view["attention_mask"] = m.to(am.dtype)
        return view

    v1 = build(h1, start)
    v2 = build(L - h1, start + h1.unsqueeze(1))
    return v1, v2, (L >= 4)


def train_func(config: dict):
    # Same init on every rank (DDP would broadcast anyway; this also makes
    # *runs* reproducible for A/B comparisons), then per-rank reseed below so
    # MLM masking differs across workers deterministically.
    torch.manual_seed(config.get("seed", 0))

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # TF32 matmuls for the FP32 ops that remain under autocast, cuDNN
        # autotune for the fixed-shape encoder — free throughput on Ampere+.
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    # Mixed precision: bf16 where the GPU supports it (Ampere+), fp16 + loss
    # scaling on older GPUs (T4/V100). CPU (CI smoke) stays fp32 — autocast and
    # the scaler are both disabled there.
    amp_dtype = None
    if use_cuda:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    vocab_path = config["vocab_path"]
    with open(vocab_path) as f:
        vocab = json.load(f)
    dynamic_fields = vocab["dynamic_fields"]
    signal_fields = vocab.get("signal_fields", [])
    weighting = config.get("loss_weighting", "uncertainty")

    # Per-column dtypes: tokens are int64, the soft-bin amount weight is float32.
    dtypes = {"d_amount_frac": torch.float32} if vocab.get("amount_mode") == "soft" else None

    model = build_model(
        vocab_path,
        arch=config["arch"],
        max_len=config["max_len"],
        infonce_negatives=config.get("infonce_negatives", 1024),
    )

    use_fsdp = config.get("use_fsdp", False) and torch.cuda.is_available()
    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        model = model.to(ray.train.torch.get_device())
        model = FSDP(model)
    else:
        model = ray.train.torch.prepare_model(model)
    base = _unwrap(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # Linear warmup + cosine decay: constant LR is unstable early (especially at
    # the large-batch scales, where lr is scaled up) and oscillates late.
    world = ray.train.get_context().get_world_size()
    steps_per_epoch = max(1, math.ceil(config["n_rows"] / (world * config["batch_size"])))
    total_steps = steps_per_epoch * config["epochs"]
    warmup_steps = max(1, int(0.05 * total_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # InfoNCE warm-up: ramp the high-cardinality merchant objective in over the
    # first ``infonce_warmup_frac`` of training so the fraud-relevant heads
    # shape the representation first, instead of the big contrastive head
    # dominating the gradient budget from step 0 (loss-budget dilution).
    infonce_warmup_steps = int(config.get("infonce_warmup_frac", 0.0) * total_steps)

    train_shard = ray.train.get_dataset_shard("train")
    mask_prob = config.get("mask_prob", 0.15)
    seq_cl_weight = config.get("seq_cl_weight", 0.0)
    torch.manual_seed(config.get("seed", 0) + ray.train.get_context().get_world_rank())

    # Resume after a worker failure: FailureConfig restarts land back here with
    # the last reported checkpoint attached.
    # TensorBoard (rank 0): per-step curves are the debugging view the epoch
    # metrics can't give — e.g. each head's loss trajectory while the InfoNCE
    # ramp is in flight. Event files land on shared cluster storage; view with
    # `tensorboard --logdir <storage_base>/tensorboard`. Telemetry must never
    # kill training, so a missing tensorboard package just warns.
    writer = None
    if config.get("tensorboard_dir") and ray.train.get_context().get_world_rank() == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(config["tensorboard_dir"])
            print(f"[pretrain] tensorboard -> {config['tensorboard_dir']}")
            # Self-document the run: the ENTIRE train_loop_config and the
            # ENTIRE scale YAML, unfiltered, as one yaml block in the TEXT tab.
            import yaml

            dump = {
                "train_loop_config": dict(config),
                "world_size": ray.train.get_context().get_world_size(),
            }
            try:  # full configs/<scale>.yaml (may be absent for a custom path)
                from .scale_config import load_scale

                dump["scale_config"] = load_scale(config["size"])
            except BaseException:  # load_scale raises SystemExit on unknown names
                pass
            writer.add_text(
                "hparams",
                "```yaml\n" + yaml.safe_dump(dump, sort_keys=False) + "```",
                0,
            )
        except ImportError:
            print("[pretrain] tensorboard not installed — skipping metric logging")

    start_epoch, global_step = 0, 0
    ckpt = ray.train.get_checkpoint()
    if ckpt is not None:
        with ckpt.as_directory() as d:
            dev = ray.train.torch.get_device()
            base.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev))
            state = torch.load(os.path.join(d, "train_state.pt"), map_location=dev)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start_epoch = state["epoch"] + 1
            global_step = state.get("global_step", start_epoch * steps_per_epoch)
        print(f"[pretrain] resumed from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        running, n_batches = 0.0, 0
        ce_sum, acc_sum, tot_n = defaultdict(float), defaultdict(float), 0
        for batch in train_shard.iter_torch_batches(
            batch_size=config["batch_size"],
            dtypes=dtypes,
            # Overlap batch prep with compute so the GPU never waits on the shard.
            prefetch_batches=2,
            # Per-epoch order variation. The dataset was globally shuffled once
            # upstream (it comes out of the tokenizer grouped by card); this
            # buffer re-randomizes locally each epoch without an all-to-all.
            # Seeded per epoch: order varies across epochs, not across runs.
            local_shuffle_buffer_size=max(8 * config["batch_size"], 1024),
            local_shuffle_seed=config.get("seed", 0) + epoch,
        ):
            corrupted, targets, masked = mask_batch(batch, dynamic_fields, mask_prob)
            n = int(masked.sum())
            if n == 0:
                continue
            infonce_scale = (
                1.0
                if infonce_warmup_steps == 0
                else min(1.0, global_step / infonce_warmup_steps)
            )
            # CoLES-style views from the UNCORRUPTED batch (clean inputs; the
            # contrastive term shares the MLM warm-up ramp).
            seq_views = (
                _contrastive_views(batch) if seq_cl_weight > 0.0 else None
            )
            optimizer.zero_grad()
            # Heads + loss run inside forward so DDP all-reduces every param.
            with torch.autocast(
                "cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None
            ):
                loss, stats = model(
                    corrupted,
                    targets=targets,
                    masked=masked,
                    weighting=weighting,
                    infonce_scale=infonce_scale,
                    seq_views=seq_views,
                    seq_cl_scale=seq_cl_weight * infonce_scale,
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            global_step += 1
            running += float(loss.item())
            n_batches += 1
            tot_n += n
            for f, d in stats.items():
                ce_sum[f] += d["ce"] * n   # weight per-field means by #masked
                acc_sum[f] += d["acc"] * n
            if writer is not None:
                # Rank 0's local values — a per-step debugging view, not a
                # world-averaged metric (the epoch metrics below are averaged).
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/infonce_scale", infonce_scale, global_step)
                for f, d in stats.items():
                    writer.add_scalar(f"field_ce/{f}", d["ce"], global_step)

        # The weighted total drifts as the log-variances learn — watch the
        # per-field accuracy and perplexity instead. Perplexity vs. the field's
        # vocab size tells you whether it learned structure (ppl << vocab = good).
        metrics = {
            "epoch": epoch,
            "mlm_loss": running / max(n_batches, 1),
            "lr": scheduler.get_last_lr()[0],
            "infonce_scale": (
                1.0 if infonce_warmup_steps == 0
                else min(1.0, global_step / infonce_warmup_steps)
            ),
        }
        macro_acc = 0.0
        for f in dynamic_fields + signal_fields:
            mean_ce = ce_sum[f] / max(tot_n, 1)
            acc = acc_sum[f] / max(tot_n, 1)
            metrics[f"acc_{f}"] = acc
            metrics[f"ppl_{f}"] = math.exp(min(mean_ce, 20.0))
            if f in dynamic_fields:
                macro_acc += acc
        metrics["acc_macro"] = macro_acc / len(dynamic_fields)
        if "seq_cl" in ce_sum:  # sequence-level contrastive health
            metrics["seq_cl_loss"] = ce_sum["seq_cl"] / max(tot_n, 1)
            metrics["acc_seq_cl"] = acc_sum["seq_cl"] / max(tot_n, 1)
        if writer is not None:
            for k, v in metrics.items():
                if k != "epoch":
                    writer.add_scalar(f"epoch/{k}", v, epoch)
            writer.flush()

        # Checkpoint every epoch (rank 0 writes) so a failure mid-run resumes
        # instead of restarting from scratch. model.pt stays weights-only — the
        # downstream consumers load it directly; resume state lives alongside
        # in train_state.pt.
        checkpoint = None
        if ray.train.get_context().get_world_rank() == 0:
            tmp = tempfile.mkdtemp()
            torch.save(base.state_dict(), os.path.join(tmp, "model.pt"))
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                },
                os.path.join(tmp, "train_state.pt"),
            )
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

    if writer is not None:
        writer.close()


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
    infonce_negatives: int = 1024,
    infonce_warmup_frac: float = 0.0,
    seq_cl_weight: float = 0.0,
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
    n_rows = ds.count()  # sizes the LR schedule (cheap: metadata / materialized)
    storage_path = os.path.join(storage_base, "ray_results") if storage_base else None
    # Unique per invocation: reusing a name resumes that run's latest
    # checkpoint, which would silently skip training on a re-run. The
    # in-run failure restore (FailureConfig) is unaffected by the name.
    # Key hyperparams go in the name so TensorBoard's run list is
    # self-describing (the full table is in the run's Text tab).
    lr_tag = f"{lr:.0e}".replace("e-0", "e-")
    run_name = (
        f"fm_{size}_seq{max_len}_b{batch_size}x{num_workers}_lr{lr_tag}"
        f"_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    tb_root = tensorboard_root(storage_base)
    tensorboard_dir = os.path.join(tb_root, run_name) if tb_root else None

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
            "infonce_negatives": infonce_negatives,
            "infonce_warmup_frac": infonce_warmup_frac,
            "seq_cl_weight": seq_cl_weight,
            "seed": seed,
            "n_rows": n_rows,
            "tensorboard_dir": tensorboard_dir,
        },
        # GPU workers request {"GPU": 1} and zero CPUs by default, so they
        # schedule fine on GPU nodes that advertise CPU: 0 (see job yamls —
        # that override keeps CPU-only tasks from scaling up GPU nodes).
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        datasets={"train": ds},
        run_config=RunConfig(
            name=run_name,
            storage_path=storage_path,
            # Restart in place (from the latest epoch checkpoint) on worker
            # loss — required for spot GPU nodes; the job-level retry would
            # otherwise redo the whole pipeline.
            failure_config=FailureConfig(max_failures=3),
            # Keep every epoch's checkpoint: scripts/probe_by_epoch.py scores
            # fraud/reco per epoch to expose the merchant-vs-fraud training
            # trade. Cost is bounded (~100MB/epoch at small, ~400MB at full)
            # and the storage dies with the cluster anyway.
            checkpoint_config=CheckpointConfig(num_to_keep=None),
        ),
    )
    result = trainer.fit()

    os.makedirs(checkpoint_out, exist_ok=True)
    # result.checkpoint.as_directory() is a context manager; copy its contents out
    # to the canonical location so downstream stages can find the weights.
    # train_state.pt is resume-only — the canonical dir stays weights + config.
    with result.checkpoint.as_directory() as ckpt_dir:
        for fn in os.listdir(ckpt_dir):
            if fn == "train_state.pt":
                continue
            shutil.copy(os.path.join(ckpt_dir, fn), os.path.join(checkpoint_out, fn))
    m = result.metrics
    print(
        f"[pretrain] final mlm_loss={m.get('mlm_loss', float('nan')):.4f} "
        f"macro_acc={m.get('acc_macro', float('nan')):.3f} -> {checkpoint_out}"
    )
    return m
