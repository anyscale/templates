"""
train_worker.py  —  Ray Train per-worker training loop for PI0.5.

This function runs independently on every GPU worker.  Ray Train handles
process group setup, gradient synchronization (DDP), and checkpointing
coordination; this file contains only the model and training logic.

High-level steps inside train_loop_per_worker:
  1. Load pretrained PI0.5 and freeze the backbone (train heads only).
  2. Optionally resume from an existing Ray Train checkpoint.
  3. Build the PI0.5 preprocessor with DROID normalization stats.
  4. Run the training loop with gradient accumulation.
  5. Save a checkpoint and report metrics at the end of every epoch.
"""

import json
import os
import tempfile

import torch
from ray.data.iterator import NumpyBatchCollateFn
from ray.train.torch import prepare_model


# ---------------------------------------------------------------------------
# PI0.5 helpers
# ---------------------------------------------------------------------------


def _apply_pi05_attention_mask_patch():
    """
    Monkey-patch make_att_2d_masks to tolerate pad/attention mask length
    mismatches that occasionally occur in PI0.5.

    Applied once per process (guarded by a flag on the module).
    """
    import lerobot.policies.pi05.modeling_pi05 as mp

    if getattr(mp, "_PI05_MASK_PATCH_APPLIED", False):
        return
    _orig = mp.make_att_2d_masks

    def _patched(pad_masks, att_masks):
        L = min(pad_masks.shape[-1], att_masks.shape[-1])
        return _orig(pad_masks[..., :L], att_masks[..., :L])

    mp.make_att_2d_masks = _patched
    mp._PI05_MASK_PATCH_APPLIED = True


def _freeze_backbone(policy) -> None:
    """
    Freeze all PI0.5 parameters, then unfreeze the action and time projection
    heads so only those weights are updated during fine-tuning.

    Trainable modules: action_in_proj, action_out_proj, time_mlp_in, time_mlp_out.
    """
    for p in policy.parameters():
        p.requires_grad = False

    trainable = {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}
    for name, module in policy.model.named_children():
        if name in trainable:
            for p in module.parameters():
                p.requires_grad = True


def _load_checkpoint(checkpoint, policy, optimizer) -> int:
    """
    Restore model and optimizer state from a Ray Train checkpoint.

    Returns the next epoch index so the training loop can resume seamlessly.
    """
    import ray.cloudpickle as pickle

    with checkpoint.as_directory() as d:
        with open(os.path.join(d, "state.pkl"), "rb") as f:
            state = pickle.load(f)
    policy.module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optim"])
    return state["epoch"] + 1


def _save_checkpoint(
    train, Checkpoint, policy, optimizer, epoch, step, metrics
) -> None:
    """
    Serialize model + optimizer state to a temporary directory and hand it to
    Ray Train.  Only rank 0 writes the file; all ranks report metrics.
    """
    import ray.cloudpickle as pickle

    if train.get_context().get_world_rank() == 0:
        with tempfile.TemporaryDirectory(prefix="pi05_ckpt_") as ckpt_dir:
            with open(os.path.join(ckpt_dir, "state.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "model": policy.module.state_dict(),
                        "optim": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": step,
                    },
                    f,
                )
            train.report(metrics, checkpoint=Checkpoint.from_directory(ckpt_dir))
    else:
        train.report(metrics)


def _truncate_batch_for_pi05(batch: dict, max_len: int) -> dict:
    """
    Clip all sequence and mask tensors to max_len tokens.

    PI0.5 can produce very long token sequences on high-memory hardware.
    Truncating keeps GPU memory predictable on smaller GPUs without
    changing the model architecture.
    """
    if not max_len:
        return batch
    mask_keys = (
        "tokens",
        "input_ids",
        "masks",
        "attention_mask",
        "pad_masks",
        "att_masks",
        "img_masks",
        "image_masks",
    )
    for k in mask_keys:
        if k in batch and hasattr(batch[k], "ndim") and batch[k].ndim >= 2:
            batch[k] = batch[k][..., :max_len]
    return batch


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


class _Collate(NumpyBatchCollateFn):
    """Convert a numpy batch dict into float32 tensors on the target device."""

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, batch: dict) -> dict:
        task = list(batch.pop("task"))
        result = {
            k: torch.tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }
        result["task"] = task
        return result


# ---------------------------------------------------------------------------
# Ray Train per-worker loop
# ---------------------------------------------------------------------------


def train_loop_per_worker(config: dict):
    """
    Entry point called by Ray Train on every GPU worker.

    config keys (set in the notebook as train_loop_config):
      stats_path  path to the DROID normalization stats JSON
      num_epochs  total training epochs
      batch_size  per-worker micro-batch size
      grad_accum  gradient accumulation steps before an optimizer step
      lr          AdamW learning rate
      max_len     maximum token sequence length (0 = no limit)
    """
    # Lazy imports keep lerobot off the driver process; workers have it
    # available via the RuntimeEnv working_dir bundle.
    from ray import train
    from ray.train import Checkpoint
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device("cuda")

    # --- 1. Model setup -------------------------------------------------------

    _apply_pi05_attention_mask_patch()

    policy = PI05Policy.from_pretrained(
        "lerobot/pi05_base",
        device="cuda",
        dtype=torch.float16,
        train_expert_only=True,
    )
    _freeze_backbone(policy)
    policy = prepare_model(policy)  # wraps in DDP, moves to device

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=config.get("lr", 1e-4),
    )

    # --- 2. Resume from checkpoint (if one exists) ----------------------------

    checkpoint = train.get_checkpoint()
    start_epoch = _load_checkpoint(checkpoint, policy, optimizer) if checkpoint else 0

    # --- 3. Preprocessor with DROID normalization stats -----------------------

    with open(config["stats_path"]) as f:
        stats = json.load(f)

    preprocessor, _ = make_pre_post_processors(
        policy.module.config,
        pretrained_path="lerobot/pi05_base",
        dataset_stats=stats,
    )

    # --- 4. Training loop -----------------------------------------------------

    shard = train.get_dataset_shard("train")
    batch_size = int(config.get("batch_size", 1))
    max_len = int(config.get("max_len", 512))
    grad_accum = int(config.get("grad_accum", 1))
    num_epochs = int(config.get("num_epochs", 1))

    step = 0
    for epoch in range(start_epoch, num_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss_sum, loss_count = 0.0, 0

        for batch in shard.iter_torch_batches(
            batch_size=batch_size,
            collate_fn=_Collate(device),
        ):
            batch = preprocessor(batch)
            batch = _truncate_batch_for_pi05(batch, max_len)
            batch.pop("task", None)  # consumed by preprocessor
            batch.pop("task_index", None)

            with torch.autocast("cuda", torch.float16):
                out = policy(batch)
                loss = out.loss if hasattr(out, "loss") else out[0]

            (loss / grad_accum).backward()
            step += 1
            loss_sum += float(loss.detach())
            loss_count += 1

            if train.get_context().get_world_rank() == 0:
                print(f"epoch={epoch}  step={step}  loss={loss.item():.4f}")

            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # --- 5. Checkpoint + metrics ------------------------------------------

        metrics = {"epoch": epoch, "steps": step, "loss": loss_sum / max(loss_count, 1)}
        _save_checkpoint(train, Checkpoint, policy, optimizer, epoch, step, metrics)
