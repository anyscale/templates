"""
Training Utilities for VLA Fine-Tuning
=======================================

Helper functions used by vla.py. These are separated out to keep the main
script focused on the Ray Data + Ray Train pipeline, while housing the
model-specific plumbing (freezing layers, checkpointing, collation, etc.)
in one place.
"""

import os
import tempfile
import warnings

import numpy as np
import torch
from ray.data.iterator import NumpyBatchCollateFn


# ============================================================================
# PI0.5 Attention Mask Patch
# ============================================================================
#
# The PI0.5 model's make_att_2d_masks function assumes pad_masks and
# att_masks always have matching sequence lengths. In practice, the
# preprocessor can produce mismatched lengths (e.g. after truncation or
# when the tokenizer pads differently). This monkey-patch detects the
# mismatch and truncates both masks to the shorter length, preventing a
# hard crash during training.
# ============================================================================


def apply_pi05_attention_mask_patch():
    """Monkey-patch make_att_2d_masks to tolerate pad/attention mask length mismatches."""
    import lerobot.policies.pi05.modeling_pi05 as mp

    if getattr(mp, "_PI05_MASK_PATCH_APPLIED", False):
        return
    _orig = mp.make_att_2d_masks

    def _patched(pad_masks, att_masks):
        pad_len, att_len = pad_masks.shape[-1], att_masks.shape[-1]
        if pad_len != att_len:
            warnings.warn(
                f"PI0.5 mask length mismatch: pad_masks={pad_len}, att_masks={att_len}; "
                f"truncating to {min(pad_len, att_len)}"
            )
            L = min(pad_len, att_len)
            return _orig(pad_masks[..., :L], att_masks[..., :L])
        return _orig(pad_masks, att_masks)

    mp.make_att_2d_masks = _patched
    mp._PI05_MASK_PATCH_APPLIED = True


# ============================================================================
# Dataset Helpers
# ============================================================================


def extract_stats(source):
    """Extract normalization stats (mean/std) from a LeRobot datasource.

    Returns a dict with keys like "action" and "observation.state", each
    containing "mean" and "std" arrays from the dataset metadata.
    """
    raw = source.meta.stats
    stats = {}
    for key in ("action", "observation.state"):
        if key in raw:
            stats[key] = {"mean": raw[key]["mean"], "std": raw[key]["std"]}
    return stats


def renamed_image_keys(source, camera_rename):
    """Get camera column names after applying the rename map."""
    return [camera_rename.get(k, k) for k in source.meta.video_keys]


# ============================================================================
# Model Loading
# ============================================================================
#
# PI0.5 is published at HuggingFace `lerobot/pi05_base`. Loading from there
# normally requires HF_TOKEN because PI0.5 references the gated PaliGemma
# backbone — but in practice `PI05Policy.from_pretrained` only fetches
# `model.safetensors` (PaliGemma weights are merged in), so a public S3
# mirror of the same files works with no token at all.
# ============================================================================

PI05_BASE_DEFAULT = "s3://anyscale-public-robotics-datasets/lerobot/lerobot/pi05_base/"

_pi05_local_cache: dict[str, str] = {}


def _mirror_s3_dir(s3_uri: str, local_dir: str) -> str:
    """Anonymously mirror a public S3 'directory' to local disk. Idempotent."""
    from urllib.parse import urlparse

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    parsed = urlparse(s3_uri)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            relpath = obj["Key"][len(prefix):].lstrip("/")
            if not relpath or any(p.startswith(".") for p in relpath.split("/")):
                continue  # skip dotfiles / hidden dirs (e.g. .cache/huggingface)
            dest = os.path.join(local_dir, relpath)
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            if os.path.exists(dest) and os.path.getsize(dest) == obj["Size"]:
                continue
            s3.download_file(bucket, obj["Key"], dest)
    return local_dir


def resolve_pi05_path(path: str = PI05_BASE_DEFAULT) -> str:
    """Resolve an `s3://` URI to a local mirror; pass through anything else.

    Memoized per-process so workers that call this for both the policy
    weights and the preprocessor share one download.
    """
    if not path.startswith("s3://"):
        return path
    if path in _pi05_local_cache:
        return _pi05_local_cache[path]
    local_dir = _mirror_s3_dir(path, "/tmp/pi05_base")
    _pi05_local_cache[path] = local_dir
    return local_dir


def load_pi05_policy(pretrained_path: str = PI05_BASE_DEFAULT):
    """Load PI0.5, apply the attention mask patch, and freeze the backbone.

    Returns the policy with only the action/time projection heads unfrozen
    (action_in_proj, action_out_proj, time_mlp_in, time_mlp_out). The large
    pretrained vision-language backbone stays frozen, dramatically reducing
    memory and compute while still adapting the model to new tasks.

    Also applies the attention mask monkey-patch (see above) so training
    doesn't crash on sequence-length mismatches.
    """
    from lerobot.policies.pi05 import PI05Policy

    apply_pi05_attention_mask_patch()

    local_path = resolve_pi05_path(pretrained_path)
    policy = PI05Policy.from_pretrained(
        local_path, device="cuda", dtype=torch.float16, train_expert_only=True,
    )

    # Freeze everything, then unfreeze the small trainable heads.
    # train_expert_only=True sets a config flag but doesn't actually toggle
    # requires_grad -- we do that manually here.
    for p in policy.parameters():
        p.requires_grad = False
    for name, module in policy.model.named_children():
        if name in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}:
            for p in module.parameters():
                p.requires_grad = True

    return policy


# ============================================================================
# Checkpoint Save / Load
# ============================================================================


def load_checkpoint(checkpoint, policy, optimizer, scaler) -> tuple[int, int]:
    """Restore model/optimizer/scaler state from a Ray Train checkpoint.

    Returns (start_epoch, start_step).
    """
    import ray.cloudpickle as pickle

    with checkpoint.as_directory() as d:
        with open(os.path.join(d, "state.pkl"), "rb") as f:
            state = pickle.load(f)
    policy.module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optim"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state["epoch"] + 1, state.get("step", 0)


def make_checkpoint(policy, optimizer, scaler, epoch, step):
    """Serialize model + optimizer + scaler state into a Ray Train Checkpoint.

    This captures everything needed to resume training: model weights,
    optimizer state, gradient scaler state, and the current epoch/step.
    The checkpoint is written to a temp directory and returned as a
    ray.train.Checkpoint for use with ray.train.report().
    """
    import ray.cloudpickle as pickle
    import ray.train

    ckpt_dir = tempfile.mkdtemp(prefix="pi05_ckpt_")
    with open(os.path.join(ckpt_dir, "state.pkl"), "wb") as f:
        pickle.dump(
            {"model": policy.module.state_dict(), "optim": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "epoch": epoch, "step": step},
            f,
        )
    return ray.train.Checkpoint.from_directory(ckpt_dir)


# ============================================================================
# Sequence Truncation
# ============================================================================


def truncate_batch(batch: dict, max_len: int) -> dict:
    """Clip sequence and mask tensors to max_len tokens.

    PI0.5 can produce very long token sequences depending on the number of
    cameras and action horizon. Truncating caps GPU memory usage at the cost
    of losing some context. Set max_len=0 to disable.
    """
    if not max_len:
        return batch
    for k in ("tokens", "input_ids", "masks", "attention_mask",
              "pad_masks", "att_masks", "img_masks", "image_masks"):
        if k in batch and hasattr(batch[k], "ndim") and batch[k].ndim >= 2:
            batch[k] = batch[k][..., :max_len]
    return batch


# ============================================================================
# Collation: numpy dicts -> torch tensors on GPU
# ============================================================================


class NumpyToTorchCollate(NumpyBatchCollateFn):
    """Convert a numpy batch dict into tensors on the target device.

    Ray Data delivers batches as numpy arrays. This collate function moves
    them to GPU as torch tensors, preserving dtype semantics: integer arrays
    become torch.long, booleans become torch.bool, and everything else
    becomes torch.float32.

    The ``task`` column is kept as a Python list of strings (the model's
    language conditioning input).
    """

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, batch: dict) -> dict:
        task = list(batch.pop("task"))
        result = {}
        for k, v in batch.items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.integer):
                result[k] = torch.tensor(arr, dtype=torch.long, device=self.device)
            elif np.issubdtype(arr.dtype, np.bool_):
                result[k] = torch.tensor(arr, dtype=torch.bool, device=self.device)
            else:
                result[k] = torch.tensor(arr, dtype=torch.float32, device=self.device)
        result["task"] = task
        return result


# ============================================================================
# Training Step Helpers
# ============================================================================
#
# These encapsulate the standard PyTorch forward/backward/optimizer mechanics
# so that vla.py's training loop can focus entirely on the Ray-specific parts:
# streaming data, distributed coordination, and checkpointing.
# ============================================================================


def train_step(policy, batch, preprocessor, max_len, grad_accum, scaler):
    """Run one forward + backward pass. Returns the scalar loss value.

    This is vanilla PyTorch training: autocast for mixed precision, scale the
    loss for gradient accumulation, and call backward. Nothing Ray-specific
    happens here -- it's the same code you'd write for single-GPU training.
    """
    batch = preprocessor(batch)
    batch = truncate_batch(batch, max_len)
    batch.pop("task", None)
    batch.pop("task_index", None)

    with torch.autocast("cuda", torch.float16):
        out = policy(batch)
        loss = out.loss if hasattr(out, "loss") else out[0]

    scaler.scale(loss / grad_accum).backward()
    return float(loss.detach())


def optimizer_step(policy, optimizer, scaler, scheduler):
    """Unscale gradients, clip, step the optimizer, and update the LR schedule.

    Standard PyTorch mixed-precision optimizer step with gradient clipping.
    Called every ``grad_accum`` micro-batches.
    """
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad], max_norm=1.0,
    )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


def build_lr_scheduler(optimizer, config, num_workers, last_step):
    """Create a linear-warmup + cosine-decay LR scheduler.

    Computes total optimizer steps from the config (total_rows, batch_size,
    grad_accum, num_epochs) and num_workers, then builds a LambdaLR that
    linearly warms up and cosine-decays to 0.
    """
    import math

    batch_size = int(config.get("batch_size", 1))
    grad_accum = int(config.get("grad_accum", 1))
    num_epochs = int(config.get("num_epochs", 1))
    total_rows = int(config.get("total_rows", 10000))
    warmup_frac = float(config.get("warmup_frac", 0.1))

    rows_per_worker = total_rows // num_workers
    total_steps = max(rows_per_worker // (batch_size * grad_accum), 1) * num_epochs
    warmup_steps = int(total_steps * warmup_frac)

    def lr_lambda(s):
        if s < warmup_steps:
            return s / max(warmup_steps, 1)
        progress = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=last_step - 1 if last_step > 0 else -1,
    )
