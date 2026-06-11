"""Step 3 — distributed masked-feature-modeling pretraining with Ray Train."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray  # noqa: E402

from src.paths import SCALE_MAP, artifact_paths, get_demo_base_dir  # noqa: E402
from src.pretrain import pretrain  # noqa: E402
from src.tokenizer import SEQ_LEN_BY_SCALE  # noqa: E402

# Per-scale training presets (kept tiny for `smoke` so CI runs on CPU).
# The model is small enough to fit one GPU at every scale, so we use DDP
# (data-parallel) throughout — FSDP would only help a model too big for one GPU.
# Set use_fsdp=True via pretrain() if you scale the model up substantially.
# Batch sizes sized to actually load the GPUs (the model is tiny; T4s were
# mostly idle at 128) with lr scaled alongside; epoch counts chosen because
# loss was still dropping at the old cutoffs.
TRAIN_PRESETS = {
    "smoke": dict(epochs=2, batch_size=64, num_workers=1, use_gpu=False, use_fsdp=False),
    "small": dict(epochs=15, batch_size=512, lr=8e-4, num_workers=2, use_gpu=True, use_fsdp=False),
    # The real thing: ~29M params (NVIDIA-parity) at seq 512. batch 64/worker
    # is the T4-safe ceiling — the B x heads x S x S attention buffers and the
    # 2000-way merchant MLM head logits dominate memory at this seq_len.
    # Epochs are cheap: non-overlapping 512-txn windows = ~38k/epoch.
    "full": dict(epochs=20, batch_size=64, lr=4e-4, num_workers=4, use_gpu=True, use_fsdp=False),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=list(SCALE_MAP), default="small")
    p.add_argument("--base-dir", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--use-gpu", action="store_true")
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    preset = dict(TRAIN_PRESETS[args.scale])
    if args.num_workers is not None:
        preset["num_workers"] = args.num_workers
    if args.use_gpu:
        preset["use_gpu"] = True

    ray.init(ignore_reinit_error=True)
    pretrain(
        tokenized_path=paths["tokenized_pretrain"],
        vocab_path=paths["vocab"],
        checkpoint_out=paths["checkpoint"],
        size=args.scale,
        max_len=SEQ_LEN_BY_SCALE[args.scale],
        storage_base=base,
        **preset,
    )


if __name__ == "__main__":
    main()
