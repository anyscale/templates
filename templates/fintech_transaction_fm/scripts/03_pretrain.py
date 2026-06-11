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
TRAIN_PRESETS = {
    "smoke": dict(epochs=2, batch_size=64, num_workers=1, use_gpu=False, use_fsdp=False),
    "small": dict(epochs=5, batch_size=128, num_workers=2, use_gpu=True, use_fsdp=False),
    "medium": dict(epochs=8, batch_size=256, num_workers=4, use_gpu=True, use_fsdp=False),
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
        tokenized_path=paths["tokenized"],
        vocab_path=paths["vocab"],
        checkpoint_out=paths["checkpoint"],
        size=args.scale,
        max_len=SEQ_LEN_BY_SCALE[args.scale],
        storage_base=base,
        **preset,
    )


if __name__ == "__main__":
    main()
