"""Step 3 — distributed masked-feature-modeling pretraining with Ray Train."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray  # noqa: E402

from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.pretrain import pretrain  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p)
    p.add_argument("--base-dir", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--run-name", default=None,
                   help="resume an existing ray_results run by name (pair "
                        "with a --scale-config that raises epochs)")
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    # Training knobs + architecture (see configs/<scale>.yaml).
    cfg = load_scale(args.scale, args.scale_config)
    preset = dict(cfg["pretrain"])
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
        max_len=cfg["tokenize"]["seq_len"],
        arch=cfg["model"],
        storage_base=base,
        run_name=args.run_name,
        **preset,
    )


if __name__ == "__main__":
    main()
