"""Step 4 — batch embedding extraction with Ray Data."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray  # noqa: E402

from src.embed import balanced_eval_sample, extract_embeddings  # noqa: E402
from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p)
    p.add_argument("--base-dir", default=None)
    # CLI flags override the scale's embed block (see configs/<scale>.yaml).
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    embed_cfg = load_scale(args.scale, args.scale_config)["embed"]

    ray.init(ignore_reinit_error=True)
    # Sample the balanced train + full holdout BEFORE embedding (NVIDIA NB04), then
    # embed each txn alone (max_ctx). Embeds ~1M+holdout windows, not all ~24M.
    sampled = balanced_eval_sample(paths["tokenized_eval"], balanced_train=embed_cfg["balanced_train"])
    extract_embeddings(
        ds=sampled,
        checkpoint_dir=paths["checkpoint"],
        output_path=paths["embeddings"],
        num_workers=args.num_workers or embed_cfg["num_workers"],
        use_gpu=args.use_gpu or embed_cfg["use_gpu"],
        batch_size=args.batch_size or embed_cfg["batch_size"],
        max_ctx=embed_cfg["max_ctx"],
    )


if __name__ == "__main__":
    main()
