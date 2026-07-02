"""Step 6 — next-merchant recommendation eval (HR@K / NDCG@K).

The second downstream consumer of the same pretrained backbone. Requires the
learned merchant vocab + InfoNCE head (tokenize.merchant_vocab: learned).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray  # noqa: E402

from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.recommend import print_summary, run_recommend  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p)
    p.add_argument("--base-dir", default=None)
    p.add_argument("--use-gpu", action="store_true")
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    e = load_scale(args.scale, args.scale_config)["embed"]

    ray.init(ignore_reinit_error=True)
    summary = run_recommend(
        checkpoint_dir=paths["checkpoint"],
        output_dir=paths["downstream"],
        tokenized_path=paths["tokenized_eval"],
        num_workers=e["num_workers"],
        use_gpu=args.use_gpu or e["use_gpu"],
        batch_size=e["batch_size"],
        gpus_per_worker=e.get("gpus_per_worker"),
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
