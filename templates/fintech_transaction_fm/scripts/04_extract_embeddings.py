"""Step 4 — batch embedding extraction with Ray Data."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray  # noqa: E402

from src.embed import extract_embeddings  # noqa: E402
from src.paths import SCALE_MAP, artifact_paths, get_demo_base_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=list(SCALE_MAP), default="small")
    p.add_argument("--base-dir", default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-gpu", action="store_true")
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)

    ray.init(ignore_reinit_error=True)
    extract_embeddings(
        tokenized_path=paths["tokenized"],
        checkpoint_dir=paths["checkpoint"],
        output_path=paths["embeddings"],
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    main()
