"""Step 5 — downstream fraud classification (raw vs FM vs fusion, temporal split)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.downstream import print_summary, run_downstream  # noqa: E402
from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p)
    p.add_argument("--base-dir", default=None)
    args = p.parse_args()

    load_scale(args.scale, args.scale_config)  # validate the name early
    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    summary = run_downstream(
        embeddings_path=paths["embeddings"],
        output_dir=paths["downstream"],
        raw_path=paths["raw"],  # full ~13-feature NVIDIA-style baseline via join
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
