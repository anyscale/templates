"""Step 5 — downstream fraud classification (raw vs FM vs fusion)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.downstream import print_summary, run_downstream  # noqa: E402
from src.paths import SCALE_MAP, artifact_paths, get_demo_base_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=list(SCALE_MAP), default="small")
    p.add_argument("--base-dir", default=None)
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    summary = run_downstream(
        embeddings_path=paths["embeddings"],
        raw_path=paths["raw"],
        output_dir=paths["downstream"],
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
