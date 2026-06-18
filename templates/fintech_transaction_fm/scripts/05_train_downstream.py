"""Step 5 — downstream fraud classification (raw vs FM vs fusion, temporal split)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.downstream import print_summary, run_downstream  # noqa: E402
from src.paths import resolve_artifact_paths  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p)
    p.add_argument("--base-dir", default=None)
    args = p.parse_args()

    load_scale(args.scale, args.scale_config)  # validate the name early
    paths = resolve_artifact_paths(args.scale, args.base_dir)
    summary = run_downstream(
        embeddings_path=paths["embeddings"],
        output_dir=paths["downstream"],
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
