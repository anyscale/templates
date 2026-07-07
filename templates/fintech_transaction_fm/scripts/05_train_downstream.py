"""Step 5 — downstream fraud detection, NVIDIA blueprint protocol.

Trains their notebook-05 XGBoost models on the benchmark rows stage 01
sampled (1M balanced train + 100k stratified val/test): the 13-raw-feature
baseline by default; add ``--with-embeddings`` to also train the
embeddings-only and combined (raw + FM embedding) models.

Synthetic data has no benchmark file — falls back to the legacy raw/fm/fusion
comparison on the heuristic eval windows.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p)
    p.add_argument("--base-dir", default=None)
    p.add_argument("--with-embeddings", action="store_true",
                   help="also train the embeddings-only and combined models "
                        "(requires stages 03/04)")
    p.add_argument("--device", default="cpu", help="XGBoost device (cpu | cuda)")
    # Overrides for scoring alternate embedding variants side-by-side.
    p.add_argument("--embeddings-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--embedding-column", default=None,
                   help="pooling variant column (embedding_last/mean/max); "
                        "default: 'embedding' if present else embedding_last")
    args = p.parse_args()

    load_scale(args.scale, args.scale_config)  # validate the name early
    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)

    if os.path.exists(paths["benchmark"]):
        from src.benchmark_downstream import print_benchmark, run_benchmark

        summary = run_benchmark(
            benchmark_path=paths["benchmark"],
            output_dir=args.output_dir or paths["downstream"],
            embeddings_path=(
                (args.embeddings_path or paths["embeddings"])
                if args.with_embeddings
                else None
            ),
            device=args.device,
            embedding_column=args.embedding_column,
        )
        print_benchmark(summary)
    else:
        # Synthetic data: no NVIDIA benchmark to compare against.
        from src.downstream import print_summary, run_downstream

        summary = run_downstream(
            embeddings_path=paths["embeddings"],
            output_dir=paths["downstream"],
            raw_path=paths["raw"],
        )
        print_summary(summary)


if __name__ == "__main__":
    main()
