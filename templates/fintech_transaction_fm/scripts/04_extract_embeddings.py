"""Step 4 — batch embedding extraction with Ray Data."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray  # noqa: E402

from src.embed import extract_embeddings  # noqa: E402
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
    # Overrides for scoring an alternate checkpoint / pooling side-by-side
    # without touching the canonical model/embeddings dirs.
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--pooling", default="last", choices=["last", "mean"])
    p.add_argument("--output", default=None)
    p.add_argument("--readout", default="pooled", choices=["pooled", "target"],
                   help="target = Run-1 readout surgery: masked-target state + "
                        "per-field surprise + single-txn embedding (TEARDOWN.md)")
    p.add_argument("--limit", type=int, default=None,
                   help="embed only the first N eval rows — subset e2e proofs")
    args = p.parse_args()

    base = args.base_dir or get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    embed_cfg = load_scale(args.scale, args.scale_config)["embed"]

    ray.init(ignore_reinit_error=True)
    ds = None
    if args.limit:
        ds = ray.data.read_parquet(paths["tokenized_eval"]).limit(args.limit)
    extract_embeddings(
        ds=ds,
        tokenized_path=paths["tokenized_eval"],
        checkpoint_dir=args.checkpoint_dir or paths["checkpoint"],
        output_path=args.output or paths["embeddings"],
        num_workers=args.num_workers or embed_cfg["num_workers"],
        use_gpu=args.use_gpu or embed_cfg["use_gpu"],
        batch_size=args.batch_size or embed_cfg["batch_size"],
        pooling=args.pooling,
        gpus_per_worker=embed_cfg.get("gpus_per_worker"),
        readout=args.readout,
    )


if __name__ == "__main__":
    main()
