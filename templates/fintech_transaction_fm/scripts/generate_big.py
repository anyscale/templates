"""Generate a large synthetic transaction dataset with Ray Data.

Used by the scaling deep-dive (Part 9). Single-node generation OOMs well before
the hundreds of millions / billions of rows needed to actually stress
multi-node execution, so this maps the per-card generator across the cluster and
writes sharded Parquet. Run it on a workspace or as an Anyscale Job:

    python scripts/generate_big.py --num-cards 1000000

The cluster autoscales CPU nodes to generate, then scales back to zero.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray  # noqa: E402

from src.generate_data import generate_dataset_distributed  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-cards", type=int, default=1_000_000)
    p.add_argument(
        "--out",
        default="/mnt/cluster_storage/transaction-fm/scale/raw.parquet",
        help="output Parquet directory (sharded)",
    )
    p.add_argument(
        "--num-blocks",
        type=int,
        default=512,
        help="Ray Data blocks — more blocks = more parallelism across the cluster",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    splits_out = os.path.join(os.path.dirname(args.out.rstrip("/")), "splits.json")
    ray.init(ignore_reinit_error=True)
    generate_dataset_distributed(
        args.out,
        splits_out,
        num_cards=args.num_cards,
        seed=args.seed,
        num_blocks=args.num_blocks,
    )


if __name__ == "__main__":
    main()
