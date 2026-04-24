"""
Anyscale Job entrypoint for the e-commerce batch embedding pipeline.

Usage:
  python scripts/run_pipeline.py --scale medium
  python scripts/run_pipeline.py --scale large --num-gpus 4

Submit as Anyscale Job:
  anyscale job submit --config-file job_config.yaml
"""
import argparse
import sys
import os

# Make src importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray

from src.generate_data import generate_catalog, SCALE_MAP
from src.pipeline import run_embedding_pipeline

BASE_DIR = "/mnt/cluster_storage/ecommerce-demo"

SCALE_PATHS = {
    "small":  (f"{BASE_DIR}/raw/products_small.parquet",  f"{BASE_DIR}/embeddings/small/"),
    "medium": (f"{BASE_DIR}/raw/products_medium.parquet", f"{BASE_DIR}/embeddings/medium/"),
    "large":  (f"{BASE_DIR}/raw/products_large.parquet",  f"{BASE_DIR}/embeddings/large/"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="E-Commerce Batch Embedding Pipeline")
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="medium",
        help="small=1K, medium=100K, large=2M products",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPU workers for embedding concurrency",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Override input path (skips data generation)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Connect to Anyscale cluster (no-op if Ray is already initialized)
    ray.init(ignore_reinit_error=True)
    print(f"Ray cluster resources: {ray.cluster_resources()}")

    default_input, default_output = SCALE_PATHS[args.scale]
    input_path = args.input or default_input
    output_path = args.output or default_output

    # Generate synthetic data if input doesn't already exist
    if not os.path.exists(input_path):
        generate_catalog(SCALE_MAP[args.scale], input_path)
    else:
        print(f"Using existing data at {input_path}")

    # Run pipeline
    metrics = run_embedding_pipeline(
        input_path=input_path,
        output_path=output_path,
        num_gpus=args.num_gpus,
    )

    print("Job complete.")
    return metrics


if __name__ == "__main__":
    main()
