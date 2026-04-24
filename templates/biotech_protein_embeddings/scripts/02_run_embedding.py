"""
Anyscale Job entrypoint for the protein embedding pipeline.

Usage:
  python scripts/02_run_embedding.py --scale medium --bucketed
  python scripts/02_run_embedding.py --scale large --num-gpus 4 --bucketed
  python scripts/02_run_embedding.py --scale small --no-bucketed  # naive mode for comparison

Submit as Anyscale Job:
  anyscale job submit --config-file job_config.yaml
"""
import argparse
import sys
import os

# Make src importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray

from src.corpus_generator import save_corpus, SCALE_MAP
from src.pipeline import run_embedding_pipeline_naive, run_embedding_pipeline_bucketed

BASE_DIR = "/mnt/cluster_storage/protein-embeddings"

SCALE_PATHS = {
    "small":  (f"{BASE_DIR}/raw", f"{BASE_DIR}/embeddings/small/"),
    "medium": (f"{BASE_DIR}/raw", f"{BASE_DIR}/embeddings/medium/"),
    "large":  (f"{BASE_DIR}/raw", f"{BASE_DIR}/embeddings/large/"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Protein Embedding Pipeline")
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="medium",
        help="small=10K, medium=100K, large=500K sequences",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPU workers for ESM-2 embedding concurrency",
    )
    parser.add_argument(
        "--model",
        default="facebook/esm2_t33_650M_UR50D",
        help="HuggingFace ESM-2 model name",
    )
    parser.add_argument(
        "--bucketed",
        action="store_true",
        default=True,
        help="Enable length bucketing (default: True)",
    )
    parser.add_argument(
        "--no-bucketed",
        action="store_true",
        help="Disable length bucketing (naive mode)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Override input directory (skips data generation)",
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

    default_input_dir, default_output = SCALE_PATHS[args.scale]
    input_dir = args.input or default_input_dir
    output_path = args.output or default_output
    input_path = os.path.join(input_dir, "corpus.parquet")
    taxonomy_path = os.path.join(input_dir, "taxonomy_lookup.parquet")

    # Generate synthetic data if input doesn't already exist
    if not os.path.exists(input_path):
        print("Corpus not found — generating...")
        save_corpus(input_dir, num_sequences=SCALE_MAP[args.scale])
    else:
        print(f"Using existing corpus at {input_path}")

    # Determine mode
    use_bucketed = not args.no_bucketed

    if use_bucketed:
        metrics = run_embedding_pipeline_bucketed(
            input_path=input_path,
            output_path=output_path,
            taxonomy_path=taxonomy_path,
            num_gpus=args.num_gpus,
            model_name=args.model,
        )
    else:
        metrics = run_embedding_pipeline_naive(
            input_path=input_path,
            output_path=output_path,
            taxonomy_path=taxonomy_path,
            num_gpus=args.num_gpus,
            model_name=args.model,
        )

    print("Job complete.")
    return metrics


if __name__ == "__main__":
    main()
