"""
Step 3: Run the Ray Data fraud scoring pipeline.

Usage:
  python scripts/03_run_scoring.py --scale medium
  python scripts/03_run_scoring.py --scale large --num-workers 20

Submit as Anyscale Job (job_config.yaml runs generate → train → score on a fresh cluster):
  anyscale job submit --config-file job_config.yaml
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray

from src.data_generator import save_dataset, SCALE_MAP
from src.pipeline import run_fraud_scoring_pipeline
from src.paths import get_demo_base_dir

BASE_DIR = get_demo_base_dir()


def main():
    parser = argparse.ArgumentParser(description="Fraud Risk Scoring Scoring Pipeline")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    print(f"Ray cluster resources: {ray.cluster_resources()}")

    input_dir = args.input_dir or f"{BASE_DIR}/raw/{args.scale}"
    output_path = args.output or f"{BASE_DIR}/scored/{args.scale}/"
    model_path = args.model or f"{BASE_DIR}/model/fraud_model.json"
    user_features_path = os.path.join(input_dir, "user_aggregates.parquet")
    merchant_features_path = os.path.join(input_dir, "merchant_aggregates.parquet")
    txn_path = os.path.join(input_dir, "transactions.parquet")

    # Generate data if not present
    if not os.path.exists(txn_path):
        print("Data not found — generating...")
        save_dataset(input_dir, num_transactions=SCALE_MAP[args.scale])

    metrics = run_fraud_scoring_pipeline(
        input_path=txn_path,
        output_path=output_path,
        model_path=model_path,
        user_features_path=user_features_path,
        merchant_features_path=merchant_features_path,
        num_workers=args.num_workers,
    )

    print("Job complete.")
    return metrics


if __name__ == "__main__":
    main()
