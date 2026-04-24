"""
Step 2: Run the Boltz-1 screening pipeline on Ray Data.

Usage:
  python scripts/02_run_screening.py --scale medium --num-gpus 4
  python scripts/02_run_screening.py --scale large --num-gpus 8 --complex-type pl

Submit as Anyscale Job:
  anyscale job submit --config-file job_config.yaml
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray

from src.candidate_generator import generate_candidates, SCALE_MAP
from src.pipeline import run_screening_pipeline

BASE_DIR = "/mnt/cluster_storage/boltz-screening"


def main():
    parser = argparse.ArgumentParser(description="Boltz-1 Protein Screening Pipeline")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium",
                        help="small=50, medium=500, large=2000 candidates")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="Number of GPU workers for Boltz-1 inference")
    parser.add_argument("--complex-type", choices=["pp", "pl"], default="pp",
                        help="pp=protein-protein, pl=protein-ligand")
    parser.add_argument("--input", default=None, help="Override input path")
    parser.add_argument("--output", default=None, help="Override output path")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    print(f"Ray cluster resources: {ray.cluster_resources()}")

    input_path = args.input or f"{BASE_DIR}/candidates/{args.scale}_{args.complex_type}.parquet"
    output_path = args.output or f"{BASE_DIR}/results/{args.scale}/"

    # Generate candidate data if not present
    if not os.path.exists(input_path):
        print("Candidate data not found — generating...")
        generate_candidates(
            input_path,
            num_candidates=SCALE_MAP[args.scale],
            complex_type=args.complex_type,
        )

    metrics = run_screening_pipeline(
        candidates_path=input_path,
        output_path=output_path,
        num_gpus=args.num_gpus,
    )

    print("Job complete.")
    return metrics


if __name__ == "__main__":
    main()
