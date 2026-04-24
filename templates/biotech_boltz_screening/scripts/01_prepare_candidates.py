"""
Step 1: Generate synthetic candidate complexes for Boltz-1 screening.

Usage:
  python scripts/01_prepare_candidates.py --scale medium --complex-type pp
  python scripts/01_prepare_candidates.py --scale large --complex-type pl
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.candidate_generator import generate_candidates, SCALE_MAP

BASE_DIR = "/mnt/cluster_storage/boltz-screening"


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic screening candidates")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--complex-type", choices=["pp", "pl"], default="pp",
                        help="pp=protein-protein, pl=protein-ligand")
    parser.add_argument("--output", default=None, help="Override output path")
    args = parser.parse_args()

    num_candidates = SCALE_MAP[args.scale]
    output_path = args.output or f"{BASE_DIR}/candidates/{args.scale}_{args.complex_type}.parquet"

    print(f"Generating {num_candidates:,} {args.complex_type} candidates (scale={args.scale})")
    generate_candidates(output_path, num_candidates, args.complex_type)
    print(f"\nCandidate generation complete.")


if __name__ == "__main__":
    main()
