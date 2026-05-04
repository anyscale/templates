"""
Step 1: Generate synthetic transaction data.

Usage:
  python scripts/01_generate_data.py --scale medium
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import save_dataset, SCALE_MAP
from src.paths import get_demo_base_dir

BASE_DIR = get_demo_base_dir()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud detection data")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or f"{BASE_DIR}/raw/{args.scale}"
    paths = save_dataset(output_dir, num_transactions=SCALE_MAP[args.scale])
    print(f"\nData generation complete. {paths['num_transactions']:,} transactions written.")


if __name__ == "__main__":
    main()
