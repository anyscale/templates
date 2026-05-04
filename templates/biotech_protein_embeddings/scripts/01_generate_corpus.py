"""
Step 1: Generate synthetic protein FASTA corpus + taxonomy + homolog test pairs.

Usage:
  python scripts/01_generate_corpus.py --scale medium
  python scripts/01_generate_corpus.py --scale large --output-dir /mnt/cluster_storage/protein-embeddings/raw
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corpus_generator import save_corpus, SCALE_MAP


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic protein corpus")
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="medium",
        help="small=10K, medium=100K, large=500K sequences",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/cluster_storage/protein-embeddings/raw",
        help="Output directory for corpus files",
    )
    args = parser.parse_args()

    print(f"Generating {args.scale} corpus ({SCALE_MAP[args.scale]:,} sequences)...")
    result = save_corpus(
        output_dir=args.output_dir,
        num_sequences=SCALE_MAP[args.scale],
    )
    print(f"\nCorpus generation complete. Files at {args.output_dir}")
    return result


if __name__ == "__main__":
    main()
